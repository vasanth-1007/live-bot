"""
Gemini Live + Weaviate RAG Server
Supports both text and voice (audio) input/output
"""

import asyncio
import json
import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import weaviate
from weaviate.classes.init import Auth

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

from server.config import Settings

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("gemini-live-rag")

# ─────────────────────────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────────────────────────
settings = Settings()
weaviate_client: weaviate.WeaviateClient | None = None
gemini_client: genai.Client | None = None

WEB_DIR = Path(__file__).parent / "web"

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful voice assistant that answers questions using ONLY the provided context from our knowledge base.

IMPORTANT RULES:
1. Answer ONLY based on the context provided below
2. If the context doesn't contain the answer, say "I don't have information about that in my knowledge base"
3. Keep responses concise and conversational since this is voice output
4. Speak naturally as if having a conversation

CONTEXT FROM KNOWLEDGE BASE:
{context}

Now answer the user's question based solely on the above context."""


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan (startup/shutdown)
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global weaviate_client, gemini_client

    # ── Weaviate ──
    logger.info("Connecting to Weaviate...")
    try:
        if settings.weaviate_url:
            # Cloud Connection
            weaviate_client = weaviate.connect_to_weaviate_cloud(
                cluster_url=settings.weaviate_url,
                auth_credentials=Auth.api_key(settings.weaviate_api_key)
                if settings.weaviate_api_key else None,
            )
        else:
            # Custom / Local Connection
            weaviate_client = weaviate.connect_to_custom(
                http_host=settings.http_host,
                http_port=settings.http_port,
                http_secure=settings.http_secure,
                grpc_host=settings.grpc_host,
                grpc_port=settings.grpc_port,
                grpc_secure=settings.grpc_secure,
                auth_credentials=Auth.api_key(settings.weaviate_api_key) 
                if settings.weaviate_api_key else None,
            )
        logger.info("Weaviate connected ✓")
    except Exception as e:
        logger.error(f"Weaviate connection failed: {e}")
        # Consider whether to raise here or allow app to start without RAG
        raise

    # ── Gemini ──
    logger.info("Initializing Gemini client...")
    gemini_client = genai.Client()
    logger.info(f"Gemini client ready, model: {settings.gemini_live_model}")

    yield

    # ── Cleanup ──
    if weaviate_client:
        weaviate_client.close()
        logger.info("Weaviate closed")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Gemini Live RAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


# ─────────────────────────────────────────────────────────────────────────────
# RAG: Retrieve from Weaviate
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_context(query: str) -> tuple[str, list[dict]]:
    """Hybrid search in Weaviate using client-side embeddings."""
    if not weaviate_client or not gemini_client:
        return "", []

    try:
        # FIX 1: Generate embedding for the query because Weaviate has no vectorizer configured
        embed_resp = gemini_client.models.embed_content(
            model="text-embedding-004",
            contents=query
        )
        query_vector = embed_resp.embeddings[0].values

        collection = weaviate_client.collections.get(settings.weaviate_collection)
        
        # Build query arguments
        query_kwargs = {
            "query": query,
            "vector": query_vector,  # Pass the generated vector
            "limit": settings.top_k,
            "return_metadata": ["score", "explain_score"],
        }
        
        if settings.weaviate_tenant:
            collection = collection.with_tenant(settings.weaviate_tenant)
        
        if settings.weaviate_target_vector:
            query_kwargs["target_vector"] = settings.weaviate_target_vector

        results = collection.query.hybrid(**query_kwargs)

        chunks = []
        sources = []
        total_chars = 0

        for obj in results.objects:
            text = obj.properties.get(settings.weaviate_text_property, "")
            if total_chars + len(text) > settings.max_context_chars:
                break
            chunks.append(text)
            total_chars += len(text)

            source_info = {
                "id": str(obj.uuid),
                "text_preview": text[:200] + "..." if len(text) > 200 else text,
                "properties": {
                    k: v for k, v in obj.properties.items()
                    if k in settings.extra_properties
                },
            }
            if obj.metadata and obj.metadata.score is not None:
                source_info["score"] = round(obj.metadata.score, 4)
            sources.append(source_info)

        context = "\n\n---\n\n".join(chunks) if chunks else "(No relevant documents found)"
        logger.info(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")
        return context, sources

    except Exception as e:
        logger.error(f"Weaviate retrieval error: {e}")
        return "(Error retrieving context)", []


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Chat Handler
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket connected")

    audio_buffer = bytearray()
    is_recording = False
    input_sample_rate = 16000

    try:
        while True:
            message = await ws.receive()

            if "bytes" in message:
                if is_recording:
                    audio_buffer.extend(message["bytes"])
                continue

            if "text" in message:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = data.get("type")

                if msg_type == "audio_start":
                    is_recording = True
                    audio_buffer.clear()
                    input_sample_rate = data.get("sample_rate_hz", 16000)
                    logger.info(f"Audio recording started, sample_rate={input_sample_rate}")

                elif msg_type == "audio_end":
                    is_recording = False
                    logger.info(f"Audio recording ended, {len(audio_buffer)} bytes")
                    
                    if len(audio_buffer) > 0:
                        await handle_audio_turn(ws, bytes(audio_buffer), input_sample_rate)
                    audio_buffer.clear()

                elif msg_type == "user_message":
                    text = data.get("text", "").strip()
                    if text:
                        await handle_text_turn(ws, text)

                elif msg_type == "ping":
                    await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}\n{traceback.format_exc()}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Handle Text Input
# ─────────────────────────────────────────────────────────────────────────────
async def handle_text_turn(ws: WebSocket, user_text: str):
    logger.info(f"Text turn: {user_text[:100]}...")

    context, sources = retrieve_context(user_text)
    system_instruction = SYSTEM_PROMPT.format(context=context)

    await ws.send_json({"type": "assistant_start"})

    # FIX 2: For Text-to-Audio, use AUDIO modality + output_audio_transcription
    config = {
        "response_modalities": ["AUDIO"],  # ONLY AUDIO for Native Audio model
        "output_audio_transcription": {},  # Get text back via transcription
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}
        },
        "system_instruction": {"parts": [{"text": system_instruction}]},
    }

    full_text = ""
    
    try:
        async with gemini_client.aio.live.connect(
            model=settings.gemini_live_model,
            config=config,
        ) as session:
            await session.send(
                input=types.LiveClientContent(
                    turns=[
                        types.Content(role="user", parts=[types.Part(text=user_text)])
                    ],
                    turn_complete=True,
                ),
                end_of_turn=True,
            )

            async for response in session.receive():
                if response.server_content:
                    # Handle Audio
                    if response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                await ws.send_json({
                                    "type": "assistant_audio_format",
                                    "mime_type": part.inline_data.mime_type or "audio/pcm;rate=24000"
                                })
                                await ws.send_bytes(part.inline_data.data)

                    # Handle Text (Output Transcription)
                    if response.server_content.output_transcription:
                        ot = response.server_content.output_transcription
                        if ot.text:
                            full_text += ot.text
                            await ws.send_json({
                                "type": "assistant_delta",
                                "delta": ot.text
                            })

                    if response.server_content.turn_complete:
                        break

        await ws.send_json({
            "type": "assistant_done",
            "text": full_text or "(Voice response sent)",
            "sources": sources,
        })

    except Exception as e:
        logger.error(f"Gemini error: {e}\n{traceback.format_exc()}")
        await ws.send_json({
            "type": "assistant_done",
            "text": f"Error: {str(e)}",
            "sources": sources,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Handle Audio Input
# ─────────────────────────────────────────────────────────────────────────────
async def handle_audio_turn(ws: WebSocket, audio_data: bytes, sample_rate: int):
    logger.info(f"Audio turn: {len(audio_data)} bytes at {sample_rate}Hz")
    await ws.send_json({"type": "assistant_start"})

    # --- Step 1: Transcription ---
    transcribed_text = ""
    try:
        transcribe_config = {
            "response_modalities": ["AUDIO"], 
            "input_audio_transcription": {}, 
            "system_instruction": "You are a transcriber. Do not speak. Wait for the next turn."
        }

        async with gemini_client.aio.live.connect(
            model=settings.gemini_live_model,
            config=transcribe_config,
        ) as session:
            await session.send(
                input=types.LiveClientRealtimeInput(
                    media_chunks=[
                        types.Blob(
                            mime_type=f"audio/pcm;rate={sample_rate}",
                            data=audio_data,
                        )
                    ]
                ),
                end_of_turn=True,
            )

            async for response in session.receive():
                if response.server_content:
                    if hasattr(response.server_content, "input_transcription"):
                        it = response.server_content.input_transcription
                        if it and it.text:
                            transcribed_text += it.text
                if response.server_content and response.server_content.turn_complete:
                    break

        logger.info(f"Transcribed: {transcribed_text[:100]}...")
        await ws.send_json({"type": "user_transcript", "text": transcribed_text})

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        transcribed_text = ""

    if not transcribed_text.strip():
        await ws.send_json({
            "type": "assistant_done",
            "text": "I couldn't understand the audio. Please try again.",
            "sources": [],
        })
        return

    # --- Step 2: RAG Retrieval ---
    context, sources = retrieve_context(transcribed_text)
    system_instruction = SYSTEM_PROMPT.format(context=context)

    # --- Step 3: Response Generation ---
    full_text = ""
    try:
        # FIX 3: Use AUDIO modality + output_audio_transcription to avoid 1007 Error
        response_config = {
            "response_modalities": ["AUDIO"],
            "output_audio_transcription": {},  # Required for getting text back
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}
            },
            "system_instruction": {"parts": [{"text": system_instruction}]},
        }

        async with gemini_client.aio.live.connect(
            model=settings.gemini_live_model,
            config=response_config,
        ) as session:
            await session.send(
                input=types.LiveClientContent(
                    turns=[
                        types.Content(
                            role="user",
                            parts=[types.Part(text=transcribed_text)]
                        )
                    ],
                    turn_complete=True,
                ),
                end_of_turn=True,
            )

            async for response in session.receive():
                if response.server_content:
                    # Handle Audio
                    if response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                await ws.send_json({
                                    "type": "assistant_audio_format",
                                    "mime_type": part.inline_data.mime_type or "audio/pcm;rate=24000"
                                })
                                await ws.send_bytes(part.inline_data.data)
                    
                    # Handle Text (from output transcription)
                    if response.server_content.output_transcription:
                        ot = response.server_content.output_transcription
                        if ot.text:
                            full_text += ot.text
                            await ws.send_json({
                                "type": "assistant_delta",
                                "delta": ot.text
                            })

                    if response.server_content.turn_complete:
                        break

        await ws.send_json({
            "type": "assistant_done",
            "text": full_text or "(Voice response sent)",
            "sources": sources,
        })

    except Exception as e:
        logger.error(f"Response generation error: {e}\n{traceback.format_exc()}")
        await ws.send_json({
            "type": "assistant_done",
            "text": f"Error generating response: {str(e)}",
            "sources": sources,
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
    )
