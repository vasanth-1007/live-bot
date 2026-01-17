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
# ADDED: Import Auth for Weaviate v4 connection
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
            # UPDATED: Added auth_credentials here to fix 401 error
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
    # UPDATED: Ensure API key is picked up correctly if env var names conflict
    # (Optional: explicitly pass api_key=settings.google_api_key if needed)
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
    """Hybrid search in Weaviate, return context string and source metadata."""
    if not weaviate_client:
        return "", []

    try:
        collection = weaviate_client.collections.get(settings.weaviate_collection)
        
        # Build query arguments
        query_kwargs = {
            "query": query,
            "limit": settings.top_k,
            "return_metadata": ["score", "explain_score"],
        }
        
        # Add tenant if configured
        if settings.weaviate_tenant:
            collection = collection.with_tenant(settings.weaviate_tenant)
        
        # Add target vector if configured
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

            # Build source info
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

    # Audio buffer for collecting voice input
    audio_buffer = bytearray()
    is_recording = False
    input_sample_rate = 16000

    try:
        while True:
            message = await ws.receive()

            # Handle binary audio data
            if "bytes" in message:
                if is_recording:
                    audio_buffer.extend(message["bytes"])
                continue

            # Handle text/JSON messages
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = data.get("type")

                if msg_type == "audio_start":
                    # User started recording
                    is_recording = True
                    audio_buffer.clear()
                    input_sample_rate = data.get("sample_rate_hz", 16000)
                    logger.info(f"Audio recording started, sample_rate={input_sample_rate}")

                elif msg_type == "audio_end":
                    # User stopped recording - process the audio
                    is_recording = False
                    logger.info(f"Audio recording ended, {len(audio_buffer)} bytes")
                    
                    if len(audio_buffer) > 0:
                        await handle_audio_turn(
                            ws, 
                            bytes(audio_buffer), 
                            input_sample_rate
                        )
                    audio_buffer.clear()

                elif msg_type == "user_message":
                    # Text message from user
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
    """Process text input, retrieve context, get response with audio."""
    logger.info(f"Text turn: {user_text[:100]}...")

    # Retrieve context from Weaviate
    context, sources = retrieve_context(user_text)
    system_instruction = SYSTEM_PROMPT.format(context=context)

    # Send start signal
    await ws.send_json({"type": "assistant_start"})

    # Configure for audio output
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO", "TEXT"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Kore"
                )
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part(text=system_instruction)]
        ),
    )

    full_text = ""
    
    try:
        async with gemini_client.aio.live.connect(
            model=settings.gemini_live_model,
            config=config,
        ) as session:
            # Send user message
            await session.send(
                input=types.LiveClientContent(
                    turns=[
                        types.Content(
                            role="user",
                            parts=[types.Part(text=user_text)]
                        )
                    ],
                    turn_complete=True,
                ),
                end_of_turn=True,
            )

            # Receive response
            async for response in session.receive():
                # Handle server content (text and audio)
                if response.server_content:
                    if response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            # Text part
                            if part.text:
                                full_text += part.text
                                await ws.send_json({
                                    "type": "assistant_delta",
                                    "delta": part.text
                                })
                            
                            # Audio part (inline_data)
                            if part.inline_data and part.inline_data.data:
                                # Send audio format info first time
                                await ws.send_json({
                                    "type": "assistant_audio_format",
                                    "mime_type": part.inline_data.mime_type or "audio/pcm;rate=24000"
                                })
                                # Send binary audio
                                await ws.send_bytes(part.inline_data.data)

                    # Check if turn is complete
                    if response.server_content.turn_complete:
                        break

        # Send completion
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
    """Process audio input, transcribe, retrieve context, respond with audio."""
    logger.info(f"Audio turn: {len(audio_data)} bytes at {sample_rate}Hz")

    # First, we need to transcribe the audio to text for RAG retrieval
    # We'll use the Live API for this

    # Send start signal
    await ws.send_json({"type": "assistant_start"})

    # For RAG, we need to first get the transcription
    # We'll do a two-step process:
    # 1. Send audio to get transcription
    # 2. Use transcription for RAG lookup
    # 3. Send context + audio to get final response

    transcribed_text = ""
    
    # Step 1: Transcribe audio
    try:
        transcribe_config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            system_instruction=types.Content(
                parts=[types.Part(text="Transcribe the user's speech accurately. Only output the transcription, nothing else.")]
            ),
        )

        async with gemini_client.aio.live.connect(
            model=settings.gemini_live_model,
            config=transcribe_config,
        ) as session:
            # Send audio
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

            # Get transcription
            async for response in session.receive():
                if response.server_content and response.server_content.model_turn:
                    for part in response.server_content.model_turn.parts:
                        if part.text:
                            transcribed_text += part.text
                
                if response.server_content and response.server_content.turn_complete:
                    break

        logger.info(f"Transcribed: {transcribed_text[:100]}...")
        
        # Send transcription to client
        await ws.send_json({
            "type": "user_transcript",
            "text": transcribed_text
        })

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

    # Step 2: Retrieve context using transcribed text
    context, sources = retrieve_context(transcribed_text)
    system_instruction = SYSTEM_PROMPT.format(context=context)

    # Step 3: Generate response with audio
    full_text = ""
    
    try:
        response_config = types.LiveConnectConfig(
            response_modalities=["AUDIO", "TEXT"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Kore"
                    )
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=system_instruction)]
            ),
        )

        async with gemini_client.aio.live.connect(
            model=settings.gemini_live_model,
            config=response_config,
        ) as session:
            # Send the transcribed text as user input
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

            # Receive response
            async for response in session.receive():
                if response.server_content:
                    if response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.text:
                                full_text += part.text
                                await ws.send_json({
                                    "type": "assistant_delta",
                                    "delta": part.text
                                })
                            
                            if part.inline_data and part.inline_data.data:
                                await ws.send_json({
                                    "type": "assistant_audio_format",
                                    "mime_type": part.inline_data.mime_type or "audio/pcm;rate=24000"
                                })
                                await ws.send_bytes(part.inline_data.data)

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


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
    )