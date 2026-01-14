from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .rag.retriever import WeaviateRetriever
from .rag.prompts import SYSTEM_INSTRUCTION, build_user_prompt
from .live.gemini_live import (
    GeminiLiveAudioSession,
    GeminiLiveTranscribeSession,
    extract_audio_bytes,
    extract_input_transcript,
    extract_output_transcript,
    is_turn_complete,
    delta_from_full,
)

load_dotenv()
settings = Settings()

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

app = FastAPI(title="Gemini Live (Voice) + Weaviate RAG (Tanglish)")
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

retriever: WeaviateRetriever | None = None


@app.on_event("startup")
def _startup():
    global retriever
    retriever = WeaviateRetriever(
        collection_name=settings.weaviate_collection,
        text_property=settings.weaviate_text_property,
        extra_properties=settings.extra_properties,
        tenant=settings.weaviate_tenant,
        target_vector=settings.weaviate_target_vector,
        weaviate_url=settings.weaviate_url,
        weaviate_api_key=settings.weaviate_api_key,
        http_host=settings.http_host,
        http_port=settings.http_port,
        http_secure=settings.http_secure,
        grpc_host=settings.grpc_host,
        grpc_port=settings.grpc_port,
        grpc_secure=settings.grpc_secure,
    )


@app.on_event("shutdown")
def _shutdown():
    global retriever
    if retriever:
        retriever.close()
        retriever = None


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()

    if retriever is None:
        await ws.send_text(json.dumps({"type": "error", "message": "Retriever not initialized"}))
        await ws.close()
        return

    # One long-lived AUDIO session for answering (TTS output).
    async with GeminiLiveAudioSession(
        model=settings.gemini_live_model,
        system_instruction=SYSTEM_INSTRUCTION,
        voice_name=getattr(settings, "gemini_voice_name", "Kore"),
    ) as answer_session:
        try:
            # Tell the client what audio format to expect from the assistant.
            await ws.send_text(json.dumps({
                "type": "assistant_audio_format",
                "mime_type": "audio/pcm;rate=24000",
                "note": "Raw PCM16 mono @ 24kHz, little-endian"
            }))

            while True:
                incoming = await ws.receive()

                # --- 1) Binary frames: we only accept them inside an audio turn.
                if incoming.get("bytes") is not None:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": "Unexpected binary frame. Send {type: audio_start} first."
                    }))
                    continue

                # --- 2) Text frames are JSON control/messages.
                raw_text = incoming.get("text")
                if not raw_text:
                    continue

                msg = json.loads(raw_text)
                mtype = msg.get("type")

                # A) Text chat (existing behavior) BUT now assistant responds in AUDIO (+ transcript deltas)
                if mtype == "user_message":
                    user_text = (msg.get("text") or "").strip()
                    if not user_text:
                        continue
                    await _handle_text_turn(ws, answer_session, user_text)

                # B) Audio chat: client sends audio_start, then binary PCM chunks, then audio_end
                elif mtype == "audio_start":
                    sample_rate = int(msg.get("sample_rate_hz") or 16000)
                    await _handle_audio_turn(ws, answer_session, sample_rate_hz=sample_rate)

                else:
                    await ws.send_text(json.dumps({"type": "error", "message": "Unknown message type"}))

        except WebSocketDisconnect:
            return


async def _handle_text_turn(ws: WebSocket, answer_session: GeminiLiveAudioSession, user_text: str) -> None:
    assert retriever is not None

    # 1) Retrieve from Weaviate
    chunks = retriever.retrieve(user_text, top_k=settings.top_k)

    if not chunks:
        await ws.send_text(json.dumps({
            "type": "assistant_done",
            "text": "Indha kelvikku thevaiyana thagaval enakku Weaviate-la kidaikkala.",
            "sources": [],
        }))
        return

    prompt = build_user_prompt(user_text, chunks, settings.max_context_chars)

    sources_payload = []
    for i, ch in enumerate(chunks, start=1):
        sources_payload.append({
            "id": f"S{i}",
            "text_preview": ch.text[:300],
            "properties": ch.properties,
            "score": ch.score,
            "distance": ch.distance,
        })

    await ws.send_text(json.dumps({"type": "assistant_start"}))

    # 2) Ask Gemini (AUDIO response)
    await answer_session.send_text_turn(prompt)

    # 3) Stream back:
    #    - audio bytes as binary frames
    #    - output transcription deltas as JSON (so your UI can show text)
    full_out = ""
    async for live_msg in answer_session.receive():
        audio = extract_audio_bytes(live_msg)
        if audio:
            await ws.send_bytes(audio)

        out_t = extract_output_transcript(live_msg)
        if out_t:
            delta = delta_from_full(full_out, out_t)
            if delta:
                full_out = out_t
                await ws.send_text(json.dumps({"type": "assistant_delta", "delta": delta}))

        if is_turn_complete(live_msg):
            break

    await ws.send_text(json.dumps({
        "type": "assistant_done",
        "text": full_out.strip(),
        "sources": sources_payload,
    }))


async def _handle_audio_turn(ws: WebSocket, answer_session: GeminiLiveAudioSession, *, sample_rate_hz: int) -> None:
    """
    Protocol:
      1) client -> {"type":"audio_start","sample_rate_hz":16000}
      2) client -> (binary PCM16 chunks)
      3) client -> {"type":"audio_end"}

    We:
      - run a transcription Live session while receiving chunks
      - stream partial transcripts to client
      - once ended, run the normal RAG flow using the final transcript text
    """
    assert retriever is not None

    await ws.send_text(json.dumps({
        "type": "audio_ack",
        "sample_rate_hz": sample_rate_hz,
        "expected_mime_type": f"audio/pcm;rate={sample_rate_hz}",
        "note": "Send raw PCM16 mono little-endian chunks as binary frames, then {type: audio_end}."
    }))

    transcript_full = ""

    async with GeminiLiveTranscribeSession(model=settings.gemini_live_model) as stt:
        mic_done = asyncio.Event()

        async def ws_to_stt():
            nonlocal transcript_full
            try:
                while True:
                    incoming = await ws.receive()
                    if incoming.get("bytes") is not None:
                        await stt.send_audio_chunk(incoming["bytes"], sample_rate_hz=sample_rate_hz)
                        continue

                    raw_text = incoming.get("text")
                    if not raw_text:
                        continue
                    ctrl = json.loads(raw_text)
                    if ctrl.get("type") == "audio_end":
                        mic_done.set()
                        await stt.audio_stream_end()
                        return
                    # Ignore other messages during recording.
            except WebSocketDisconnect:
                mic_done.set()
                raise

        async def stt_to_ws():
            nonlocal transcript_full
            last_sent = ""
            async for live_msg in stt.receive():
                t = extract_input_transcript(live_msg)
                if t:
                    transcript_full = t
                    delta = delta_from_full(last_sent, t)
                    if delta:
                        last_sent = t
                        await ws.send_text(json.dumps({"type": "user_transcript_delta", "delta": delta}))

                # Once mic is done, we can exit when the turn is complete.
                if mic_done.is_set() and is_turn_complete(live_msg):
                    return

        await asyncio.gather(ws_to_stt(), stt_to_ws())

    final_text = (transcript_full or "").strip()
    if not final_text:
        await ws.send_text(json.dumps({"type": "error", "message": "No transcription produced."}))
        return

    # Now run the normal RAG+answer pipeline using the transcript as user_text
    await _handle_text_turn(ws, answer_session, final_text)
