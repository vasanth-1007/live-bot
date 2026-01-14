from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .rag.retriever import WeaviateRetriever
from .rag.prompts import SYSTEM_INSTRUCTION, build_user_prompt
from .live.gemini_live import GeminiLiveTextSession

load_dotenv()
settings = Settings()

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

app = FastAPI(title="Gemini Live + Weaviate RAG (Tanglish)")
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

    async with GeminiLiveTextSession(
        model=settings.gemini_live_model,
        system_instruction=SYSTEM_INSTRUCTION,
    ) as live:
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)

                if msg.get("type") != "user_message":
                    await ws.send_text(json.dumps({"type": "error", "message": "Unknown message type"}))
                    continue

                user_text = (msg.get("text") or "").strip()
                if not user_text:
                    continue

                # 1) Retrieve from YOUR existing Weaviate
                chunks = retriever.retrieve(user_text, top_k=settings.top_k)

                # Strongest enforcement: if no context, we do NOT call Gemini.
                if not chunks:
                    await ws.send_text(json.dumps({
                        "type": "assistant_done",
                        "text": "Indha kelvikku thevaiyana thagaval enakku Weaviate-la kidaikkala.",
                        "sources": [],
                    }))
                    continue

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

                full = []
                async for delta in live.stream_answer(prompt):
                    full.append(delta)
                    await ws.send_text(json.dumps({"type": "assistant_delta", "delta": delta}))

                await ws.send_text(json.dumps({
                    "type": "assistant_done",
                    "text": "".join(full).strip(),
                    "sources": sources_payload,
                }))

        except WebSocketDisconnect:
            return
