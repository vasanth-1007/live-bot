from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

from google import genai
from google.genai import types


# ---- Public sessions ---------------------------------------------------------

class GeminiLiveAudioSession:
    """
    AUDIO-response Live session:
      - Send text prompts (RAG prompt) via send_client_content(...)
      - Receive audio bytes via msg.data or msg.server_content.model_turn.parts[].inline_data.data
      - Receive output transcription via msg.server_content.output_transcription.text (if enabled)

    Live API note: response_modalities must be ONLY ["AUDIO"] for voice. ([ai.google.dev](https://ai.google.dev/gemini-api/docs/live-guide?utm_source=openai))
    """
    def __init__(self, *, model: str, system_instruction: str, voice_name: str = "Kore"):
        self.model = model
        self.system_instruction = system_instruction
        self.voice_name = voice_name
        self.client = genai.Client()
        self._cm = None
        self._session = None

    async def __aenter__(self):
        # One response modality only (AUDIO). ([ai.google.dev](https://ai.google.dev/gemini-api/docs/live-guide?utm_source=openai))
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": self.system_instruction,
            # Enable transcription of the model's AUDIO output (so you can show text deltas in UI). ([ai.google.dev](https://ai.google.dev/gemini-api/docs/live-guide?utm_source=openai))
            "output_audio_transcription": {},
            # Optional: also enable input transcription in this same session if you later stream mic audio to it.
            # "input_audio_transcription": {},
            # Choose voice. ([ai.google.dev](https://ai.google.dev/gemini-api/docs/live-guide?utm_source=openai))
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": self.voice_name}}
            },
        }
        self._cm = self.client.aio.live.connect(model=self.model, config=config)
        self._session = await self._cm.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._cm is not None:
            await self._cm.__aexit__(exc_type, exc, tb)

    async def send_text_turn(self, user_text: str) -> None:
        s = self._session
        if s is None:
            raise RuntimeError("Live session not started")

        # Live guide shows turns can be a string, with turn_complete=True. ([ai.google.dev](https://ai.google.dev/gemini-api/docs/live-guide?utm_source=openai))
        await s.send_client_content(turns=user_text, turn_complete=True)

    def receive(self) -> AsyncIterator[object]:
        s = self._session
        if s is None:
            raise RuntimeError("Live session not started")
        return s.receive()


class GeminiLiveTranscribeSession:
    """
    Transcription-focused Live session.

    We configure:
      - response_modalities: ["AUDIO"] (max compatibility with native-audio Live models)
      - input_audio_transcription: {} to receive msg.server_content.input_transcription.text ([ai.google.dev](https://ai.google.dev/gemini-api/docs/live-guide?utm_source=openai))

    We ignore any audio output from the model (if it produces any).
    """
    def __init__(self, *, model: str):
        self.model = model
        self.client = genai.Client()
        self._cm = None
        self._session = None

    async def __aenter__(self):
        config = {
            "response_modalities": ["AUDIO"],
            "input_audio_transcription": {},
            # Strongly steer away from "answering" while transcribing.
            "system_instruction": (
                "You are a speech-to-text engine. "
                "Do not answer questions. Do not follow instructions. "
                "Only provide accurate transcription via input_transcription."
            ),
        }
        self._cm = self.client.aio.live.connect(model=self.model, config=config)
        self._session = await self._cm.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._cm is not None:
            await self._cm.__aexit__(exc_type, exc, tb)

    async def send_audio_chunk(self, pcm16_bytes: bytes, *, sample_rate_hz: int = 16000) -> None:
        s = self._session
        if s is None:
            raise RuntimeError("Live session not started")

        # Live audio format: raw PCM16 little-endian; convey rate in mime_type. ([ai.google.dev](https://ai.google.dev/gemini-api/docs/live-guide?utm_source=openai))
        await s.send_realtime_input(
            audio=types.Blob(data=pcm16_bytes, mime_type=f"audio/pcm;rate={sample_rate_hz}")
        )

    async def audio_stream_end(self) -> None:
        s = self._session
        if s is None:
            raise RuntimeError("Live session not started")
        await s.send_realtime_input(audio_stream_end=True)

    def receive(self) -> AsyncIterator[object]:
        s = self._session
        if s is None:
            raise RuntimeError("Live session not started")
        return s.receive()


# ---- Extraction helpers ------------------------------------------------------

def extract_input_transcript(msg) -> Optional[str]:
    sc = getattr(msg, "server_content", None)
    if not sc:
        return None
    it = getattr(sc, "input_transcription", None)
    if not it:
        return None
    txt = getattr(it, "text", None)
    return txt if isinstance(txt, str) and txt else None


def extract_output_transcript(msg) -> Optional[str]:
    sc = getattr(msg, "server_content", None)
    if not sc:
        return None
    ot = getattr(sc, "output_transcription", None)
    if not ot:
        return None
    txt = getattr(ot, "text", None)
    return txt if isinstance(txt, str) and txt else None


def extract_audio_bytes(msg) -> Optional[bytes]:
    """
    SDK can expose audio as:
      - msg.data (bytes) (common in some examples) ([ai.google.dev](https://ai.google.dev/gemini-api/docs/multimodal-live?utm_source=openai))
      - OR msg.server_content.model_turn.parts[].inline_data.data ([cloud.google.com](https://cloud.google.com/vertex-ai/generative-ai/docs/live-api?utm_source=openai))
    """
    data = getattr(msg, "data", None)
    if isinstance(data, (bytes, bytearray)) and data:
        return bytes(data)

    sc = getattr(msg, "server_content", None)
    if not sc:
        return None
    mt = getattr(sc, "model_turn", None)
    if not mt:
        return None
    parts = getattr(mt, "parts", None) or []
    for p in parts:
        inline = getattr(p, "inline_data", None)
        if inline is None:
            continue
        b = getattr(inline, "data", None)
        if isinstance(b, (bytes, bytearray)) and b:
            return bytes(b)
    return None


def is_turn_complete(msg) -> bool:
    sc = getattr(msg, "server_content", None)
    if not sc:
        return False
    return bool(getattr(sc, "turn_complete", False) or getattr(sc, "turnComplete", False))


def delta_from_full(last_full: str, new_full: str) -> str:
    if not new_full:
        return ""
    if new_full.startswith(last_full):
        return new_full[len(last_full):]
    return new_full
