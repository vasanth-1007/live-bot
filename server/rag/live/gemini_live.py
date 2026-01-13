from __future__ import annotations
from typing import AsyncIterator, Optional
from google import genai


class GeminiLiveTextSession:
    def __init__(self, *, model: str, system_instruction: str):
        self.model = model
        self.system_instruction = system_instruction
        self.client = genai.Client()
        self._cm = None
        self._session = None

    async def __aenter__(self):
        config = {
            "response_modalities": ["TEXT"],
            "system_instruction": self.system_instruction,
        }
        self._cm = self.client.aio.live.connect(model=self.model, config=config)
        self._session = await self._cm.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._cm is not None:
            await self._cm.__aexit__(exc_type, exc, tb)

    async def stream_answer(self, user_text: str) -> AsyncIterator[str]:
        """
        Live API text sending: send_client_content(turns=..., turn_complete=True). ([ai.google.dev](https://ai.google.dev/gemini-api/docs/live-guide))
        """
        s = self._session
        if s is None:
            raise RuntimeError("Live session not started")

        await s.send_client_content(turns=user_text, turn_complete=True)

        last_emitted = ""
        async for msg in s.receive():
            full = _extract_text(msg)
            if full:
                # de-dup / delta extraction
                if full.startswith(last_emitted):
                    delta = full[len(last_emitted):]
                else:
                    delta = full
                if delta:
                    last_emitted = full
                    yield delta

            if _is_turn_complete(msg):
                break


def _extract_text(msg) -> Optional[str]:
    # Some SDK shapes expose msg.text
    t = getattr(msg, "text", None)
    if isinstance(t, str) and t:
        return t

    sc = getattr(msg, "server_content", None)
    if not sc:
        return None

    mt = getattr(sc, "model_turn", None)
    if not mt:
        return None

    parts = getattr(mt, "parts", None) or []
    out = []
    for p in parts:
        txt = getattr(p, "text", None)
        if isinstance(txt, str) and txt:
            out.append(txt)
    return "".join(out) if out else None


def _is_turn_complete(msg) -> bool:
    sc = getattr(msg, "server_content", None)
    if not sc:
        return False
    # handle both possible attribute names
    return bool(getattr(sc, "turn_complete", False) or getattr(sc, "turnComplete", False))
