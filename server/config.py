import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class Settings:
    # Gemini Live
    gemini_live_model: str = os.getenv(
        "GEMINI_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"
    )

    # Weaviate schema mapping
    weaviate_collection: str = os.getenv("WEAVIATE_COLLECTION", "Docs")
    weaviate_text_property: str = os.getenv("WEAVIATE_TEXT_PROPERTY", "text")
    weaviate_extra_properties_csv: str = os.getenv("WEAVIATE_EXTRA_PROPERTIES", "")

    # Weaviate tenancy / named vectors
    weaviate_tenant: str = os.getenv("WEAVIATE_TENANT", "").strip()
    weaviate_target_vector: str = os.getenv("WEAVIATE_TARGET_VECTOR", "").strip()

    # Weaviate Cloud
    weaviate_url: str = os.getenv("WEAVIATE_URL", "").strip()
    weaviate_api_key: str = os.getenv("WEAVIATE_API_KEY", "").strip()

    # Weaviate Custom/Local
    http_host: str = os.getenv("WEAVIATE_HTTP_HOST", "localhost")
    http_port: int = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
    http_secure: bool = _get_bool("WEAVIATE_HTTP_SECURE", False)

    grpc_host: str = os.getenv("WEAVIATE_GRPC_HOST", "localhost")
    grpc_port: int = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    grpc_secure: bool = _get_bool("WEAVIATE_GRPC_SECURE", False)

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))

    # Server
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))

    @property
    def extra_properties(self) -> list[str]:
        return [p.strip() for p in self.weaviate_extra_properties_csv.split(",") if p.strip()]
