from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import weaviate
from weaviate.classes.query import MetadataQuery

try:
    from weaviate.classes.init import Auth, AdditionalConfig, Timeout
except Exception:  # pragma: no cover
    from weaviate.classes.init import Auth  # type: ignore
    from weaviate.config import AdditionalConfig, Timeout  # type: ignore


@dataclass
class RetrievedChunk:
    text: str
    score: Optional[float]
    distance: Optional[float]
    properties: Dict[str, Any]


class WeaviateRetriever:
    def __init__(
        self,
        *,
        collection_name: str,
        text_property: str,
        extra_properties: List[str],
        tenant: str,
        target_vector: str,
        # Cloud
        weaviate_url: str,
        weaviate_api_key: str,
        # Custom/local
        http_host: str,
        http_port: int,
        http_secure: bool,
        grpc_host: str,
        grpc_port: int,
        grpc_secure: bool,
    ):
        self.collection_name = collection_name
        self.text_property = text_property
        self.extra_properties = extra_properties
        self.tenant = tenant
        self.target_vector = target_vector

        if weaviate_url:
            if not weaviate_api_key:
                raise RuntimeError("WEAVIATE_URL is set but WEAVIATE_API_KEY is empty.")
            # Cloud connection helper (REST endpoint + API key). ([docs.weaviate.io](https://docs.weaviate.io/weaviate/connections/connect-cloud?utm_source=openai))
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=Auth.api_key(weaviate_api_key),
                additional_config=AdditionalConfig(timeout=Timeout(init=30, query=60, insert=120)),
            )
        else:
            # Custom connection helper. ([docs.weaviate.io](https://docs.weaviate.io/weaviate/connections/connect-custom?utm_source=openai))
            auth = Auth.api_key(weaviate_api_key) if weaviate_api_key else None
            self.client = weaviate.connect_to_custom(
                http_host=http_host,
                http_port=http_port,
                http_secure=http_secure,
                grpc_host=grpc_host,
                grpc_port=grpc_port,
                grpc_secure=grpc_secure,
                auth_credentials=auth,
                additional_config=AdditionalConfig(timeout=Timeout(init=30, query=60, insert=120)),
            )

        if not self.client.is_ready():
            raise RuntimeError("Weaviate is not ready (check URL/host/ports/auth).")

        base = self.client.collections.get(self.collection_name)
        # Multi-tenancy support if you need it (otherwise unused). ([weaviate.io](https://weaviate.io/developers/weaviate/api/graphql/get?utm_source=openai))
        self.collection = base.with_tenant(self.tenant) if self.tenant else base

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass

    def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        return_props = list(dict.fromkeys([self.text_property] + self.extra_properties))
        meta = MetadataQuery(score=True, distance=True)

        # 1) Try HYBRID (vector + keyword). ([docs.weaviate.io](https://docs.weaviate.io/weaviate/search/hybrid?utm_source=openai))
        try:
            kwargs = dict(
                query=query,
                limit=top_k,
                return_properties=return_props,
                return_metadata=meta,
            )
            # If your collection has named vectors, target_vector is required. ([docs.weaviate.io](https://docs.weaviate.io/weaviate/search/hybrid?utm_source=openai))
            if self.target_vector:
                kwargs["target_vector"] = self.target_vector

            resp = self.collection.query.hybrid(**kwargs)
            return _to_chunks(resp.objects, self.text_property)

        except Exception:
            # 2) Fallback: BM25 keyword search (works even when vectors/hybrid aren’t configured)
            resp = self.collection.query.bm25(
                query=query,
                limit=top_k,
                return_properties=return_props,
                return_metadata=meta,
            )
            return _to_chunks(resp.objects, self.text_property)


def _to_chunks(objects, text_property: str) -> List[RetrievedChunk]:
    chunks: List[RetrievedChunk] = []
    for obj in objects:
        props = obj.properties or {}
        text = props.get(text_property)

        if not isinstance(text, str) or not text.strip():
            # last-resort: any non-empty string property
            for _, v in props.items():
                if isinstance(v, str) and v.strip():
                    text = v
                    break

        if not isinstance(text, str) or not text.strip():
            continue

        md = getattr(obj, "metadata", None)
        chunks.append(
            RetrievedChunk(
                text=text.strip(),
                score=getattr(md, "score", None),
                distance=getattr(md, "distance", None),
                properties=props,
            )
        )
    return chunks
