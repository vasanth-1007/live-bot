import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import List, Optional

import weaviate
import weaviate.classes.config as wvcc
import weaviate.classes.query as wvcq

import fitz  # PyMuPDF


# ----------------------------
# Configuration
# ----------------------------
DEFAULT_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "SOPChunks")

# Gemini
OCR_MODEL = os.getenv("GEMINI_OCR_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

# Chunking
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Rendering quality (2.0–3.0 usually good; higher = bigger images, higher cost)
RENDER_ZOOM = float(os.getenv("RENDER_ZOOM", "2.5"))

DOCNO_RE = re.compile(r"\bSOP-[A-Z0-9]+(?:-[A-Z0-9]+)*\b", re.IGNORECASE)


# ----------------------------
# Gemini helpers (Google GenAI SDK)
# ----------------------------
def get_gemini_client():
    # Uses GEMINI_API_KEY or GOOGLE_API_KEY automatically if set
    # https://ai.google.dev/gemini-api/docs/api-key
    from google import genai
    return genai.Client()


def gemini_ocr_page_png(png_bytes: bytes) -> str:
    """
    Uses Gemini multimodal generation as an OCR/transcription step.
    """
    from google.genai import types

    client = get_gemini_client()

    ocr_prompt = (
        "You are an OCR engine.\n"
        "Task: Transcribe ALL visible text from the provided scanned document page image.\n"
        "Rules:\n"
        "- Do NOT summarize, paraphrase, or correct wording.\n"
        "- Preserve reading order as best as possible.\n"
        "- Keep headings, bullet points, numbering.\n"
        "- If there are tables, output them as plain text with clear row/column separation.\n"
        "- If text is unreadable, output [ILLEGIBLE] in that spot.\n"
        "- Do not add any text that is not present in the image.\n"
        "Output: plain text only."
    )

    resp = client.models.generate_content(
        model=OCR_MODEL,
        contents=[
            ocr_prompt,
            types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            # Raise if pages are dense; too low can truncate.
            max_output_tokens=8192,
        ),
    )

    return (resp.text or "").strip()


def gemini_embed_texts(texts: List[str], task_type: str) -> List[List[float]]:
    """
    Embeds a list of texts using Gemini embeddings.
    task_type: "RETRIEVAL_DOCUMENT" for stored chunks, "RETRIEVAL_QUERY" for queries.
    """
    from google.genai import types

    client = get_gemini_client()
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )

    # result.embeddings is a list; each embedding has .values
    return [e.values for e in result.embeddings]


# ----------------------------
# Weaviate helpers
# ----------------------------
#def get_weaviate_client():
    # Local defaults: http://localhost:8080 and grpc localhost:50051
    # https://weaviate.io/developers/weaviate/connections/connect-local
#    return weaviate.connect_to_local(
#        host=os.getenv("WEAVIATE_HOST", "127.0.0.1"),
#        port=int(os.getenv("WEAVIATE_PORT", "8080")),
#        grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
#    )
def get_weaviate_client():
    # Local defaults: http://localhost:8080 and grpc localhost:50051
    # https://weaviate.io/developers/weaviate/connections/connect-local
    
    # Get the key from environment
    api_key = os.getenv("WEAVIATE_API_KEY")

    # If key exists, use it. Otherwise, try anonymous (which will fail for you).
    if api_key:
        return weaviate.connect_to_local(
            host=os.getenv("WEAVIATE_HOST", "127.0.0.1"),
            port=int(os.getenv("WEAVIATE_PORT", "8080")),
            grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
            auth_credentials=weaviate.auth.AuthApiKey(api_key),  # <--- ADDED THIS
        )
    else:
        return weaviate.connect_to_local(
            host=os.getenv("WEAVIATE_HOST", "127.0.0.1"),
            port=int(os.getenv("WEAVIATE_PORT", "8080")),
            grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
        )



def recreate_collection(client, name: str):
    """
    Deletes the collection (and all its objects), then recreates it.
    Deleting a collection deletes its objects. Use carefully.
    """
    try:
        client.collections.delete(name)
    except Exception:
        pass

    client.collections.create(
        name=name,
        # bring your own vectors (self_provided)
        #vector_config=wvcc.Configure.Vectors.self_provided(),
        #vector_index_config=wvcc.Configure.VectorIndex.hnsw(
        #    distance_metric=wvcc.VectorDistances.COSINE
        #),
        vector_config=wvcc.Configure.Vectors.self_provided(
            vector_index_config=wvcc.Configure.VectorIndex.hnsw(
                distance_metric=wvcc.VectorDistances.COSINE
            )
        ),
        properties=[
            wvcc.Property(name="doc_id", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="doc_no", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="source_file", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="page", data_type=wvcc.DataType.INT),
            wvcc.Property(name="chunk_index", data_type=wvcc.DataType.INT),
            wvcc.Property(name="text", data_type=wvcc.DataType.TEXT),
        ],
    )


def get_collection(client, name: str):
    return client.collections.get(name)


def delete_doc_chunks(client, collection_name: str, doc_id: str):
    col = get_collection(client, collection_name)
    col.data.delete_many(where=wvcq.Filter.by_property("doc_id").equal(doc_id))


# ----------------------------
# PDF -> images -> OCR -> chunking
# ----------------------------
def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def infer_doc_no(pdf_path: Path, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.strip()
    stem = pdf_path.stem.strip()
    if stem.upper().startswith("SOP-"):
        return stem
    # fallback: use filename stem
    return stem


def render_pdf_pages_to_png_bytes(pdf_path: Path) -> List[bytes]:
    doc = fitz.open(str(pdf_path))
    out = []
    mat = fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM)

    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out.append(pix.tobytes("png"))
    doc.close()
    return out


def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def ingest_scanned_pdf(
    client,
    collection_name: str,
    pdf_path: Path,
    doc_no: Optional[str],
    reset_doc_first: bool,
):
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    # doc_id strategy:
    # - If you want “same file always same doc”: hash the file bytes (default).
    # - If you want “same SOP number = same doc”: use doc_no as doc_id instead.
    computed_doc_id = file_sha1(pdf_path)
    inferred_doc_no = infer_doc_no(pdf_path, doc_no)

    if reset_doc_first:
        delete_doc_chunks(client, collection_name, computed_doc_id)

    col = get_collection(client, collection_name)

    # 1) Render -> 2) OCR per page
    png_pages = render_pdf_pages_to_png_bytes(pdf_path)

    objects = []
    global_chunk_index = 0

    for page_i, png_bytes in enumerate(png_pages, start=1):
        page_text = gemini_ocr_page_png(png_bytes)

        # If OCR returns empty, you may still want to store a marker
        # so you know page existed.
        if not page_text.strip():
            page_text = "[NO_TEXT_DETECTED]"

        chunks = chunk_text(page_text, CHUNK_CHARS, CHUNK_OVERLAP)

        for c in chunks:
            objects.append(
                {
                    "doc_id": computed_doc_id,
                    "doc_no": inferred_doc_no,
                    "source_file": pdf_path.name,
                    "page": page_i,
                    "chunk_index": global_chunk_index,
                    "text": c,
                }
            )
            global_chunk_index += 1

    # 3) Embed chunks (document task type)
    texts = [o["text"] for o in objects]
    vectors = gemini_embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    # 4) Insert with self-provided vectors (batch)
    with col.batch.fixed_size(batch_size=100) as batch:
        for props, vec in zip(objects, vectors):
            batch.add_object(properties=props, vector=vec)

    failed = col.batch.failed_objects
    if failed:
        print(f"[WARN] Some objects failed to insert: {len(failed)} (showing 1)")
        print(failed[0])

    print(f"[OK] Ingested {len(objects)} chunks from {pdf_path.name}")
    print(f"     doc_no={inferred_doc_no}")
    print(f"     doc_id={computed_doc_id}")


# ----------------------------
# Retrieval
# ----------------------------
def full_retrieve_by_doc_id(client, collection_name: str, doc_id: str, limit: int = 20000) -> str:
    col = get_collection(client, collection_name)
    resp = col.query.fetch_objects(
        filters=wvcq.Filter.by_property("doc_id").equal(doc_id),
        sort=wvcq.Sort.by_property("chunk_index", ascending=True),
        limit=limit,
        return_properties=["doc_no", "source_file", "page", "chunk_index", "text"],
    )
    if not resp.objects:
        raise ValueError(f"No chunks found for doc_id={doc_id}")

    if len(resp.objects) >= limit:
        print(f"[WARN] Hit limit={limit}. Increase it if the document is longer.")

    return "\n\n".join(o.properties["text"] for o in resp.objects)


def semantic_search(client, collection_name: str, query: str, doc_id: Optional[str], k: int = 5):
    col = get_collection(client, collection_name)
    qvec = gemini_embed_texts([query], task_type="RETRIEVAL_QUERY")[0]

    filters = None
    if doc_id:
        filters = wvcq.Filter.by_property("doc_id").equal(doc_id)

    resp = col.query.near_vector(
        near_vector=qvec,
        limit=k,
        filters=filters,
        return_metadata=wvcq.MetadataQuery(distance=True),
        return_properties=["doc_no", "source_file", "page", "chunk_index", "text"],
    )
    for i, o in enumerate(resp.objects, 1):
        dist = getattr(o.metadata, "distance", None)
        p = o.properties
        print(f"\n--- Hit {i} | distance={dist} | page={p.get('page')} | chunk={p.get('chunk_index')}")
        print(p.get("text", ""))


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser("Gemini OCR -> Weaviate ingestion + retrieval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s_reset = sub.add_parser("reset", help="DELETE + recreate the whole collection (wipes old data).")
    s_reset.add_argument("--collection", default=DEFAULT_COLLECTION)

    s_ingest = sub.add_parser("ingest", help="Ingest a scanned PDF using Gemini OCR.")
    s_ingest.add_argument("pdf", help="Path to PDF")
    s_ingest.add_argument("--collection", default=DEFAULT_COLLECTION)
    s_ingest.add_argument("--doc-no", default=None)
    s_ingest.add_argument("--reset-doc-first", action="store_true", help="Delete prior chunks for this doc_id first")

    s_full = sub.add_parser("full", help="Rebuild full document text by doc_id (ordered).")
    s_full.add_argument("doc_id")
    s_full.add_argument("--collection", default=DEFAULT_COLLECTION)
    s_full.add_argument("--limit", type=int, default=20000)

    s_del = sub.add_parser("delete-doc", help="Delete chunks for a doc_id (keeps other docs).")
    s_del.add_argument("doc_id")
    s_del.add_argument("--collection", default=DEFAULT_COLLECTION)

    s_search = sub.add_parser("search", help="Semantic search (optionally within a single doc_id).")
    s_search.add_argument("query")
    s_search.add_argument("--collection", default=DEFAULT_COLLECTION)
    s_search.add_argument("--doc-id", default=None)
    s_search.add_argument("-k", type=int, default=5)

    args = ap.parse_args()

    with get_weaviate_client() as client:
        if not client.is_ready():
            raise RuntimeError("Weaviate is not ready. Check host/ports (8080/50051) and server logs.")

        if args.cmd == "reset":
            recreate_collection(client, args.collection)
            print(f"[OK] Recreated collection: {args.collection}")

        elif args.cmd == "ingest":
            ingest_scanned_pdf(
                client=client,
                collection_name=args.collection,
                pdf_path=Path(args.pdf),
                doc_no=args.doc_no,
                reset_doc_first=args.reset_doc_first,
            )

        elif args.cmd == "full":
            text = full_retrieve_by_doc_id(client, args.collection, args.doc_id, limit=args.limit)
            print(text)

        elif args.cmd == "delete-doc":
            delete_doc_chunks(client, args.collection, args.doc_id)
            print(f"[OK] Deleted doc_id={args.doc_id} (if it existed).")

        elif args.cmd == "search":
            semantic_search(client, args.collection, args.query, doc_id=args.doc_id, k=args.k)


if __name__ == "__main__":
    main()
