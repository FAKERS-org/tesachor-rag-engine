import json
import asyncio
from pathlib import Path
from app.services.factory import get_embeddings_service
from app.services.vector_store import VectorStoreService
from app.config import settings
from app.utils.data_transformer import ConversationTransformer


async def build_with_hf_inference():
    embeddings = get_embeddings_service()
    test_embed = await embeddings.aembed_query("test")
    print(f"Embedding dimension: {len(test_embed)}")

    vector_store = VectorStoreService(embeddings=embeddings)

    dataset_dir = Path(settings.DATASET_DIR)
    if not dataset_dir.exists():
        dataset_dir = Path(settings.ALT_DATASET_DIR)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found. Checked: {settings.DATASET_DIR} and {settings.ALT_DATASET_DIR}"
        )

    output_path = Path("data/rag_documents.jsonl")
    transformer = ConversationTransformer(
        data_dir=str(dataset_dir), flat_format=True)
    # Specify the number of files to process here (e.g., 5)
    transformer.transform_all(output_file=str(output_path), max_files=5)

    documents = []
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))

    print(f"Building vector store from {len(documents)} transformed chunks...")
    vector_store.reset_collection()

    batch_size = settings.EMBEDDING_BATCH_SIZE

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        texts = [d["content"] for d in batch]
        metadatas = [d.get("metadata", {}) for d in batch]

        for j, meta in enumerate(metadatas):
            meta["doc_id"] = meta.get("doc_id", f"doc_{i+j}")
            meta["chunk_id"] = meta.get("chunk_id", f"chunk_{i+j}")

        vector_store.add_documents(texts, metadatas)
        print(
            f"  Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")

    stats = vector_store.get_collection_stats()
    print(f"Vector store ready: {stats}")

if __name__ == "__main__":
    asyncio.run(build_with_hf_inference())
