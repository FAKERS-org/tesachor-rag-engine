from typing import Any, Dict, List


def chunk_json_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert generic JSON records into vector-store chunks.
    Expected minimum fields:
    - text/content/answer/messages
    """
    chunks: List[Dict[str, Any]] = []

    for idx, record in enumerate(records):
        text = (
            record.get("text")
            or record.get("content")
            or record.get("answer")
            or ""
        )

        if not text and "messages" in record and isinstance(record["messages"], list):
            message_text = []
            for msg in record["messages"]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if content:
                    message_text.append(f"{role}: {content}")
            text = "\n".join(message_text)

        if not text:
            continue

        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": str(metadata)}

        metadata["chunk_id"] = metadata.get("chunk_id", f"chunk_{idx}")
        chunks.append({"text": text, "metadata": metadata})

    return chunks
