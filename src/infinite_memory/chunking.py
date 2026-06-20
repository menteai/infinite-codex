from __future__ import annotations


def chunk_text(text: str, chunk_chars: int = 800, overlap_chars: int = 100) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            split_at = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end))
            if split_at > start + chunk_chars // 2:
                end = split_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks
