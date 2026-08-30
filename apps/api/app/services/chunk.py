CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def chunk_text(text: str, *, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    if not text:
        return []
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs or [text]:
        if len(paragraph) <= size:
            pieces.append(paragraph)
            continue
        start = 0
        while start < len(paragraph):
            end = min(start + size, len(paragraph))
            pieces.append(paragraph[start:end].strip())
            if end == len(paragraph):
                break
            start = end - overlap
    return [piece for piece in pieces if piece]
