from infinite_memory.chunking import chunk_text


def test_chunk_text_short():
    assert chunk_text("hello", 10, 2) == ["hello"]


def test_chunk_text_long():
    chunks = chunk_text("a" * 25, 10, 2)
    assert len(chunks) >= 3
    assert all(chunks)
