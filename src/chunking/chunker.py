def chunk_text(text, max_words=800):
    """
    Chunk text paragraph-wise, keeping each chunk under max_words.
    Paragraphs are assumed to be separated by double newlines (\n\n).
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_words + para_words <= max_words:
            current_chunk += para + "\n\n"
            current_words += para_words
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
            current_words = para_words

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks