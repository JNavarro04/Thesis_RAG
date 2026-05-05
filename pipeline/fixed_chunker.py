# Splits the guidebook into fixed-size chunks with overlap.

# Design choices:
#   Chunk size : 256 tokens (Qu et al., 2025)
#   Overlap    : 50 tokens (Gomez-Cabello et al., 2025)

# Token approximation:
#   Tokens taken as words (split every whitespace). This matches
#   the approach used in Qu et al. and avoids a dependency on a
#   tokeniser, keeping the chunker model-agnostic.

import re
from pathlib import Path


CHUNK_SIZE = 256
OVERLAP    = 50

#Helper functions

def load_guidebook(filepath: str) -> str:
    """
    Reads guidebook.txt (or any file) and returns it as a string.
    """
    return Path(filepath).read_text(encoding="utf-8")


def tokenise(text: str) -> list[str]:
    """
    Split text into tokens (words) every whitespace while preserving punctuation. Returns a list of all words.
    """
    return text.split()

def normalise(text: str) -> str:
    """
    Remove excess whitespace at the end of lines, and remove whitelines (empty lines).
    """
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def chunk_fixed(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[dict]:
    """
    Split text into fixed-size chunks with overlap.

    Parameters:
    - text: full document text
    - chunk_size: number of tokens per chunk
    - overlap: number of tokens shared between consecutive chunks

    Returns:
    List of chunk dicts with keys: chunk_id, text, char_start, char_end, token_count.
    """
    tokens = tokenise(text)
    total_tokens = len(tokens)

    # Build a character offset map: map token to its starting character index in the guidebook, which allows for exact chunk start/end.
    char_offsets = []
    pos = 0
    for token in tokens:
        # find the token in the text starting from pos
        idx = text.index(token, pos)
        char_offsets.append(idx)
        pos = idx + len(token)

    chunks = []
    step = chunk_size - overlap  #how far to advance the window each time
    chunk_index = 0
    start = 0

    while start < total_tokens:
        #end once there is no more text to the chunked, min between the total token count and the step
        end = min(start + chunk_size, total_tokens)

        chunk_tokens = tokens[start:end]
        chunk_text = normalise(" ".join(chunk_tokens))

        char_start = char_offsets[start]
        #char_end: end of the last token in the chunk
        last_token_idx = end - 1
        char_end = char_offsets[last_token_idx] + len(tokens[last_token_idx])

        #chunk metadata to be returned
        chunks.append({
            "chunk_id":    f"fixed_{chunk_index:03d}",
            "text":        chunk_text,
            "char_start":  char_start,
            "char_end":    char_end,
            "token_count": len(chunk_tokens)
        })

        chunk_index += 1

        if end == total_tokens:
            break

        start += step #start of the next chunk is the previous chunk size - overlap (allows for missing context).

    return chunks


# Main - for offline testing

def run(guidebook_path: str) -> list[dict]:
    """
    Load the guidebook and return fixed-size chunks.
    """
    text = load_guidebook(guidebook_path)
    chunks = chunk_fixed(text)
    return chunks


if __name__ == "__main__":
    import json

    guidebook_path = Path(__file__).parent.parent / "data" / "guidebook.txt"
    chunks = run(guidebook_path)

    #create a json with all the chunks
    output_path = Path(__file__).parent.parent / "data" / "chunks_fixed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    # Chunking information
    print("\n===========================")
    print("CHUNKING INFORMATION")   
    print("===========================\n")

    print(f"Total chunks produced : {len(chunks)}")
    print(f"Chunk size (tokens)   : {CHUNK_SIZE}")
    print(f"Overlap (tokens)      : {OVERLAP}")
    print(f"Avg tokens per chunk  : {sum(c['token_count'] for c in chunks) / len(chunks):.1f}")
    print(f"Saved to {output_path}")   

    # Preview first 3 chunks
    print("\n===========================")
    print("FIRST THREE CHUNKS")   
    print("===========================\n")
    for c in chunks[:3]:
        print(f"--- {c['chunk_id']} | tokens: {c['token_count']} | "
              f"chars: {c['char_start']}–{c['char_end']} ---")
        print(c["text"])
        print()
