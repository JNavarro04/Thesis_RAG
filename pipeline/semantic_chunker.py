# Splits the guidebook into semantically coherent chunks using the document's dividers as chunk boundaries (----).

# The guidebook uses === and --- dividers to define section and entry boundaries (SECTION, ZONE_XX, AISLE_XX, etc.) 
# and each block is one semantic unit (one zone, aisle, item, route, feature, or scenario). 
# Boundaries are defined by meaning rather than token counts (Kiss et al., 2025; Gomez-Cabello et al., 2025).

# Any chunk exceeding MAX_TOKENS is split on paragraph then line boundaries to stay within the embedding model's 
# input limit (500 this case) (Karpukhin et al., 2020).

import re
import json
from pathlib import Path
 
MAX_TOKENS = 500
 
 
def normalise(text: str) -> str:
    """
    Remove excess whitespace at the end of lines, and remove whitelines (empty lines).
    """
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
 
 
def split_large(text: str, max_tokens: int) -> list[str]:
    """
    Recursively split text over max_tokens on paragraph, otherwise split on line breaks.
    """
    if len(text.split()) <= max_tokens:
        return [text]

    #split on paragraphs, if there is more than 1 paragraph (\n\n, whitespaces)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    units, joiner = (paragraphs, '\n\n') if len(paragraphs) > 1 \
        else ([l.strip() for l in text.split('\n') if l.strip()], '\n') #split on line break otherwise
    result, current, current_tokens = [], [], 0
    for unit in units:
        ut = len(unit.split())
        if current_tokens + ut > max_tokens and current:
            result.append(joiner.join(current))
            current, current_tokens = [unit], ut
        else:
            current.append(unit)
            current_tokens += ut
    if current:
        result.append(joiner.join(current))
    final = []
    for r in result:
        final.extend(split_large(r, max_tokens) if len(r.split()) > max_tokens else [r])
    return final or [text]
 
 
def run(guidebook_path: str) -> list[dict]:
    text = Path(guidebook_path).read_text(encoding='utf-8')
 
    #Split on === and --- divider lines (10+ equal characters)
    raw = re.split(r'\n[=\-]{10,}\s*\n', text)
 
    #Strip residual divider lines and normalise each segment
    cleaned = []
    for seg in raw:
        lines = [l for l in seg.split('\n')
                 if not re.fullmatch(r'[=\-]{10,}', l.strip())]
        seg = normalise('\n'.join(lines))
        if seg:
            cleaned.append(seg)
 
    #Build chunks: split oversized blocks, assign IDs and compute offsets for chunk start/end
    chunks = []
    chunk_index = 0
    search_pos = 0
 
    for block in cleaned:
        for sub in split_large(block, MAX_TOKENS):
            sub = sub.strip()
            if not sub:
                continue
            idx = text.find(sub[:50].strip(), search_pos)
            if idx == -1:
                idx = text.find(sub[:50].strip())
            char_start = idx if idx != -1 else search_pos
            char_end = char_start + len(sub)
            search_pos = max(search_pos, char_end)
            chunks.append({
                'chunk_id':     f'semantic_{chunk_index:03d}',
                'text':         normalise(sub),
                'char_start':   char_start,
                'char_end':     char_end,
                'token_count':  len(sub.split()),
                'entry_header': next(
                    (l.strip() for l in sub.split('\n') if l.strip()), ''
                ),
            })
            chunk_index += 1
 
    return chunks
 

if __name__ == '__main__':
    guidebook_path = Path(__file__).parent.parent / 'data' / 'guidebook.txt'
    chunks = run(str(guidebook_path))
    token_counts = [c['token_count'] for c in chunks]

    print("\n===========================")
    print("CHUNKING INFORMATION")   
    print("===========================\n")
    print(f'Total chunks : {len(chunks)}')
    print(f'Avg tokens   : {sum(token_counts) / len(token_counts):.1f}')
    print(f'Min tokens   : {min(token_counts)}')
    print(f'Max tokens   : {max(token_counts)}')
    print(f'Over 512     : {sum(1 for t in token_counts if t > 512)}')
    
    print("\n===========================")
    print("FIRST THREE CHUNKS")   
    print("===========================\n")
    for c in chunks[:3]:
        print(f"--- {c['chunk_id']} | {c['token_count']}t | {c['entry_header'][:60]}")
        print(f"    {c['text']}")
        print()

    output_path = Path(__file__).parent.parent / 'data' / 'chunks_semantic.json' #save chunks to a json file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f'Saved {len(chunks)} chunks to {output_path}')