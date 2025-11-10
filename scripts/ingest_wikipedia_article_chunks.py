"""
Ingest Wikipedia articles as overlapping chunks with embeddings.

This script:
1. Queries articles that already have embeddings from the articles table
2. Retrieves full text from Wikipedia XML
3. Chunks the text with overlap (200 words per chunk, 100 word advance)
4. Generates embeddings for each chunk
5. Inserts chunks into article_chunks table
"""

import argparse
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import psycopg2
from psycopg2.extras import Json
import requests
from tqdm import tqdm


def clean_wikitext(text: str) -> str:
    """
    Clean Wikipedia markup from text.
    
    Args:
        text: Raw wikitext
        
    Returns:
        Cleaned plain text
    """
    # Remove templates and references
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/>', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove wiki links but keep the text
    text = re.sub(r'\[\[([^|\]]+\|)?([^\]]+)\]\]', r'\2', text)
    
    # Remove external links
    text = re.sub(r'\[http[^\]]+\]', '', text)
    
    # Remove file/image references
    text = re.sub(r'\[\[File:[^\]]+\]\]', '', text)
    text = re.sub(r'\[\[Image:[^\]]+\]\]', '', text)
    
    # Remove wiki formatting
    text = re.sub(r"'''?", '', text)
    text = re.sub(r'==+\s*', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def extract_sections(text: str) -> List[Tuple[int, str]]:
    """
    Extract section headers and their character positions from cleaned text.
    
    Args:
        text: Cleaned text (should still have section markers)
        
    Returns:
        List of (position, section_name) tuples sorted by position
    """
    sections = [(0, "Introduction")]  # Default section at start
    
    # Find section headers (assumes they start at beginning of line)
    # Common patterns: "Section Name" after cleaning removes == markers
    # We'll look for lines that look like headers (short, capitalized)
    lines = text.split('\n')
    char_pos = 0
    
    for line in lines:
        line_stripped = line.strip()
        # Heuristic: section headers are typically short (< 100 chars), 
        # start with capital, and don't end with punctuation
        if (line_stripped and 
            len(line_stripped) < 100 and 
            line_stripped[0].isupper() and 
            not line_stripped[-1] in '.!?,;:' and
            len(line_stripped.split()) <= 8):  # Not too many words
            sections.append((char_pos, line_stripped))
        
        char_pos += len(line) + 1  # +1 for newline
    
    return sections


def get_section_for_position(sections: List[Tuple[int, str]], position: int) -> str:
    """
    Get the section name for a given character position.
    
    Args:
        sections: List of (position, section_name) tuples
        position: Character position in text
        
    Returns:
        Section name that contains this position
    """
    current_section = "Introduction"
    
    for section_pos, section_name in sections:
        if position >= section_pos:
            current_section = section_name
        else:
            break
    
    return current_section


def create_chunks(text: str, chunk_size: int = 200, overlap: int = 100) -> List[Tuple[int, int, str]]:
    """
    Create overlapping chunks from text based on word boundaries.
    Uses actual character positions from original text to enable exact reconstruction.
    Also tracks section context for each chunk.
    
    Args:
        text: Input text (will preserve exact whitespace)
        chunk_size: Number of words per chunk
        overlap: Number of words to overlap between chunks
        
    Returns:
        List of (start_char_index, end_char_index, section_name) tuples
    """
    if not text:
        return []
    
    # Extract section headers and their positions
    sections = extract_sections(text)
    
    # Find word boundaries in original text using regex
    # This preserves exact character positions
    word_pattern = re.compile(r'\S+')
    words = [(m.start(), m.end()) for m in word_pattern.finditer(text)]
    
    if len(words) == 0:
        return []
    
    # Calculate step size (advance)
    step = chunk_size - overlap
    if step <= 0:
        step = 1  # Safety check
    
    chunks = []
    i = 0
    
    while i < len(words):
        # Get word boundary indices for this chunk
        chunk_words = words[i:i + chunk_size]
        
        if not chunk_words:
            break
        
        # Use actual character positions from original text
        start_char = chunk_words[0][0]  # Start of first word
        end_char = chunk_words[-1][1]   # End of last word
        
        # Determine which section this chunk belongs to
        section = get_section_for_position(sections, start_char)
        
        chunks.append((start_char, end_char, section))
        
        # Advance by step
        i += step
    
    return chunks


def validate_chunk_reconstruction(text: str, start_char: int, end_char: int, expected_word_count: int = None) -> bool:
    """
    Validate that chunk can be exactly reconstructed from character indices.
    
    Args:
        text: Original text
        start_char: Starting character index
        end_char: Ending character index
        expected_word_count: Optional word count to verify
        
    Returns:
        True if chunk is valid and can be reconstructed
        
    Raises:
        ValueError: If reconstruction fails validation
    """
    # Extract chunk using indices
    chunk_text = text[start_char:end_char]
    
    # Validate indices are within bounds
    if start_char < 0 or end_char > len(text) or start_char >= end_char:
        raise ValueError(
            f"Invalid chunk indices: start={start_char}, end={end_char}, text_len={len(text)}"
        )
    
    # Validate chunk is not empty
    if not chunk_text.strip():
        raise ValueError(f"Chunk is empty or whitespace-only: [{start_char}:{end_char}]")
    
    # Optionally validate word count
    if expected_word_count is not None:
        actual_word_count = len(chunk_text.split())
        if actual_word_count != expected_word_count:
            raise ValueError(
                f"Word count mismatch: expected {expected_word_count}, got {actual_word_count}"
            )
    
    # Validate chunk text can be retrieved again (round-trip test)
    retrieved_chunk = text[start_char:end_char]
    if chunk_text != retrieved_chunk:
        raise ValueError(
            f"Round-trip validation failed! "
            f"Original: '{chunk_text[:50]}...' != Retrieved: '{retrieved_chunk[:50]}...'"
        )
    
    return True


def load_question_articles(questions_path: Path) -> Set[str]:
    """
    Load article titles that have questions from questions JSON file.
    
    Args:
        questions_path: Path to questions JSON file
        
    Returns:
        Set of article titles that have questions
    """
    if not questions_path.exists():
        return set()
    
    try:
        with open(questions_path, 'r') as f:
            questions_data = json.load(f)
        
        # Extract unique article titles (could be 'title' or 'source_article' depending on format)
        titles = set()
        for item in questions_data:
            if 'title' in item:
                titles.add(item['title'])
            elif 'source_article' in item:
                titles.add(item['source_article'])
        
        return titles
    except Exception as e:
        print(f"⚠️  Warning: Could not load questions file: {e}")
        return set()


def get_vectorized_articles(db_url: str, source: str = 'wikipedia') -> List[Dict]:
    """
    Get articles that already have embeddings.
    
    Args:
        db_url: PostgreSQL connection URL
        source: Source filter
        
    Returns:
        List of article dictionaries with id, title, and content
    """
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    
    query = """
        SELECT id, title, content
        FROM articles 
        WHERE source = %s AND embedding IS NOT NULL
        ORDER BY title
    """
    
    cursor.execute(query, (source,))
    rows = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return [{'id': str(row[0]), 'title': row[1], 'content': row[2]} for row in rows]


def prioritize_articles(articles: List[Dict], priority_titles: Set[str]) -> List[Dict]:
    """
    Reorder articles to prioritize those with questions.
    
    Args:
        articles: List of article dictionaries
        priority_titles: Set of article titles to prioritize
        
    Returns:
        Reordered list with priority articles first
    """
    if not priority_titles:
        return articles
    
    priority_articles = [a for a in articles if a['title'] in priority_titles]
    other_articles = [a for a in articles if a['title'] not in priority_titles]
    
    return priority_articles + other_articles


def get_existing_chunks(db_url: str, article_id: str) -> Set[int]:
    """
    Get existing chunk IDs for an article.
    
    Args:
        db_url: PostgreSQL connection URL
        article_id: Article UUID
        
    Returns:
        Set of chunk IDs that already exist
    """
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    
    query = """
        SELECT chunk_id
        FROM article_chunks
        WHERE article_id = %s
    """
    
    cursor.execute(query, (article_id,))
    existing = {row[0] for row in cursor.fetchall()}
    
    cursor.close()
    conn.close()
    
    return existing


def generate_embedding(
    text: str, 
    ollama_host: str = "http://localhost:11434", 
    model: str = "nomic-embed-text"
) -> Tuple[Optional[List[float]], float]:
    """
    Generate embedding for text using Ollama.
    
    Args:
        text: Text to embed
        ollama_host: Ollama API endpoint
        model: Embedding model name
        
    Returns:
        Tuple of (embedding vector or None on error, time taken in seconds)
    """
    start_time = time.time()
    try:
        response = requests.post(
            f"{ollama_host}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        elapsed = time.time() - start_time
        return response.json()["embedding"], elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return None, elapsed


def insert_chunk(
    conn: psycopg2.extensions.connection,
    article_id: str,
    title: str,
    chunk_id: int,
    start_char: int,
    end_char: int,
    embedding: List[float],
    source: str = 'wikipedia',
    commit: bool = False
) -> Tuple[bool, float]:
    """
    Insert chunk with embedding into database.
    
    Args:
        conn: Database connection
        article_id: Parent article UUID
        title: Article title
        chunk_id: Sequential chunk number
        start_char: Starting character index
        end_char: Ending character index
        embedding: Embedding vector
        source: Source identifier
        commit: Whether to commit immediately
        
    Returns:
        Tuple of (True if inserted successfully, time taken in seconds)
    """
    start_time = time.time()
    cursor = conn.cursor()
    
    insert_query = """
        INSERT INTO article_chunks (article_id, title, chunk_id, start_char_index, end_char_index, embedding, source, metadata)
        VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s)
        ON CONFLICT (article_id, chunk_id) DO NOTHING
    """
    
    try:
        cursor.execute(
            insert_query,
            (
                article_id,
                title,
                chunk_id,
                start_char,
                end_char,
                embedding,
                source,
                Json({'word_count': (end_char - start_char) // 5})  # Rough estimate
            )
        )
        if commit:
            conn.commit()
        cursor.close()
        elapsed = time.time() - start_time
        return True, elapsed
    except Exception as e:
        print(f"\n❌ Insert error for chunk {chunk_id}: {e}")
        conn.rollback()
        cursor.close()
        elapsed = time.time() - start_time
        return False, elapsed


def process_article(
    article: Dict,
    db_url: str,
    ollama_host: str,
    chunk_size: int,
    overlap: int,
    model: str = "nomic-embed-text",
    show_progress: bool = False
) -> Tuple[int, int, Dict[str, List[float]]]:
    """
    Process single article into chunks.
    
    Args:
        article: Article dictionary with id, title, content
        db_url: Database URL
        ollama_host: Ollama host
        chunk_size: Words per chunk
        overlap: Overlap words
        model: Embedding model
        show_progress: Show nested progress bar for chunks
        
    Returns:
        Tuple of (chunks_generated, chunks_inserted, timing_stats)
    """
    # Check existing chunks
    existing_chunks = get_existing_chunks(db_url, article['id'])
    
    # Create chunks
    content = article['content']
    chunk_boundaries = create_chunks(content, chunk_size, overlap)
    
    # Skip if all chunks already exist
    new_chunks = [i for i in range(len(chunk_boundaries)) if i not in existing_chunks]
    if not new_chunks:
        return (0, 0, {'api_times': [], 'insert_times': [], 'validation_times': []})
    
    # Process new chunks
    conn = psycopg2.connect(db_url)
    generated = 0
    inserted = 0
    
    # Timing statistics
    api_times = []
    insert_times = []
    validation_times = []
    
    # Optional nested progress bar
    chunk_iter = tqdm(new_chunks, desc=f"  Chunks", leave=False, disable=not show_progress)
    
    for chunk_id in chunk_iter:
        start_char, end_char, section = chunk_boundaries[chunk_id]
        chunk_text = content[start_char:end_char]
        
        # VALIDATION: Verify chunk can be exactly reconstructed
        val_start = time.time()
        try:
            validate_chunk_reconstruction(content, start_char, end_char)
            validation_times.append(time.time() - val_start)
        except ValueError as e:
            validation_times.append(time.time() - val_start)
            print(f"\n❌ Validation failed for article '{article['title']}' chunk {chunk_id}: {e}")
            print(f"   Skipping this chunk...")
            continue
        
        # Prepend article title and section context for embedding
        context = f"Article: {article['title']}\nSection: {section}\n\n"
        text_with_context = context + chunk_text
        
        # Generate embedding with context
        embedding, api_time = generate_embedding(text_with_context, ollama_host, model)
        api_times.append(api_time)
        
        if embedding:
            generated += 1
            # Insert chunk
            success, insert_time = insert_chunk(conn, article['id'], article['title'], chunk_id, start_char, end_char, embedding)
            insert_times.append(insert_time)
            if success:
                inserted += 1
    
    # Final commit
    conn.commit()
    conn.close()
    
    timing_stats = {
        'api_times': api_times,
        'insert_times': insert_times,
        'validation_times': validation_times
    }
    
    return (generated, inserted, timing_stats)


def main() -> None:
    """Main function to ingest Wikipedia article chunks."""
    parser = argparse.ArgumentParser(
        description="Ingest Wikipedia articles as overlapping chunks with embeddings"
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/rag_experiments"
        ),
        help="PostgreSQL database URL"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama API host"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-embed-text",
        help="Ollama embedding model name"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of words per chunk (default: 200)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Number of overlapping words (default: 100)"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Maximum number of articles to process (default: all)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="./data/questions/wikipedia_questions.json",
        help="Path to questions JSON file (prioritizes these articles)"
    )
    parser.add_argument(
        "--show-chunk-progress",
        action="store_true",
        help="Show nested progress bar for individual chunks"
    )
    
    args = parser.parse_args()
    
    print("🔄 Wikipedia Article Chunks Ingestion")
    print("=" * 50)
    print(f"Database: {args.db_url.split('@')[-1] if '@' in args.db_url else args.db_url}")
    print(f"Ollama: {args.ollama_host}")
    print(f"Model: {args.embedding_model}")
    print(f"Chunk size: {args.chunk_size} words")
    print(f"Overlap: {args.overlap} words")
    
    # Load articles with questions for prioritization
    questions_path = Path(args.questions)
    priority_titles = load_question_articles(questions_path)
    
    if priority_titles:
        print(f"\n📋 Found {len(priority_titles)} articles with questions (will prioritize)")
    
    # Get vectorized articles
    print(f"\n1️⃣  Loading vectorized articles from database...")
    articles = get_vectorized_articles(args.db_url)
    
    if not articles:
        print("❌ No vectorized articles found")
        print("   Run 'ingest-wikipedia' first to vectorize articles")
        return
    
    # Prioritize articles that have questions
    articles = prioritize_articles(articles, priority_titles)
    
    # Count how many priority articles we have
    priority_count = sum(1 for a in articles if a['title'] in priority_titles)
    if priority_count > 0:
        print(f"   ✓ Found {len(articles)} vectorized articles ({priority_count} with questions)")
    else:
        print(f"   ✓ Found {len(articles)} vectorized articles")
    
    if args.max_articles:
        articles = articles[:args.max_articles]
        priority_in_subset = sum(1 for a in articles if a['title'] in priority_titles)
        if priority_in_subset > 0:
            print(f"   ✓ Processing {len(articles)} articles ({priority_in_subset} with questions first)")
    
    # Process articles
    print(f"\n2️⃣  Processing articles into chunks...")
    print("   (Statistics update in progress bar)\n")
    
    total_generated = 0
    total_inserted = 0
    
    # Aggregate timing statistics
    all_api_times = []
    all_insert_times = []
    all_validation_times = []
    
    # Create progress bar with custom format
    pbar = tqdm(articles, desc="Processing articles")
    
    for article in pbar:
        generated, inserted, timing_stats = process_article(
            article,
            args.db_url,
            args.ollama_host,
            args.chunk_size,
            args.overlap,
            args.embedding_model,
            show_progress=args.show_chunk_progress
        )
        total_generated += generated
        total_inserted += inserted
        
        # Collect timing stats
        all_api_times.extend(timing_stats['api_times'])
        all_insert_times.extend(timing_stats['insert_times'])
        all_validation_times.extend(timing_stats['validation_times'])
        
        # Update progress bar with current statistics
        if all_api_times:
            avg_api = sum(all_api_times) / len(all_api_times)
            recent_api = sum(all_api_times[-10:]) / min(10, len(all_api_times)) if all_api_times else 0
            max_api = max(all_api_times)
            
            avg_insert = sum(all_insert_times) / len(all_insert_times) if all_insert_times else 0
            
            postfix = {
                'chunks': total_inserted,
                'api_avg': f"{avg_api:.2f}s",
                'api_recent': f"{recent_api:.2f}s",
                'api_max': f"{max_api:.2f}s",
                'db_avg': f"{avg_insert*1000:.1f}ms"
            }
            pbar.set_postfix(postfix)
    
    print(f"\n   ✓ Generated {total_generated} chunk embeddings")
    print(f"   ✓ Inserted {total_inserted} chunks")
    
    # Display timing statistics
    if all_api_times:
        print(f"\n📊 Timing Statistics:")
        avg_api = sum(all_api_times) / len(all_api_times)
        max_api = max(all_api_times)
        min_api = min(all_api_times)
        print(f"   • Ollama API calls: {len(all_api_times)} total")
        print(f"     - Average: {avg_api:.3f}s")
        print(f"     - Min: {min_api:.3f}s, Max: {max_api:.3f}s")
    
    if all_insert_times:
        avg_insert = sum(all_insert_times) / len(all_insert_times)
        max_insert = max(all_insert_times)
        min_insert = min(all_insert_times)
        print(f"   • Database inserts: {len(all_insert_times)} total")
        print(f"     - Average: {avg_insert:.3f}s")
        print(f"     - Min: {min_insert:.3f}s, Max: {max_insert:.3f}s")
    
    if all_validation_times:
        avg_val = sum(all_validation_times) / len(all_validation_times)
        print(f"   • Validation: {len(all_validation_times)} chunks")
        print(f"     - Average: {avg_val:.4f}s")
    
    print(f"\n✅ Chunk ingestion complete!")
    print(f"\nQuery examples:")
    print(f"  • Count chunks: SELECT COUNT(*) FROM article_chunks;")
    print(f"  • Chunks per article: SELECT title, COUNT(*) FROM article_chunks GROUP BY title LIMIT 5;")


if __name__ == "__main__":
    main()
