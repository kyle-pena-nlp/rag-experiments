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


def create_chunks(text: str, chunk_size: int = 200, overlap: int = 100) -> List[Tuple[int, int]]:
    """
    Create overlapping chunks from text based on word boundaries.
    Uses actual character positions from original text to enable exact reconstruction.
    
    Args:
        text: Input text (will preserve exact whitespace)
        chunk_size: Number of words per chunk
        overlap: Number of words to overlap between chunks
        
    Returns:
        List of (start_char_index, end_char_index) tuples that can be used
        to extract exact text via text[start:end]
    """
    if not text:
        return []
    
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
        
        chunks.append((start_char, end_char))
        
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
) -> Optional[List[float]]:
    """
    Generate embedding for text using Ollama.
    
    Args:
        text: Text to embed
        ollama_host: Ollama API endpoint
        model: Embedding model name
        
    Returns:
        Embedding vector or None on error
    """
    try:
        response = requests.post(
            f"{ollama_host}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        return None


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
) -> bool:
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
        True if inserted successfully
    """
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
        return True
    except Exception as e:
        print(f"\n❌ Insert error for chunk {chunk_id}: {e}")
        conn.rollback()
        cursor.close()
        return False


def process_article(
    article: Dict,
    db_url: str,
    ollama_host: str,
    chunk_size: int,
    overlap: int,
    model: str = "nomic-embed-text"
) -> Tuple[int, int]:
    """
    Process single article into chunks.
    
    Args:
        article: Article dictionary with id, title, content
        db_url: Database URL
        ollama_host: Ollama host
        chunk_size: Words per chunk
        overlap: Overlap words
        model: Embedding model
        
    Returns:
        Tuple of (chunks_generated, chunks_inserted)
    """
    # Check existing chunks
    existing_chunks = get_existing_chunks(db_url, article['id'])
    
    # Create chunks
    content = article['content']
    chunk_boundaries = create_chunks(content, chunk_size, overlap)
    
    # Skip if all chunks already exist
    new_chunks = [i for i in range(len(chunk_boundaries)) if i not in existing_chunks]
    if not new_chunks:
        return (0, 0)
    
    # Process new chunks
    conn = psycopg2.connect(db_url)
    generated = 0
    inserted = 0
    
    for chunk_id in new_chunks:
        start_char, end_char = chunk_boundaries[chunk_id]
        chunk_text = content[start_char:end_char]
        
        # VALIDATION: Verify chunk can be exactly reconstructed
        try:
            validate_chunk_reconstruction(content, start_char, end_char)
        except ValueError as e:
            print(f"\n❌ Validation failed for article '{article['title']}' chunk {chunk_id}: {e}")
            print(f"   Skipping this chunk...")
            continue
        
        # Generate embedding
        embedding = generate_embedding(f"{article['title']}\n\n{chunk_text}", ollama_host, model)
        
        if embedding:
            generated += 1
            # Insert chunk
            if insert_chunk(conn, article['id'], article['title'], chunk_id, start_char, end_char, embedding):
                inserted += 1
    
    # Final commit
    conn.commit()
    conn.close()
    
    return (generated, inserted)


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
    
    total_generated = 0
    total_inserted = 0
    
    for article in tqdm(articles, desc="Processing articles"):
        generated, inserted = process_article(
            article,
            args.db_url,
            args.ollama_host,
            args.chunk_size,
            args.overlap,
            args.embedding_model
        )
        total_generated += generated
        total_inserted += inserted
    
    print(f"\n   ✓ Generated {total_generated} chunk embeddings")
    print(f"   ✓ Inserted {total_inserted} chunks")
    
    print(f"\n✅ Chunk ingestion complete!")
    print(f"\nQuery examples:")
    print(f"  • Count chunks: SELECT COUNT(*) FROM article_chunks;")
    print(f"  • Chunks per article: SELECT title, COUNT(*) FROM article_chunks GROUP BY title LIMIT 5;")


if __name__ == "__main__":
    main()
