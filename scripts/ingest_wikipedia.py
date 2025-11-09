"""
Ingest Wikipedia articles into PostgreSQL with vector embeddings.

This script:
1. Parses Simple Wikipedia XML dump
2. Generates embeddings using Ollama (nomic-embed-text)
3. Stores articles with embeddings in PostgreSQL
"""

import argparse
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set

import psycopg2
from psycopg2.extras import execute_batch, Json
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


def parse_wikipedia_xml(xml_path: Path, max_articles: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Parse Wikipedia XML and extract articles.
    
    Args:
        xml_path: Path to Wikipedia XML dump
        max_articles: Maximum number of articles to parse (None for all)
        
    Returns:
        List of article dictionaries with title and content
    """
    print(f"\n📖 Parsing Wikipedia XML: {xml_path.name}")
    
    articles = []
    namespace = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}
    
    # Use iterparse for memory efficiency
    context = ET.iterparse(str(xml_path), events=('end',))
    
    for event, elem in context:
        if elem.tag == '{http://www.mediawiki.org/xml/export-0.11/}page':
            title_elem = elem.find('mw:title', namespace)
            text_elem = elem.find('.//mw:text', namespace)
            
            if title_elem is not None and text_elem is not None and text_elem.text:
                title = title_elem.text
                
                # Skip special pages
                if not any(prefix in title for prefix in ['Wikipedia:', 'Category:', 'Template:', 'File:', 'Help:', 'Portal:']):
                    text = text_elem.text
                    
                    # Only include substantial articles
                    if len(text) > 200:
                        cleaned_text = clean_wikitext(text)
                        
                        # Skip if cleaning removed too much
                        if len(cleaned_text) > 100:
                            articles.append({
                                'title': title,
                                'content': cleaned_text
                            })
                            
                            if max_articles and len(articles) >= max_articles:
                                break
            
            # Clear element to save memory
            elem.clear()
    
    print(f"   Found {len(articles)} articles")
    return articles


# Hardcoded Ollama instances
OLLAMA_INSTANCES = [
    "http://localhost:11434",
    #"http://localhost:11435",
    #"http://localhost:11436",
    #"http://localhost:11437",
]


class WorkCoordinator:
    """
    Thread-safe coordinator for work distribution among workers.
    """
    def __init__(self, articles: List[Dict], db_url: str, existing_titles: Set[str] = None):
        self.articles = articles
        self.db_url = db_url
        self.existing_titles = existing_titles or set()
        self.acquired: Set[int] = set()
        self.lock = threading.Lock()
        self.total_generated = 0
        self.total_inserted = 0
        self.total_skipped = 0
        self.next_index = 0
        
    def get_next_article(self) -> Optional[tuple[int, Dict]]:
        """Get next unacquired article (thread-safe). Skips already-vectorized articles."""
        with self.lock:
            # Find next unacquired article that hasn't been vectorized
            while self.next_index < len(self.articles):
                if self.next_index not in self.acquired:
                    article = self.articles[self.next_index]
                    
                    # Skip if already vectorized
                    if article['title'] in self.existing_titles:
                        self.total_skipped += 1
                        self.next_index += 1
                        continue
                    
                    idx = self.next_index
                    self.acquired.add(idx)
                    self.next_index += 1
                    return (idx, article)
                self.next_index += 1
            return None
    
    def record_success(self, generated: bool, inserted: bool):
        """Record successful embedding/insertion (thread-safe)."""
        with self.lock:
            if generated:
                self.total_generated += 1
            if inserted:
                self.total_inserted += 1
    
    def get_stats(self) -> tuple[int, int, int]:
        """Get current stats (thread-safe)."""
        with self.lock:
            return (self.total_generated, self.total_inserted, self.total_skipped)


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
        Embedding vector or None if failed
    """
    try:
        # Truncate text if too long (nomic-embed-text has 8192 token limit)
        if len(text) > 8000:
            text = text[:8000]
        
        response = requests.post(
            f"{ollama_host}/api/embeddings",
            json={
                "model": model,
                "prompt": text
            },
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get('embedding')
        
    except Exception as e:
        # Don't print errors in concurrent mode to avoid spam
        return None


def get_db_connection(db_url: str) -> psycopg2.extensions.connection:
    """
    Create database connection.
    
    Args:
        db_url: PostgreSQL connection URL
        
    Returns:
        Database connection
    """
    return psycopg2.connect(db_url)


def get_existing_articles(db_url: str, source: str = 'wikipedia') -> Set[str]:
    """
    Get set of article titles that already have embeddings in the database.
    
    Args:
        db_url: PostgreSQL connection URL
        source: Source filter (default: 'wikipedia')
        
    Returns:
        Set of article titles that have been vectorized
    """
    conn = get_db_connection(db_url)
    cursor = conn.cursor()
    
    query = """
        SELECT title 
        FROM articles 
        WHERE source = %s AND embedding IS NOT NULL
    """
    
    cursor.execute(query, (source,))
    existing = {row[0] for row in cursor.fetchall()}
    
    cursor.close()
    conn.close()
    
    return existing


def insert_article(
    conn: psycopg2.extensions.connection,
    article: Dict,
    commit: bool = False
) -> bool:
    """
    Insert single article with embedding into database.
    
    Args:
        conn: Database connection
        article: Article with embedding
        commit: Whether to commit immediately (default: False for batching)
        
    Returns:
        True if inserted successfully
    """
    if not article.get('embedding'):
        return False
    
    cursor = conn.cursor()
    
    insert_query = """
        INSERT INTO articles (title, content, embedding, source, metadata)
        VALUES (%s, %s, %s::vector, %s, %s)
        ON CONFLICT (title, source) DO NOTHING
    """
    
    try:
        cursor.execute(
            insert_query,
            (
                article['title'],
                article['content'],
                article['embedding'],
                'wikipedia',
                Json({'word_count': len(article['content'].split())})
            )
        )
        if commit:
            conn.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"\n❌ Insert error for '{article.get('title', 'unknown')}': {e}")
        conn.rollback()
        cursor.close()
        return False


def worker(
    worker_id: int,
    ollama_host: str,
    coordinator: WorkCoordinator,
    progress_bar: tqdm,
    model: str = "nomic-embed-text",
    batch_size: int = 10
) -> None:
    """
    Worker thread that processes articles using a dedicated Ollama instance.
    
    Args:
        worker_id: Worker identifier
        ollama_host: Dedicated Ollama instance URL
        coordinator: Work coordinator
        progress_bar: Shared progress bar
        model: Embedding model name
        batch_size: Number of articles to process before committing
    """
    # Create dedicated database connection for this worker
    try:
        conn = get_db_connection(coordinator.db_url)
    except Exception as e:
        print(f"\n❌ Worker {worker_id} failed to connect to database: {e}")
        return
    
    articles_since_commit = 0
    
    while True:
        # Get next article to process
        result = coordinator.get_next_article()
        if result is None:
            break  # No more work
        
        idx, article = result
        
        # Generate embedding
        embedding = generate_embedding(
            f"{article['title']}\n\n{article['content']}",
            ollama_host,
            model
        )
        
        generated = embedding is not None
        inserted = False
        
        if generated:
            article['embedding'] = embedding
            # Insert into database (commit every batch_size articles)
            articles_since_commit += 1
            should_commit = articles_since_commit >= batch_size
            inserted = insert_article(conn, article, commit=should_commit)
            
            if should_commit:
                articles_since_commit = 0
        
        # Record stats
        coordinator.record_success(generated, inserted)
        
        # Update progress bar
        progress_bar.update(1)
    
    # Final commit for remaining articles
    if articles_since_commit > 0:
        conn.commit()
    
    # Close connection
    conn.close()


def main() -> None:
    """Main function to ingest Wikipedia articles."""
    parser = argparse.ArgumentParser(
        description="Ingest Wikipedia articles with embeddings into PostgreSQL"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/wikipedia/simple/simplewiki-latest-pages-articles.xml",
        help="Path to Wikipedia XML file"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Maximum number of articles to ingest (default: all)"
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag_experiments"),
        help="PostgreSQL connection URL"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-embed-text",
        help="Ollama embedding model"
    )
    
    args = parser.parse_args()
    
    xml_path = Path(args.input)
    if not xml_path.exists():
        print(f"❌ Wikipedia XML file not found: {xml_path}")
        print("\nDownload it first with: download-wikipedia --type simple")
        return
    
    print("🔄 Wikipedia Ingestion Pipeline")
    print("=" * 50)
    print(f"Input: {xml_path.name}")
    print(f"Database: {args.db_url.split('@')[1] if '@' in args.db_url else args.db_url}")
    print(f"Ollama Instances: {len(OLLAMA_INSTANCES)} workers (one per instance)")
    for i, host in enumerate(OLLAMA_INSTANCES, 1):
        print(f"  Worker {i}: {host}")
    print(f"Model: {args.embedding_model}")
    
    # Check if embedding model is available
    print(f"\n1️⃣  Checking embedding model...")
    try:
        response = requests.get(f"{OLLAMA_INSTANCES[0]}/api/tags")
        models = response.json().get('models', [])
        model_names = [m['name'] for m in models]
        
        if not any(args.embedding_model in name for name in model_names):
            print(f"⚠️  Model '{args.embedding_model}' not found.")
            print(f"   Run: setup")
            return
        else:
            print(f"✓ Model '{args.embedding_model}' available")
            print(f"   (All instances share the same model volume)")
    except Exception as e:
        print(f"⚠️  Could not connect to Ollama: {e}")
        return
    
    # Parse Wikipedia XML
    print(f"\n2️⃣  Parsing Wikipedia XML...")
    articles = parse_wikipedia_xml(xml_path, args.max_articles)
    
    if not articles:
        print("❌ No articles found")
        return
    
    # Check which articles already have embeddings
    print(f"\n3️⃣  Checking existing embeddings...")
    try:
        existing_titles = get_existing_articles(args.db_url)
        already_vectorized = sum(1 for a in articles if a['title'] in existing_titles)
        to_process = len(articles) - already_vectorized
        print(f"   ✓ Found {len(existing_titles)} existing embeddings in database")
        print(f"   ✓ {already_vectorized} articles already vectorized")
        print(f"   ✓ {to_process} articles to process")
    except Exception as e:
        print(f"   ⚠️  Could not check existing articles: {e}")
        print(f"   Proceeding without skip optimization...")
        existing_titles = set()
        to_process = len(articles)
    
    if to_process == 0:
        print("\n✅ All articles already vectorized!")
        return
    
    # Create work coordinator
    print(f"\n4️⃣  Initializing work coordinator...")
    coordinator = WorkCoordinator(articles, args.db_url, existing_titles)
    print(f"   ✓ Ready to process {to_process} new articles")
    
    # Start worker threads (one per Ollama instance)
    print(f"\n5️⃣  Starting {len(OLLAMA_INSTANCES)} worker threads...")
    
    # Create shared progress bar (total includes skipped articles)
    progress_bar = tqdm(total=len(articles), desc="Processing", unit="article")
    
    # Fast-forward progress bar for already-skipped articles
    if existing_titles:
        progress_bar.update(already_vectorized)
    
    # Start workers
    threads = []
    for i, ollama_host in enumerate(OLLAMA_INSTANCES):
        thread = threading.Thread(
            target=worker,
            args=(i + 1, ollama_host, coordinator, progress_bar, args.embedding_model),
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    # Wait for all workers to complete
    for thread in threads:
        thread.join()
    
    progress_bar.close()
    
    # Get final stats
    total_generated, total_inserted, total_skipped = coordinator.get_stats()
    print(f"\n   ✓ Skipped {total_skipped} already-vectorized articles")
    print(f"   ✓ Generated {total_generated} new embeddings")
    print(f"   ✓ Inserted {total_inserted} articles")
    print(f"   ✓ Total in database: {len(existing_titles) + total_inserted}")
    
    print(f"\n✅ Ingestion complete!")
    print(f"\nQuery examples:")
    print(f"  • Count: SELECT COUNT(*) FROM articles;")
    print(f"  • Sample: SELECT title FROM articles LIMIT 5;")
    print(f"  • Vector search: SELECT title FROM articles ORDER BY embedding <=> '[...]' LIMIT 5;")


if __name__ == "__main__":
    main()
