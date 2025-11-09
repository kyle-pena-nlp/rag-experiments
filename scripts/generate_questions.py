"""
Generate questions from Wikipedia articles using Ollama LLM.

This script:
1. Parses Simple Wikipedia XML dump
2. Randomly samples 100 articles
3. Generates a meaningful question for each article using Llama 3.1
4. Saves results to JSON file
"""

import argparse
import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set

import psycopg2
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


def get_vectorized_article_titles(db_url: str, num_samples: int, source: str = 'wikipedia') -> List[str]:
    """
    Get random sample of vectorized article titles from database.
    
    Args:
        db_url: PostgreSQL connection URL
        num_samples: Number of articles to sample
        source: Source filter (default: 'wikipedia')
        
    Returns:
        List of article titles that have been vectorized
        
    Raises:
        ValueError: If insufficient vectorized articles available
    """
    print(f"\n📖 Querying vectorized articles from database...")
    
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    
    # First, count available vectorized articles
    count_query = """
        SELECT COUNT(*) 
        FROM articles 
        WHERE source = %s AND embedding IS NOT NULL
    """
    
    cursor.execute(count_query, (source,))
    total_count = cursor.fetchone()[0]
    
    print(f"   Found {total_count} vectorized articles in database")
    
    # Validate we have enough articles
    if total_count < num_samples:
        cursor.close()
        conn.close()
        raise ValueError(
            f"Insufficient vectorized articles: requested {num_samples}, "
            f"but only {total_count} available. "
            f"Run 'ingest-wikipedia' to vectorize more articles."
        )
    
    # Get random sample of article titles
    print(f"\n🎲 Randomly sampling {num_samples} articles...")
    
    select_query = """
        SELECT title
        FROM articles 
        WHERE source = %s AND embedding IS NOT NULL
        ORDER BY RANDOM()
        LIMIT %s
    """
    
    cursor.execute(select_query, (source, num_samples))
    rows = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return [row[0] for row in rows]


def fetch_articles_from_xml(xml_path: Path, titles: List[str]) -> Dict[str, str]:
    """
    Fetch full article text from Wikipedia XML for specified titles.
    
    Args:
        xml_path: Path to Wikipedia XML dump
        titles: List of article titles to fetch
        
    Returns:
        Dictionary mapping title to full cleaned article text
    """
    print(f"\n📖 Fetching full article text from XML: {xml_path.name}")
    print("   This may take a few minutes...")
    
    titles_set = set(titles)
    articles = {}
    namespace = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}
    
    # Use iterparse for memory efficiency
    context = ET.iterparse(str(xml_path), events=('end',))
    
    for event, elem in tqdm(context, desc="Scanning XML", unit=" pages", disable=False):
        if elem.tag == '{http://www.mediawiki.org/xml/export-0.11/}page':
            title_elem = elem.find('mw:title', namespace)
            text_elem = elem.find('.//mw:text', namespace)
            
            if title_elem is not None and text_elem is not None and text_elem.text:
                title = title_elem.text
                
                # Check if this is one of our target articles
                if title in titles_set:
                    raw_text = text_elem.text
                    cleaned_text = clean_wikitext(raw_text)
                    
                    # Truncate to reasonable length for question generation
                    if len(cleaned_text) > 2000:
                        cleaned_text = cleaned_text[:2000] + "..."
                    
                    articles[title] = cleaned_text
                    
                    # Early exit if we found all articles
                    if len(articles) >= len(titles_set):
                        break
            
            # Clear element to save memory
            elem.clear()
    
    print(f"   ✓ Found {len(articles)}/{len(titles)} articles in XML")
    
    return articles




def generate_question(title: str, text: str, ollama_host: str = "http://ollama:11434") -> str:
    """
    Generate a meaningful question about the article using Ollama.
    
    Args:
        title: Article title
        text: Article text
        ollama_host: Ollama API endpoint
        
    Returns:
        Generated question
    """
    prompt = f"""Read this Wikipedia article and generate ONE specific factual question about concrete details mentioned in the content.

The question should:
- Ask about specific facts, numbers, dates, locations, or details from the article
- Be answerable using information in the article
- Avoid generic questions like "What is this article about?"

Examples of GOOD questions:
- "How much does an elephant weigh?" (if article mentions weight)
- "When was the city founded?" (if article mentions founding date)
- "Who discovered this element?" (if article mentions a discovery)

Article Title: {title}

Article Content:
{text}

Generate ONLY the question, nothing else:"""

    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": "llama3.1",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            },
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        question = result.get('response', '').strip()
        
        # Clean up the question
        if not question.endswith('?'):
            question += '?'
        
        return question
        
    except Exception as e:
        print(f"\n⚠️  Error generating question for '{title}': {e}")
        return "What is the main topic of this article?"


def main() -> None:
    """Main function to generate questions from Wikipedia articles."""
    parser = argparse.ArgumentParser(
        description="Generate questions from Wikipedia articles using Ollama"
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
        "--input",
        type=str,
        default="./data/wikipedia/simple/simplewiki-latest-pages-articles.xml",
        help="Path to Wikipedia XML file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/questions/wikipedia_questions.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--num-articles",
        type=int,
        default=100,
        help="Number of articles to sample (default: 100)"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama API host"
    )
    
    args = parser.parse_args()
    
    # Validate XML file exists
    xml_path = Path(args.input)
    if not xml_path.exists():
        print(f"❌ Wikipedia XML file not found: {xml_path}")
        print("\nDownload it first with: download-wikipedia --type simple")
        sys.exit(1)
    
    print("🤖 Wikipedia Question Generator")
    print("=" * 50)
    print(f"\nDatabase: {args.db_url.split('@')[-1]}")
    print(f"XML Source: {xml_path.name}")
    print(f"Requested articles: {args.num_articles}")
    
    # Get vectorized article titles from database
    try:
        titles = get_vectorized_article_titles(args.db_url, args.num_articles)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Database error: {e}")
        sys.exit(1)
    
    # Fetch full article text from XML
    article_texts = fetch_articles_from_xml(xml_path, titles)
    
    # Prepare articles with full text
    articles = []
    for title in titles:
        if title in article_texts:
            articles.append({
                'title': title,
                'text': article_texts[title]
            })
        else:
            print(f"⚠️  Warning: Could not find article '{title}' in XML")
    
    # Generate questions
    print(f"\n✅ Successfully retrieved {len(articles)} articles with full text")
    print(f"\n🤖 Generating questions using Ollama...")
    print(f"   Model: llama3.1:8b-instruct")
    print(f"   Endpoint: {args.ollama_host}\n")
    
    results = []
    
    for article in tqdm(articles, desc="Generating questions"):
        question = generate_question(
            article['title'],
            article['text'],
            args.ollama_host
        )
        
        results.append({
            'title': article['title'],
            'text': article['text'],
            'question': question
        })
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Generated {len(results)} questions")
    print(f"📁 Saved to: {output_path}")
    
    # Show sample
    print("\n📋 Sample questions:")
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. Article: {result['title']}")
        print(f"   Question: {result['question']}")


if __name__ == "__main__":
    main()
