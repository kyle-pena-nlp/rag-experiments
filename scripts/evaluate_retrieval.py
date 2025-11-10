"""
Evaluate retrieval performance by querying articles with questions.

This script:
1. Loads questions from wikipedia_questions.json
2. Generates embeddings for each question
3. Performs cosine similarity search against article database
4. Saves top N retrieved articles for each question
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import psycopg2
import requests
from tqdm import tqdm


def load_questions(questions_path: Path) -> List[Dict]:
    """
    Load questions from JSON file.
    
    Args:
        questions_path: Path to questions JSON file
        
    Returns:
        List of question dictionaries
    """
    with open(questions_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_embedding(text: str, ollama_host: str = "http://localhost:11434", model: str = "nomic-embed-text") -> Optional[List[float]]:
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
        print(f"\n⚠️  Error generating embedding: {e}")
        return None


def search_similar_articles(
    conn: psycopg2.extensions.connection,
    query_embedding: List[float],
    top_k: int = 5
) -> List[Dict]:
    """
    Search for similar articles using cosine similarity.
    
    Args:
        conn: Database connection
        query_embedding: Query embedding vector
        top_k: Number of results to return
        
    Returns:
        List of similar articles with scores
    """
    cursor = conn.cursor()
    
    # Convert embedding to PostgreSQL vector format
    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
    
    query = """
        SELECT 
            id,
            title,
            content,
            1 - (embedding <=> %s::vector) AS similarity_score
        FROM articles
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    
    cursor.execute(query, (embedding_str, embedding_str, top_k))
    
    results = []
    for row in cursor.fetchall():
        results.append({
            'id': str(row[0]),
            'title': row[1],
            'content': row[2][:500],  # First 500 chars for preview
            'similarity_score': float(row[3])
        })
    
    cursor.close()
    return results


def evaluate_questions(
    questions: List[Dict],
    conn: psycopg2.extensions.connection,
    ollama_host: str,
    top_k: int = 5
) -> List[Dict]:
    """
    Evaluate all questions and retrieve matching articles.
    
    Args:
        questions: List of question dictionaries
        conn: Database connection
        ollama_host: Ollama API host
        top_k: Number of articles to retrieve per question
        
    Returns:
        List of results with questions and retrieved articles
    """
    results = []
    
    for item in tqdm(questions, desc="Evaluating"):
        question = item.get('question', '')
        source_title = item.get('title', '')
        
        if not question:
            continue
        
        # Generate embedding for question
        question_embedding = generate_embedding(question, ollama_host)
        
        if not question_embedding:
            results.append({
                'question': question,
                'source_article': source_title,
                'retrieved_articles': [],
                'error': 'Failed to generate embedding'
            })
            continue
        
        # Search for similar articles
        retrieved = search_similar_articles(conn, question_embedding, top_k)
        
        # Check if source article is in retrieved results
        source_found = any(
            article['title'] == source_title 
            for article in retrieved
        )
        
        source_rank = None
        if source_found:
            for idx, article in enumerate(retrieved):
                if article['title'] == source_title:
                    source_rank = idx + 1
                    break
        
        results.append({
            'question': question,
            'source_article': source_title,
            'source_found': source_found,
            'source_rank': source_rank,
            'retrieved_articles': retrieved
        })
    
    return results


def compute_metrics(results: List[Dict], top_k: int) -> Dict:
    """
    Compute retrieval metrics.
    
    Args:
        results: Evaluation results
        top_k: Number of retrieved articles per question
        
    Returns:
        Dictionary of metrics
    """
    total = len(results)
    found = sum(1 for r in results if r.get('source_found', False))
    
    # Mean Reciprocal Rank (MRR)
    mrr_sum = 0
    ranks = []
    for r in results:
        if r.get('source_rank'):
            mrr_sum += 1.0 / r['source_rank']
            ranks.append(r['source_rank'])
    
    mrr = mrr_sum / total if total > 0 else 0
    
    # Average rank when found (conditional on source being found)
    avg_rank_when_found = sum(ranks) / len(ranks) if ranks else None
    
    # Recall@K
    recall_at_k = found / total if total > 0 else 0
    
    return {
        'total_questions': total,
        'source_found_count': found,
        'recall_at_k': recall_at_k,
        'mean_reciprocal_rank': mrr,
        'average_rank_when_found': avg_rank_when_found,
        'top_k': top_k
    }


def save_results(results: List[Dict], metrics: Dict, output_path: Path) -> None:
    """
    Save results to JSON file.
    
    Args:
        results: Evaluation results
        metrics: Computed metrics
        output_path: Output file path
    """
    output_data = {
        'metrics': metrics,
        'results': results
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval performance with Wikipedia questions"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="./data/questions/wikipedia_questions.json",
        help="Path to questions JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/evaluation/retrieval_results.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of articles to retrieve per question (default: 5)"
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag_experiments"),
        help="PostgreSQL connection URL"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama API host"
    )
    
    args = parser.parse_args()
    
    questions_path = Path(args.questions)
    output_path = Path(args.output)
    
    if not questions_path.exists():
        print(f"❌ Questions file not found: {questions_path}")
        print("\nGenerate questions first with: generate-questions")
        return
    
    print("🔍 Retrieval Evaluation")
    print("=" * 50)
    print(f"Questions: {questions_path.name}")
    print(f"Top-K: {args.top_k}")
    print(f"Database: {args.db_url.split('@')[1] if '@' in args.db_url else args.db_url}")
    print(f"Ollama: {args.ollama_host}")
    
    # Load questions
    print(f"\n1️⃣  Loading questions...")
    questions = load_questions(questions_path)
    print(f"   Loaded {len(questions)} questions")
    
    # Connect to database
    print(f"\n2️⃣  Connecting to database...")
    try:
        conn = psycopg2.connect(args.db_url)
        
        # Check article count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM articles WHERE embedding IS NOT NULL;")
        article_count = cursor.fetchone()[0]
        cursor.close()
        
        if article_count == 0:
            print("   ❌ No articles with embeddings found in database")
            print("   Run: ingest-wikipedia --max-articles 100")
            conn.close()
            return
        
        print(f"   ✓ Connected - {article_count} articles available")
        
    except Exception as e:
        print(f"   ❌ Database error: {e}")
        return
    
    # Evaluate questions
    print(f"\n3️⃣  Evaluating questions...")
    results = evaluate_questions(questions, conn, args.ollama_host, args.top_k)
    
    conn.close()
    
    # Compute metrics
    print(f"\n4️⃣  Computing metrics...")
    metrics = compute_metrics(results, args.top_k)
    
    print(f"\n📊 Results:")
    print(f"   • Total questions: {metrics['total_questions']}")
    print(f"   • Source found: {metrics['source_found_count']}/{metrics['total_questions']}")
    print(f"   • Recall@{args.top_k}: {metrics['recall_at_k']:.2%}")
    print(f"   • Mean Reciprocal Rank: {metrics['mean_reciprocal_rank']:.3f}")
    if metrics['average_rank_when_found'] is not None:
        print(f"   • Average Rank (when found): {metrics['average_rank_when_found']:.2f}")
    
    # Save results
    print(f"\n5️⃣  Saving results...")
    save_results(results, metrics, output_path)
    print(f"   ✓ Saved to: {output_path}")
    
    print(f"\n✅ Evaluation complete!")
    print(f"\nExample queries to explore results:")
    print(f"  • View results: cat {output_path} | jq '.metrics'")
    print(f"  • Check failures: cat {output_path} | jq '.results[] | select(.source_found == false)'")


if __name__ == "__main__":
    main()
