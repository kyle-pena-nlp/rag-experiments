#!/usr/bin/env python3
"""
Check table sizes and row counts for articles and article_chunks tables.
"""

import argparse
import os
import sys

import psycopg2


def format_bytes(bytes_value):
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def get_table_stats(conn, table_name):
    """
    Get row count and size statistics for a table.
    
    Returns:
        Dict with row_count, table_size, index_size, total_size (all in bytes except row_count)
    """
    cursor = conn.cursor()
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    row_count = cursor.fetchone()[0]
    
    # Get size information
    cursor.execute(f"""
        SELECT 
            pg_relation_size('{table_name}') as table_size,
            pg_indexes_size('{table_name}') as index_size,
            pg_total_relation_size('{table_name}') as total_size
    """)
    
    table_size, index_size, total_size = cursor.fetchone()
    
    cursor.close()
    
    return {
        'row_count': row_count,
        'table_size': table_size,
        'index_size': index_size,
        'total_size': total_size
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Check table sizes and row counts"
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
    
    args = parser.parse_args()
    
    try:
        # Connect to database
        conn = psycopg2.connect(args.db_url)
        
        print("\n" + "="*70)
        print(" 📊 DATABASE TABLE STATISTICS")
        print("="*70)
        
        # Articles table
        print("\n📄 ARTICLES TABLE")
        print("-" * 70)
        articles_stats = get_table_stats(conn, 'articles')
        
        print(f"  Rows:          {articles_stats['row_count']:,}")
        print(f"  Table Size:    {format_bytes(articles_stats['table_size']):>15}")
        print(f"  Index Size:    {format_bytes(articles_stats['index_size']):>15}")
        print(f"  Total Size:    {format_bytes(articles_stats['total_size']):>15}")
        
        if articles_stats['row_count'] > 0:
            avg_per_row = articles_stats['total_size'] / articles_stats['row_count']
            print(f"  Avg/Row:       {format_bytes(avg_per_row):>15}")
        
        # Article chunks table
        print("\n🧩 ARTICLE_CHUNKS TABLE")
        print("-" * 70)
        chunks_stats = get_table_stats(conn, 'article_chunks')
        
        print(f"  Rows:          {chunks_stats['row_count']:,}")
        print(f"  Table Size:    {format_bytes(chunks_stats['table_size']):>15}")
        print(f"  Index Size:    {format_bytes(chunks_stats['index_size']):>15}")
        print(f"  Total Size:    {format_bytes(chunks_stats['total_size']):>15}")
        
        if chunks_stats['row_count'] > 0:
            avg_per_row = chunks_stats['total_size'] / chunks_stats['row_count']
            print(f"  Avg/Row:       {format_bytes(avg_per_row):>15}")
        
        # Summary
        print("\n📈 SUMMARY")
        print("-" * 70)
        total_rows = articles_stats['row_count'] + chunks_stats['row_count']
        total_size = articles_stats['total_size'] + chunks_stats['total_size']
        
        print(f"  Total Rows:    {total_rows:,}")
        print(f"  Total Size:    {format_bytes(total_size):>15}")
        
        # Chunks per article (only for articles that have chunks)
        if chunks_stats['row_count'] > 0:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT article_id) FROM article_chunks;")
            articles_with_chunks = cursor.fetchone()[0]
            cursor.close()
            
            if articles_with_chunks > 0:
                chunks_per_article = chunks_stats['row_count'] / articles_with_chunks
                print(f"  Articles with Chunks: {articles_with_chunks:,}")
                print(f"  Chunks/Article: {chunks_per_article:.2f}")
        
        # Storage efficiency
        if chunks_stats['row_count'] > 0:
            chunk_overhead_pct = (chunks_stats['index_size'] / chunks_stats['total_size']) * 100
            print(f"  Chunk Index Overhead: {chunk_overhead_pct:.1f}%")
        
        print("\n" + "="*70 + "\n")
        
        conn.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
