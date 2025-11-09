"""
Setup script for RAG experiments environment.

This script:
1. Checks Docker Compose stack is running
2. Pulls required Ollama models (LLM + embeddings)
3. Initializes database schema
4. Verifies everything is working
"""

import argparse
import os
import subprocess
import sys
import time

import psycopg2
import requests


def run_command(cmd: list[str], capture_output: bool = False) -> tuple[int, str]:
    """
    Run a shell command.
    
    Args:
        cmd: Command and arguments as list
        capture_output: Whether to capture and return output
        
    Returns:
        Tuple of (return_code, output)
    """
    if capture_output:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout + result.stderr
    else:
        result = subprocess.run(cmd)
        return result.returncode, ""


def check_docker_compose() -> bool:
    """Check if Docker Compose stack is running."""
    print("\n1️⃣  Checking Docker Compose stack...")
    
    returncode, output = run_command(
        ["docker", "compose", "ps", "--format", "json"],
        capture_output=True
    )
    
    if returncode != 0:
        print("   ❌ Docker Compose not running")
        print("   Run: compose-up")
        return False
    
    # Check if services are running
    if "ollama" not in output or "postgres" not in output:
        print("   ❌ Required services not found")
        print("   Run: compose-up")
        return False
    
    print("   ✓ Docker Compose stack is running")
    return True


def check_ollama_health(host: str = "http://localhost:11434") -> bool:
    """Check if Ollama is healthy."""
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        response.raise_for_status()
        return True
    except Exception:
        return False


def pull_ollama_model(model: str, container: str = "ollama") -> bool:
    """
    Pull an Ollama model.
    
    Args:
        model: Model name to pull
        container: Container name (ollama, ollama-2, etc.)
        
    Returns:
        True if successful
    """
    print(f"\n   Pulling {model} to {container}...")
    
    returncode, output = run_command(
        ["docker", "compose", "exec", container, "ollama", "pull", model],
        capture_output=False
    )
    
    if returncode == 0:
        print(f"   ✓ {model} ready on {container}")
        return True
    else:
        print(f"   ❌ Failed to pull {model} on {container}")
        return False


def setup_ollama_models(host: str = "http://localhost:11434") -> bool:
    """Setup required Ollama models."""
    print("\n2️⃣  Setting up Ollama models...")
    
    # Check Ollama health
    if not check_ollama_health(host):
        print("   ❌ Ollama is not healthy")
        return False
    
    print("   ✓ Ollama is healthy")
    
    # Check which models are already installed
    try:
        response = requests.get(f"{host}/api/tags")
        models = response.json().get('models', [])
        installed = {m['name'] for m in models}
        print(f"   Currently installed: {', '.join(installed) if installed else 'none'}")
    except Exception as e:
        print(f"   ⚠️  Could not check installed models: {e}")
        installed = set()
    
    # Pull required models (only to first instance - they share the volume)
    required_models = {
        'llama3.1': 'LLM for question generation and chat',
        'nomic-embed-text': 'Embedding model (768-dim)'
    }
    
    print("   Note: All Ollama instances share the same model volume")
    
    success = True
    for model, description in required_models.items():
        # Check if already installed
        if any(model in name for name in installed):
            print(f"   ✓ {model} already installed ({description})")
        else:
            print(f"   📥 {description}")
            # Pull to first instance only (shared volume)
            if not pull_ollama_model(model, "ollama"):
                success = False
    
    return success


def check_database(db_url: str) -> bool:
    """Check database connection and schema."""
    print("\n3️⃣  Checking database...")
    
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Check if pgvector extension exists
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        if not cursor.fetchone():
            print("   ❌ pgvector extension not found")
            cursor.close()
            conn.close()
            return False
        
        print("   ✓ pgvector extension enabled")
        
        # Check if articles table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'articles'
            );
        """)
        
        if not cursor.fetchone()[0]:
            print("   ❌ articles table not found")
            print("   The database needs to be recreated with init.sql")
            cursor.close()
            conn.close()
            return False
        
        print("   ✓ articles table exists")
        
        # Check article count
        cursor.execute("SELECT COUNT(*) FROM articles;")
        count = cursor.fetchone()[0]
        print(f"   ℹ️  {count} articles in database")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.OperationalError as e:
        print(f"   ❌ Cannot connect to database: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Database error: {e}")
        return False


def verify_setup(ollama_host: str, db_url: str) -> bool:
    """Verify the complete setup."""
    print("\n4️⃣  Verifying setup...")
    
    all_good = True
    
    # Test Ollama LLM
    print("   Testing LLM...")
    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": "llama3.1",
                "prompt": "Say 'OK' if you work.",
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            print("   ✓ LLM works")
        else:
            print("   ❌ LLM test failed")
            all_good = False
    except Exception as e:
        print(f"   ❌ LLM error: {e}")
        all_good = False
    
    # Test embeddings
    print("   Testing embeddings...")
    try:
        response = requests.post(
            f"{ollama_host}/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": "test"
            },
            timeout=30
        )
        if response.status_code == 200:
            embedding = response.json().get('embedding')
            if embedding and len(embedding) == 768:
                print(f"   ✓ Embeddings work (768-dim)")
            else:
                print("   ❌ Embedding dimension incorrect")
                all_good = False
        else:
            print("   ❌ Embedding test failed")
            all_good = False
    except Exception as e:
        print(f"   ❌ Embedding error: {e}")
        all_good = False
    
    return all_good


def main() -> None:
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup RAG experiments environment"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama API host"
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag_experiments"),
        help="PostgreSQL connection URL"
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip pulling Ollama models"
    )
    
    args = parser.parse_args()
    
    print("🔧 RAG Experiments Setup")
    print("=" * 50)
    
    # Check Docker Compose
    if not check_docker_compose():
        sys.exit(1)
    
    # Setup Ollama models
    if not args.skip_models:
        if not setup_ollama_models(args.ollama_host):
            print("\n⚠️  Some models failed to install")
            print("You can try again later or pull manually:")
            print("  docker compose exec ollama ollama pull llama3.1")
            print("  docker compose exec ollama ollama pull nomic-embed-text")
    else:
        print("\n⏭️  Skipping model setup")
    
    # Check database
    if not check_database(args.db_url):
        print("\n⚠️  Database setup incomplete")
        print("To recreate the database with proper schema:")
        print("  1. docker compose down -v")
        print("  2. compose-up")
        sys.exit(1)
    
    # Verify everything
    if verify_setup(args.ollama_host, args.db_url):
        print("\n" + "=" * 50)
        print("✅ Setup complete! Ready to use:")
        print("\n📚 Data ingestion:")
        print("  ingest-wikipedia --max-articles 100")
        print("\n💬 Question generation:")
        print("  generate-questions")
        print("\n🔍 Query examples:")
        print("  docker compose exec postgres psql -U postgres -d rag_experiments")
    else:
        print("\n⚠️  Setup incomplete - see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
