"""
Start Docker Compose stack with pre-checks and health validation.

This script:
1. Checks if stack is running and stops it if needed
2. Starts the stack with docker compose up -d
3. Waits for all services to be healthy
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True,
        )
        return result
    except subprocess.CalledProcessError as e:
        if not check:
            return e
        print(f"❌ Command failed: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        sys.exit(1)


def is_stack_running() -> bool:
    """Check if Docker Compose stack is running."""
    result = run_command(
        ["docker", "compose", "ps", "-q"],
        check=False,
    )
    return bool(result.stdout.strip())


def stop_stack(remove_ollama: bool = True) -> None:
    """Stop the Docker Compose stack.
    
    Args:
        remove_ollama: Whether to also stop Ollama containers
    """
    print("🛑 Stopping existing stack...")
    cmd = ["docker", "compose"]
    if remove_ollama:
        cmd.extend(["--profile", "ollama"])
    cmd.extend(["down"])
    run_command(cmd, capture=False)
    print("✅ Stack stopped")


def start_stack(with_ollama: bool = True) -> None:
    """Start the Docker Compose stack.
    
    Args:
        with_ollama: Whether to start Ollama containers (default: True)
    """
    print("\n🚀 Starting Docker Compose stack...")
    cmd = ["docker", "compose"]
    if with_ollama:
        cmd.extend(["--profile", "ollama"])
    cmd.extend(["up", "-d"])
    run_command(cmd, capture=False)


def wait_for_health(timeout: int = 60) -> bool:
    """
    Wait for all services to be healthy.
    
    Args:
        timeout: Maximum time to wait in seconds
        
    Returns:
        True if all services are healthy, False otherwise
    """
    print("\n⏳ Waiting for services to be healthy...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        result = run_command(
            ["docker", "compose", "ps", "--format", "json"],
            check=True,
        )
        
        if not result.stdout.strip():
            time.sleep(1)
            continue
        
        # Parse container status
        import json
        containers = []
        for line in result.stdout.strip().split('\n'):
            try:
                containers.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        
        if not containers:
            time.sleep(1)
            continue
        
        # Check if all containers are healthy or running (if no healthcheck)
        all_healthy = True
        for container in containers:
            health = container.get('Health', '')
            state = container.get('State', '')
            name = container.get('Service', 'unknown')
            
            if health:
                if health == 'healthy':
                    print(f"  ✓ {name}: healthy")
                else:
                    print(f"  ⏳ {name}: {health}")
                    all_healthy = False
            elif state == 'running':
                print(f"  ✓ {name}: running (no healthcheck)")
            else:
                print(f"  ⏳ {name}: {state}")
                all_healthy = False
        
        if all_healthy:
            return True
        
        time.sleep(2)
    
    return False


def show_status() -> None:
    """Show the status of running services."""
    print("\n📊 Service Status:")
    run_command(["docker", "compose", "ps"], capture=False)


def setup_ollama_model() -> None:
    """Pull Llama 3.1 8B Instruct model if not already present."""
    print("\n🤖 Setting up Ollama model...")
    
    # Check if model is already pulled
    result = run_command(
        ["docker", "compose", "exec", "-T", "ollama", "ollama", "list"],
        check=False,
    )
    
    if "llama3.1:8b-instruct" in result.stdout:
        print("✓ Llama 3.1 8B Instruct already available")
        return
    
    print("⬇️  Pulling Llama 3.1 8B Instruct model (this may take a few minutes)...")
    print("   Model size: ~4.7GB")
    
    # Pull the model
    result = run_command(
        ["docker", "compose", "exec", "-T", "ollama", "ollama", "pull", "llama3.1:8b-instruct"],
        check=False,
        capture=False,
    )
    
    if result.returncode == 0:
        print("✅ Model ready!")
    else:
        print("⚠️  Model pull may have failed. You can manually pull with:")
        print("   docker compose exec ollama ollama pull llama3.1:8b-instruct")


def main() -> None:
    """Main function to start the stack."""
    parser = argparse.ArgumentParser(
        description="Start Docker Compose stack"
    )
    parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Skip starting Ollama containers (use native Ollama instead)"
    )
    args = parser.parse_args()
    
    print("🐳 Docker Compose Stack Manager\n")
    print("=" * 50)
    
    # Check if stack is already running
    if is_stack_running():
        print("⚠️  Stack is already running")
        # Stop everything including Ollama if no-ollama flag is set
        stop_stack(remove_ollama=True)
    else:
        print("✓ No existing stack found")
    
    if args.no_ollama:
        print("ℹ️  Skipping Ollama containers (--no-ollama flag)")
        print("   Make sure to start Ollama manually on ports 11434-11437\n")
    
    # Start the stack
    start_stack(with_ollama=not args.no_ollama)
    
    # Wait for services to be healthy
    if wait_for_health():
        print("\n✅ All services are healthy!")
    else:
        print("\n⚠️  Warning: Some services may not be fully healthy")
        print("Check logs with: docker compose logs -f")
    
    # Pull Ollama model if needed
    setup_ollama_model()
    
    # Show final status
    show_status()
    
    print("\n" + "=" * 50)
    print("🎉 Stack is ready!")
    print("\nServices:")
    print("  • PostgreSQL:      localhost:5432")
    print("  • Ollama API:      http://localhost:11434")
    print("\nUseful commands:")
    print("  • View logs:       docker compose logs -f")
    print("  • Test Ollama:     curl http://localhost:11434/api/tags")
    print("  • List models:     docker compose exec ollama ollama list")
    print("  • Stop stack:      compose-down")
    print("  • Check status:    docker compose ps")


if __name__ == "__main__":
    main()
