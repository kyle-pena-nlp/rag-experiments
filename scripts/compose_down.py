"""
Stop Docker Compose stack gracefully.

This script stops all running containers and removes networks.
Use --volumes flag to also remove volumes (deletes all data).
"""

import argparse
import subprocess
import sys


def run_command(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
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
        print(f"❌ Command failed: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        sys.exit(1)


def is_stack_running() -> bool:
    """Check if Docker Compose stack is running."""
    result = run_command(
        ["docker", "compose", "ps", "-q"],
        check=True,
        capture=True,
    )
    return bool(result.stdout.strip())


def stop_stack(remove_volumes: bool = False) -> None:
    """
    Stop the Docker Compose stack.
    
    Args:
        remove_volumes: If True, also remove volumes (deletes all data)
    """
    cmd = ["docker", "compose", "down"]
    
    if remove_volumes:
        print("⚠️  Removing volumes - all data will be deleted!")
        cmd.append("-v")
    
    print("🛑 Stopping Docker Compose stack...")
    run_command(cmd, capture=False)
    print("✅ Stack stopped successfully")


def main() -> None:
    """Main function to stop Docker Compose stack."""
    parser = argparse.ArgumentParser(
        description="Stop Docker Compose stack"
    )
    parser.add_argument(
        "--volumes",
        "-v",
        action="store_true",
        help="Also remove volumes (deletes all data)"
    )
    
    args = parser.parse_args()
    
    print("🐳 Docker Compose Stack Manager\n")
    print("=" * 50)
    
    # Check if stack is running
    if not is_stack_running():
        print("ℹ️  No running stack found")
        print("=" * 50)
        return
    
    # Confirm if removing volumes
    if args.volumes:
        print("\n⚠️  WARNING: This will delete all database data!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != "yes":
            print("Cancelled.")
            return
    
    # Stop the stack
    stop_stack(remove_volumes=args.volumes)
    
    print("=" * 50)
    print("🎉 Stack stopped!")
    
    if not args.volumes:
        print("\nNote: Data is preserved. Use --volumes to delete data.")


if __name__ == "__main__":
    main()
