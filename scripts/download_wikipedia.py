"""
Download Wikipedia dataset for RAG experiments.

Options:
- Simple Wikipedia: ~200K articles, easier language, ~2GB compressed
- English Wikipedia: ~6.5M articles, ~20GB compressed (full dump ~90GB uncompressed)
"""

import argparse
import bz2
import os
import shutil
from pathlib import Path

import requests
from tqdm import tqdm


def download_file(url: str, output_path: Path) -> None:
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)


def extract_bz2(input_path: Path, output_path: Path) -> None:
    """Extract a bz2 compressed file with progress bar."""
    print(f"\n📦 Extracting {input_path.name}...")
    
    # Get compressed file size for progress
    compressed_size = input_path.stat().st_size
    
    with bz2.open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out, tqdm(
        desc="Extracting",
        total=compressed_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        while True:
            chunk = f_in.read(8192)
            if not chunk:
                break
            f_out.write(chunk)
            # Update based on input file position
            pbar.update(len(chunk))
    
    print(f"✅ Extracted to: {output_path}")


def download_simple_wikipedia(output_dir: Path) -> None:
    """
    Download Simple English Wikipedia dump.
    
    Size: ~200K articles, ~2GB compressed
    Best for: Quick testing, smaller memory footprint
    """
    print("\n📚 Simple English Wikipedia")
    print("Size: ~2GB compressed (~6GB uncompressed)")
    print("Articles: ~200,000")
    
    # Create subfolder for simple wikipedia
    wiki_dir = output_dir / "simple"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    
    compressed_file = wiki_dir / "simplewiki-latest-pages-articles.xml.bz2"
    extracted_file = wiki_dir / "simplewiki-latest-pages-articles.xml"
    
    # Check if already extracted
    if extracted_file.exists():
        print(f"\n✅ Already extracted: {extracted_file}")
        print(f"   Size: {extracted_file.stat().st_size / (1024**3):.2f} GB")
        return
    
    # Check if already downloaded but not extracted
    if compressed_file.exists():
        print(f"\n✅ Already downloaded: {compressed_file}")
    else:
        # Download the file
        print("\n⬇️  Downloading...")
        url = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
        download_file(url, compressed_file)
        print(f"\n✅ Downloaded to: {compressed_file}")
    
    # Extract the file
    extract_bz2(compressed_file, extracted_file)
    
    # Optionally remove compressed file to save space
    print("\n🗑️  Keep compressed file? (no = delete to save space)")
    keep = input("Keep compressed file? (yes/no): ").lower()
    if keep != "yes":
        compressed_file.unlink()
        print(f"   Deleted: {compressed_file.name}")
    
    print(f"\n🎉 Ready to use: {extracted_file}")


def download_english_wikipedia(output_dir: Path, size: str = "sample") -> None:
    """
    Download English Wikipedia dump.
    
    Sizes:
    - sample: ~100 articles for testing (~10MB)
    - medium: Using Wikipedia API to get ~10K articles
    - full: Complete dump (~20GB compressed, ~90GB uncompressed)
    """
    if size == "sample":
        print("\n📚 Downloading Wikipedia Sample (100 articles)...")
        print("Size: ~10MB")
        print("Best for: Quick testing and development")
        
        # This would need to be implemented using Wikipedia API
        # to fetch a sample of articles
        print("\n⚠️  Sample download via API not implemented in this script.")
        print("Consider using 'simple' mode or 'full' dump instead.")
        
    elif size == "full":
        print("\n📚 Full English Wikipedia")
        print("⚠️  WARNING: This is ~20GB compressed (~90GB uncompressed)")
        print("Articles: ~6.5 million")
        print("This will take a while...")
        
        # Create subfolder for english wikipedia
        wiki_dir = output_dir / "english"
        wiki_dir.mkdir(parents=True, exist_ok=True)
        
        compressed_file = wiki_dir / "enwiki-latest-pages-articles.xml.bz2"
        extracted_file = wiki_dir / "enwiki-latest-pages-articles.xml"
        
        # Check if already extracted
        if extracted_file.exists():
            print(f"\n✅ Already extracted: {extracted_file}")
            print(f"   Size: {extracted_file.stat().st_size / (1024**3):.2f} GB")
            return
        
        # Check if already downloaded but not extracted
        if compressed_file.exists():
            print(f"\n✅ Already downloaded: {compressed_file}")
        else:
            confirm = input("\nAre you sure you want to proceed? (yes/no): ")
            if confirm.lower() != "yes":
                print("Download cancelled.")
                return
            
            # Download the file
            print("\n⬇️  Downloading...")
            url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
            download_file(url, compressed_file)
            print(f"\n✅ Downloaded to: {compressed_file}")
        
        # Extract the file
        extract_bz2(compressed_file, extracted_file)
        
        # Optionally remove compressed file to save space
        print("\n🗑️  Keep compressed file? (no = delete to save ~20GB space)")
        keep = input("Keep compressed file? (yes/no): ").lower()
        if keep != "yes":
            compressed_file.unlink()
            print(f"   Deleted: {compressed_file.name}")
        
        print(f"\n🎉 Ready to use: {extracted_file}")


def main() -> None:
    """Main function to download Wikipedia datasets."""
    parser = argparse.ArgumentParser(
        description="Download Wikipedia datasets for RAG experiments"
    )
    parser.add_argument(
        "--type",
        choices=["simple", "english"],
        default="simple",
        help="Type of Wikipedia to download (default: simple)"
    )
    parser.add_argument(
        "--size",
        choices=["sample", "full"],
        default="full",
        help="Size for English Wikipedia (default: full)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/wikipedia",
        help="Output directory for downloads"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.type == "simple":
        download_simple_wikipedia(output_dir)
    else:
        download_english_wikipedia(output_dir, args.size)


if __name__ == "__main__":
    main()
