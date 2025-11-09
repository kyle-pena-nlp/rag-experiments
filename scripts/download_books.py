"""
Download book datasets for RAG experiments.

Options:
- Project Gutenberg: 70K+ public domain books
- BookCorpus: Large collection of novels (11K books)
"""

import argparse
import os
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


def download_gutenberg_sample(output_dir: Path, num_books: int = 100) -> None:
    """
    Download sample books from Project Gutenberg.
    
    Args:
        output_dir: Directory to save books
        num_books: Number of books to download (default: 100)
    """
    print("\n📚 Project Gutenberg Sample")
    print("Source: Public domain literature")
    print("Format: Plain text")
    
    # Create subfolder for gutenberg books
    gutenberg_dir = output_dir / "gutenberg"
    gutenberg_dir.mkdir(parents=True, exist_ok=True)
    
    # Popular book IDs from Project Gutenberg
    popular_books = [
        (84, "Frankenstein"),
        (1342, "Pride_and_Prejudice"),
        (2701, "Moby_Dick"),
        (1661, "Sherlock_Holmes"),
        (11, "Alice_in_Wonderland"),
        (1952, "The_Yellow_Wallpaper"),
        (16, "Peter_Pan"),
        (174, "The_Picture_of_Dorian_Gray"),
        (345, "Dracula"),
        (1260, "Jane_Eyre"),
        (145, "Middlemarch"),
        (98, "A_Tale_of_Two_Cities"),
        (1400, "Great_Expectations"),
        (46, "A_Christmas_Carol"),
        (2600, "War_and_Peace"),
    ]
    
    # Limit to available books or requested number
    books_to_download = popular_books[:min(num_books, len(popular_books))]
    
    # Check which books are already downloaded
    existing_books = set()
    for book_id, title in books_to_download:
        output_path = gutenberg_dir / f"{book_id}_{title}.txt"
        if output_path.exists():
            existing_books.add(book_id)
    
    if existing_books:
        print(f"\n✅ Already have {len(existing_books)} books")
    
    # Download missing books
    to_download = [(bid, title) for bid, title in books_to_download if bid not in existing_books]
    
    if not to_download:
        print(f"\n🎉 All {len(books_to_download)} books already downloaded!")
        print(f"   Location: {gutenberg_dir}")
        return
    
    print(f"\n⬇️  Downloading {len(to_download)} new books...")
    
    downloaded = 0
    for book_id, title in tqdm(to_download, desc="Downloading books"):
        output_path = gutenberg_dir / f"{book_id}_{title}.txt"
        
        try:
            url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            download_file(url, output_path)
            downloaded += 1
        except Exception:
            # Try alternative URL format
            try:
                url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                download_file(url, output_path)
                downloaded += 1
            except Exception:
                # Silently skip - tqdm will show progress
                pass
    
    print(f"\n🎉 Downloaded {downloaded} new books to: {gutenberg_dir}")
    print(f"   Total books: {len(existing_books) + downloaded}")


def download_gutenberg_instructions(output_dir: Path) -> None:
    """
    Provide instructions for downloading full Gutenberg corpus.
    
    The full corpus is ~60GB and requires rsync.
    """
    print("\n📚 Project Gutenberg Full Corpus Instructions...")
    print("Size: ~60GB (70K+ books)")
    print("Format: Multiple formats (txt, html, epub, etc.)")
    
    print("\nTo download the full corpus using rsync:")
    print("  rsync -av --del --include='*.txt' --exclude='*' \\")
    print("    aleph.gutenberg.org::gutenberg \\")
    print(f"    {output_dir}")
    
    print("\n⚠️  This will take several hours and requires significant bandwidth.")


def download_bookcorpus_info() -> None:
    """
    Provide information about BookCorpus dataset.
    
    Note: Original BookCorpus is no longer directly available due to copyright.
    """
    print("\n📚 BookCorpus Information...")
    print("⚠️  Note: Original BookCorpus is no longer publicly available.")
    print("\nAlternatives:")
    print("1. HuggingFace 'bookcorpusopen' - recreation with open books")
    print("   pip install datasets")
    print("   from datasets import load_dataset")
    print("   dataset = load_dataset('bookcorpusopen')")
    print("\n2. Use Project Gutenberg instead (shown above)")


def main() -> None:
    """Main function to download book datasets."""
    parser = argparse.ArgumentParser(
        description="Download book datasets for RAG experiments"
    )
    parser.add_argument(
        "--type",
        choices=["gutenberg-sample", "gutenberg-full", "bookcorpus"],
        default="gutenberg-sample",
        help="Type of dataset (default: gutenberg-sample)"
    )
    parser.add_argument(
        "--num-books",
        type=int,
        default=100,
        help="Number of books for sample mode (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/books",
        help="Output directory for downloads"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.type == "gutenberg-sample":
        download_gutenberg_sample(output_dir, args.num_books)
    elif args.type == "gutenberg-full":
        download_gutenberg_instructions(output_dir)
    else:
        download_bookcorpus_info()


if __name__ == "__main__":
    main()
