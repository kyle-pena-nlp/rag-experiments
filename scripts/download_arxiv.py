"""
Download arXiv papers dataset for RAG experiments.

Dataset: Academic papers from arXiv.org
Size: Configurable from 1K to 2M+ papers
Best for: Scientific/technical content, citation networks
"""

import argparse
import json
import os
import tarfile
import xml.etree.ElementTree as ET
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


def download_arxiv_metadata(output_dir: Path) -> None:
    """
    Download arXiv metadata dataset.
    
    Size: ~5GB compressed
    Papers: 2M+ papers with titles, abstracts, categories, citations
    Best for: Metadata search, categorization, smaller memory footprint
    """
    print("\n📄 Downloading arXiv Metadata Dataset...")
    print("Size: ~5GB compressed")
    print("Papers: 2M+ papers (metadata only: titles, abstracts, etc.)")
    print("Best for: Testing without downloading full PDFs")
    
    # Kaggle dataset - requires kaggle API
    print("\n⚠️  This requires Kaggle API credentials.")
    print("1. Install kaggle: pip install kaggle")
    print("2. Get API credentials from https://www.kaggle.com/settings")
    print("3. Place kaggle.json in ~/.kaggle/")
    print("\nThen run:")
    print("  kaggle datasets download -d Cornell-University/arxiv")
    print(f"  unzip arxiv.zip -d {output_dir}")


def download_arxiv_sample(output_dir: Path, num_papers: int = 1000) -> None:
    """
    Download sample of arXiv papers using the arXiv API.
    
    Args:
        output_dir: Directory to save papers
        num_papers: Number of papers to download (default: 1000)
    """
    print("\n📄 arXiv Papers Sample")
    print("Source: arXiv.org API")
    print("Format: JSON metadata (titles, abstracts, authors)")
    
    # Create subfolder for arxiv papers
    arxiv_dir = output_dir / "sample"
    arxiv_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = arxiv_dir / f"arxiv_papers_{num_papers}.json"
    
    # Check if already downloaded
    if output_file.exists():
        with open(output_file, 'r') as f:
            existing_papers = json.load(f)
        print(f"\n✅ Already downloaded: {output_file}")
        print(f"   Papers: {len(existing_papers)}")
        return
    
    print(f"\n⬇️  Downloading {num_papers} papers...")
    
    base_url = "http://export.arxiv.org/api/query"
    papers = []
    batch_size = 100
    
    for start in tqdm(range(0, num_papers, batch_size), desc="Fetching papers"):
        params = {
            "search_query": "all",
            "start": start,
            "max_results": min(batch_size, num_papers - start),
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                paper = {
                    'id': entry.find('atom:id', ns).text if entry.find('atom:id', ns) is not None else '',
                    'title': entry.find('atom:title', ns).text.strip() if entry.find('atom:title', ns) is not None else '',
                    'summary': entry.find('atom:summary', ns).text.strip() if entry.find('atom:summary', ns) is not None else '',
                    'authors': [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns) if author.find('atom:name', ns) is not None],
                    'published': entry.find('atom:published', ns).text if entry.find('atom:published', ns) is not None else '',
                }
                papers.append(paper)
        except Exception as e:
            print(f"\n⚠️  Error fetching batch at {start}: {e}")
            continue
    
    # Save parsed papers
    with open(output_file, 'w') as f:
        json.dump(papers, f, indent=2)
    
    print(f"\n🎉 Downloaded {len(papers)} papers to: {output_file}")


def main() -> None:
    """Main function to download arXiv datasets."""
    parser = argparse.ArgumentParser(
        description="Download arXiv datasets for RAG experiments"
    )
    parser.add_argument(
        "--type",
        choices=["metadata", "sample"],
        default="sample",
        help="Type of dataset (default: sample)"
    )
    parser.add_argument(
        "--num-papers",
        type=int,
        default=1000,
        help="Number of papers for sample mode (default: 1000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/arxiv",
        help="Output directory for downloads"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.type == "metadata":
        download_arxiv_metadata(output_dir)
    else:
        download_arxiv_sample(output_dir, args.num_papers)


if __name__ == "__main__":
    main()
