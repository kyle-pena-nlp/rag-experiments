"""
Download news article datasets for RAG experiments.

Options:
- CNN/DailyMail: News articles with summaries (~300K articles)
- Common Crawl News: Massive news archive (billions of articles)
"""

import argparse
import os
from pathlib import Path


def download_cnn_dailymail_info(output_dir: Path) -> None:
    """
    Download CNN/DailyMail dataset using HuggingFace datasets.
    
    Size: ~300K articles with summaries
    Best for: News summarization, QA tasks
    """
    print("\n📰 CNN/DailyMail Dataset")
    print("Size: ~1.5GB")
    print("Articles: ~300,000 news articles with summaries")
    print("Best for: News understanding, summarization tasks")
    
    # Create subfolder
    cnn_dir = output_dir / "cnn_dailymail"
    
    # Check if already downloaded
    if cnn_dir.exists() and any(cnn_dir.iterdir()):
        print(f"\n✅ Already downloaded: {cnn_dir}")
        return
    
    # Try to import datasets
    try:
        from datasets import load_dataset
    except ImportError:
        print("\n⚠️  HuggingFace datasets not installed.")
        print("\nInstall with: pip install datasets")
        print("\nOr add to project: poetry add datasets")
        return
    
    print("\n⬇️  Downloading from HuggingFace...")
    print("⚠️  This is ~1.5GB and may take a while...")
    
    try:
        dataset = load_dataset('cnn_dailymail', '3.0.0')
        cnn_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(cnn_dir))
        
        print(f"\n🎉 Downloaded to: {cnn_dir}")
        print(f"   Train: {len(dataset['train'])} articles")
        print(f"   Validation: {len(dataset['validation'])} articles")
        print(f"   Test: {len(dataset['test'])} articles")
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")


def download_common_crawl_news_info(output_dir: Path) -> None:
    """
    Provide information about Common Crawl News dataset.
    
    Size: Massive (TBs)
    Best for: Large-scale experiments
    """
    print("\n📰 Common Crawl News Information...")
    print("Size: Terabytes (highly configurable)")
    print("Articles: Billions of news articles")
    print("Best for: Large-scale production systems")
    
    print("\nAccess via:")
    print("1. AWS S3: s3://commoncrawl/")
    print("2. Index files available at: https://index.commoncrawl.org/")
    print("\nNote: This is best accessed programmatically with filters")
    print("to download only what you need.")


def download_newsapi_sample(output_dir: Path) -> None:
    """
    Provide instructions for using NewsAPI for recent news.
    """
    print("\n📰 NewsAPI Sample Collection...")
    print("Best for: Recent news articles (current events)")
    print("Requires: Free API key from https://newsapi.org")
    
    print("\nSample code:")
    print("  pip install newsapi-python")
    print("  ")
    print("  from newsapi import NewsApiClient")
    print("  newsapi = NewsApiClient(api_key='YOUR_API_KEY')")
    print("  articles = newsapi.get_everything(q='technology', page_size=100)")


def download_ag_news(output_dir: Path) -> None:
    """
    Download AG News dataset using HuggingFace datasets.
    
    Size: ~120K articles
    Best for: News categorization, smaller dataset
    """
    print("\n📰 AG News Dataset")
    print("Size: ~30MB")
    print("Articles: ~120,000 news articles")
    print("Categories: World, Sports, Business, Sci/Tech")
    
    # Create subfolder for ag news
    ag_news_dir = output_dir / "ag_news"
    
    # Check if already downloaded
    if ag_news_dir.exists() and any(ag_news_dir.iterdir()):
        print(f"\n✅ Already downloaded: {ag_news_dir}")
        return
    
    # Try to import datasets
    try:
        from datasets import load_dataset
    except ImportError:
        print("\n⚠️  HuggingFace datasets not installed.")
        print("\nInstall with: pip install datasets")
        print("\nOr add to project: poetry add datasets")
        return
    
    print("\n⬇️  Downloading from HuggingFace...")
    
    try:
        dataset = load_dataset('ag_news')
        ag_news_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(ag_news_dir))
        
        print(f"\n🎉 Downloaded to: {ag_news_dir}")
        print(f"   Train: {len(dataset['train'])} articles")
        print(f"   Test: {len(dataset['test'])} articles")
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")


def download_ag_news_info(output_dir: Path) -> None:
    """
    Provide information about AG News dataset (legacy function).
    """
    download_ag_news(output_dir)


def main() -> None:
    """Main function to show news dataset options."""
    parser = argparse.ArgumentParser(
        description="Download news datasets for RAG experiments"
    )
    parser.add_argument(
        "--type",
        choices=["cnn-dailymail", "common-crawl", "newsapi", "ag-news"],
        default="ag-news",
        help="Type of news dataset (default: ag-news)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/news",
        help="Output directory for downloads"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.type == "cnn-dailymail":
        download_cnn_dailymail_info(output_dir)
    elif args.type == "common-crawl":
        download_common_crawl_news_info(output_dir)
    elif args.type == "newsapi":
        download_newsapi_sample(output_dir)
    else:
        download_ag_news_info(output_dir)


if __name__ == "__main__":
    main()
