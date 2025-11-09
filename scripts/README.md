# Dataset Download Scripts

Scripts to download various datasets for RAG experiments.

## Available Datasets

### 1. Wikipedia (`download_wikipedia.py`)

**Best for**: General knowledge, encyclopedic content

| Option | Size | Documents | Description |
|--------|------|-----------|-------------|
| Simple English | ~2GB compressed | ~200K articles | Easier language, faster download |
| Full English | ~20GB compressed | ~6.5M articles | Complete Wikipedia dump |

```bash
# Download Simple English Wikipedia (recommended for testing)
poetry run python scripts/download_wikipedia.py --type simple

# Download Full English Wikipedia (large!)
poetry run python scripts/download_wikipedia.py --type english --size full
```

### 2. arXiv Papers (`download_arxiv.py`)

**Best for**: Scientific/academic content, technical documents

| Option | Size | Documents | Description |
|--------|------|-----------|-------------|
| Sample | ~10MB | 1K-10K papers | Metadata via API |
| Full Metadata | ~5GB | 2M+ papers | Titles, abstracts, categories |

```bash
# Download sample of 1000 papers
poetry run python scripts/download_arxiv.py --type sample --num-papers 1000

# Instructions for full metadata dataset
poetry run python scripts/download_arxiv.py --type metadata
```

### 3. Books (`download_books.py`)

**Best for**: Long-form narrative, literary content

| Option | Size | Documents | Description |
|--------|------|-----------|-------------|
| Gutenberg Sample | ~50MB | 100 books | Classic literature |
| Gutenberg Full | ~60GB | 70K+ books | Complete public domain corpus |

```bash
# Download 100 popular books from Project Gutenberg
poetry run python scripts/download_books.py --type gutenberg-sample --num-books 100

# Instructions for full Gutenberg corpus
poetry run python scripts/download_books.py --type gutenberg-full
```

### 4. News Articles (`download_news.py`)

**Best for**: Current events, news content, temporal data

| Option | Size | Documents | Description |
|--------|------|-----------|-------------|
| AG News | ~30MB | 120K articles | News categorization dataset |
| CNN/DailyMail | ~1.5GB | 300K articles | News with summaries |
| Common Crawl | Terabytes | Billions | Massive news archive |

```bash
# AG News (quick, good for testing)
poetry run python scripts/download_news.py --type ag-news

# CNN/DailyMail (larger, with summaries)
poetry run python scripts/download_news.py --type cnn-dailymail
```

## Recommendations by Use Case

### Quick Testing (< 1 hour, < 5GB)
1. **Simple English Wikipedia** - Great general knowledge baseline
2. **AG News** - Good for news/current events
3. **Gutenberg Sample (100 books)** - Literary content

### Medium Scale (1-4 hours, 5-20GB)
1. **arXiv Sample (10K papers)** - Technical/scientific
2. **CNN/DailyMail** - News with summaries
3. **Full English Wikipedia** - Comprehensive knowledge base

### Large Scale (> 4 hours, > 20GB)
1. **Full English Wikipedia** - Complete encyclopedia
2. **arXiv Full Metadata** - All academic papers
3. **Gutenberg Full** - Complete book corpus

## Requirements

Install additional dependencies for dataset downloads:

```bash
poetry add requests tqdm
```

For HuggingFace datasets (optional):
```bash
poetry add datasets
```

For Kaggle datasets (optional):
```bash
poetry add kaggle
```

## Output Structure

All datasets are downloaded to the `data/` directory:

```
data/
├── wikipedia/
├── arxiv/
├── books/
└── news/
```

**Note**: The `data/` directory is gitignored by default.
