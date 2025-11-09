# RAG Experiments

A Python project for RAG (Retrieval-Augmented Generation) experiments with PostgreSQL database support.

## Prerequisites

- Python 3.11 or higher
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose for running PostgreSQL locally

## Setup

### 1. Install Dependencies

```bash
poetry install
```

### 2. Configure Environment Variables

Copy the example environment file and update with your settings:

```bash
cp .env.example .env
```

Edit `.env` with your preferred database credentials.

### 3. Start PostgreSQL Database

Start all services (PostgreSQL + Ollama) using the smart startup script:

```bash
compose-up
```

This will:
- Start PostgreSQL and Ollama containers
- Run health checks
- Automatically pull Llama 3.1 8B Instruct model (~4.7GB, first time only)

To stop all services:

```bash
compose-down
```

To stop and remove all data (including model):

```bash
compose-down --volumes
```

### 4. Activate Virtual Environment

```bash
poetry shell
```

## Services

### PostgreSQL Database

The PostgreSQL database will be available at:
- **Host**: localhost
- **Port**: 5432 (or 5433 if port conflict)
- **Database**: rag_experiments
- **Username**: postgres (default)
- **Password**: postgres (default - change in `.env`)

Connection string format:
```
postgresql://postgres:postgres@localhost:5432/rag_experiments
```

### Ollama LLM API (Llama 3.1 8B Instruct)

OpenAI-compatible API endpoint:
- **Base URL**: http://localhost:11434
- **Model**: llama3.1:8b-instruct
- **API Docs**: http://localhost:11434/api/tags

Example usage:
```python
import requests

response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'llama3.1:8b-instruct',
        'prompt': 'Explain RAG in one sentence',
        'stream': False
    }
)
print(response.json()['response'])
```

Or using OpenAI-compatible format:
```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'  # required but unused
)

response = client.chat.completions.create(
    model='llama3.1:8b-instruct',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
```

### Ollama Embeddings

Ollama can also generate embeddings using the nomic-embed-text model:
- **Base URL**: http://localhost:11434
- **Model**: nomic-embed-text (~275MB)
- **Embedding Dimension**: 768

First, pull the embedding model:
```bash
docker compose exec ollama ollama pull nomic-embed-text
```

Example usage:
```python
import requests

# Generate embeddings
response = requests.post(
    'http://localhost:11434/api/embeddings',
    json={
        'model': 'nomic-embed-text',
        'prompt': 'This is a test sentence'
    }
)

embedding = response.json()['embedding']
print(f"Embedding dimension: {len(embedding)}")  # 768
```

Batch embeddings:
```python
import requests

texts = ['First sentence', 'Second sentence', 'Third sentence']
embeddings = []

for text in texts:
    response = requests.post(
        'http://localhost:11434/api/embeddings',
        json={'model': 'nomic-embed-text', 'prompt': text}
    )
    embeddings.append(response.json()['embedding'])

print(f"Generated {len(embeddings)} embeddings")
```

## Project Structure

```
rag-experiments/
├── .gitignore          # Python gitignore rules
├── README.md           # This file
├── pyproject.toml      # Poetry configuration and dependencies
├── docker-compose.yml  # PostgreSQL database setup
├── .env.example        # Environment variables template
└── .env               # Your local environment variables (not tracked)
```

## Development

### Adding Dependencies

```bash
poetry add <package-name>
```

### Adding Development Dependencies

```bash
poetry add --group dev <package-name>
```

### Running Tests

```bash
poetry run pytest
```

## Useful Commands

### View Database Logs

```bash
docker-compose logs -f postgres
```

### Connect to PostgreSQL via CLI

```bash
docker-compose exec postgres psql -U postgres -d rag_experiments
```

### Check Database Status

```bash
docker-compose ps
```

## License

MIT
