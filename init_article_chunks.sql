-- Create article_chunks table for chunked article embeddings
-- This table stores overlapping chunks of articles for more granular retrieval

CREATE TABLE IF NOT EXISTS article_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    article_id UUID NOT NULL,  -- Reference to parent article
    title TEXT NOT NULL,  -- Denormalized for convenience
    chunk_id INTEGER NOT NULL,  -- Sequential chunk number (0, 1, 2, ...)
    start_char_index INTEGER NOT NULL,  -- Starting character position in cleaned text
    end_char_index INTEGER NOT NULL,  -- Ending character position in cleaned text
    embedding VECTOR(768),  -- nomic-embed-text produces 768-dim embeddings
    source TEXT DEFAULT 'wikipedia',
    metadata JSONB,  -- Can store word_count, overlap info, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(article_id, chunk_id),  -- Prevent duplicate chunks
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);

-- Create index for vector similarity search on chunks
CREATE INDEX IF NOT EXISTS article_chunks_embedding_idx ON article_chunks 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create index for article_id lookup
CREATE INDEX IF NOT EXISTS article_chunks_article_id_idx ON article_chunks (article_id);

-- Create index for source filtering
CREATE INDEX IF NOT EXISTS article_chunks_source_idx ON article_chunks (source);

-- Create index for title search
CREATE INDEX IF NOT EXISTS article_chunks_title_idx ON article_chunks USING gin (title gin_trgm_ops);

-- Trigger to automatically update updated_at
CREATE TRIGGER update_article_chunks_updated_at BEFORE UPDATE ON article_chunks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
