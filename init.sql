-- Initial database setup script
-- This script runs automatically when the PostgreSQL container is first created

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create articles table for Wikipedia content
CREATE TABLE IF NOT EXISTS articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768),  -- nomic-embed-text produces 768-dim embeddings
    source TEXT DEFAULT 'wikipedia',
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(title, source)  -- Prevent duplicate articles from same source
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS articles_embedding_idx ON articles USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create index for title search
CREATE INDEX IF NOT EXISTS articles_title_idx ON articles USING gin (title gin_trgm_ops);

-- Create index for source filtering
CREATE INDEX IF NOT EXISTS articles_source_idx ON articles (source);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at
CREATE TRIGGER update_articles_updated_at BEFORE UPDATE ON articles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
