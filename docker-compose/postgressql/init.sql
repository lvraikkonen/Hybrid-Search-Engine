CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embeddings (
  id SERIAL PRIMARY KEY,
  source text,
  text text,
  embedding vector,
  created_at timestamptz DEFAULT now()
);