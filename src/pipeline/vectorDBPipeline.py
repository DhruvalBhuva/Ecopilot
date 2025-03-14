from src.load_config import LoadConfig
from src.logger import logger
from src.components.pineconeWrapper import PineconeWrapper
from src.components.openAIWrapper import OpenAIWrapper
from src.components.jinaaiWrapper import JinaaiWrapper
from src.components.data_preprocessing import (
    load_documents,
    save_all_chunks_to_json,
    compute_bm25_scores,
)
from src.components.postgreSQLWrapper import PostgreSQLWrapper

# Load config
config_loader = LoadConfig()

# ========== Data Loading and chunking ==========
logger.info("Loading documents...")
chunked_documents = load_documents(config_loader.data_sources)

# ========== BM25 Scoring ==========
logger.info("Computing BM25 scores...")
chunked_documents = compute_bm25_scores(chunked_documents)

save_all_chunks_to_json(
    chunked_documents, config_loader.chunks_dict, max_chunks_per_file=500
)

# # ========== OpenAI Embedding ==========
openai_embedder = OpenAIWrapper(
    embedding_model_name=config_loader.embedding_model,
)

# Count tokens for German documents in batches
docs_token_count = openai_embedder.count_tokens(chunked_documents, batch_size=16)
logger.info(f"Total token count for documents: {docs_token_count}")

logger.info("Embedding documents...")
for doc in chunked_documents:
    embedding = openai_embedder.embed_documents(doc.page_content)
    doc.metadata["embedding"] = embedding

# # ========== Pinecone Vector Database ==========
# # Initialize Pinecone
pinecone_wrapper = PineconeWrapper(config_loader.pinecone_index)

# Insert German documents into Pinecone
logger.info("Inserting embeddings into Pinecone...")
pinecone_wrapper.upsert_documents(chunked_documents, index_name=config_loader.pinecone_index)

# ========== PostgreSQL Vector Database ==========
# logger.info("Initializing PostgreSQL with pgvector...")
# postgresql_wrapper = PostgreSQLWrapper()

# logger.info("Inserting embeddings into PostgreSQL...")
# postgresql_wrapper.insert_embeddings(chunked_documents)

# # Close the PostgreSQL connection
# postgresql_wrapper.close_connection()

logger.info("Pipeline completed successfully!")
