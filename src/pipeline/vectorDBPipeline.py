from src.load_config import LoadConfig
from src.logger import logger
from src.components.openAIWrapper import OpenAIWrapper
from src.components.jinaaiWrapper import JinaaiWrapper
from src.components.data_preprocessing import (
    load_documents,
    save_all_chunks_to_json,
)
from src.components.postgreSQLWrapper import PostgreSQLWrapper

# Load config
config_loader = LoadConfig()

# ========== Data Loading and chunking ==========
logger.info("Loading documents...")
chunked_documents = load_documents(config_loader.data_sources)
logger.info(f"Total documents loaded: {len(chunked_documents)}")


# save_all_chunks_to_json(
#     chunked_documents, config_loader.chunks_dict, max_chunks_per_file=500
# )

# ========== OpenAI Embedding ==========
# openai_embedder = OpenAIWrapper(
#     embedding_model_name=config_loader.embedding_model,
# )

# # Count tokens for German documents in batches
# docs_token_count = openai_embedder.count_tokens(chunked_documents, batch_size=16)
# logger.info(f"Total token count for documents: {docs_token_count}")

# logger.info("Embedding documents...")
# for doc in chunked_documents:
#     embedding = openai_embedder.embed_documents(doc.page_content)
#     doc.metadata["embedding"] = embedding

# ========== JinaAI Embedding ==========
jinaai_embedder = JinaaiWrapper()

logger.info("Counting tokens for documents...")
docs_token_count = jinaai_embedder.count_tokens(chunked_documents, batch_size=16)
logger.info(f"Total token count for documents: {docs_token_count}")

logger.info("Embedding documents...")
for doc in chunked_documents:
    # Note: We wrap doc.page_content in a list, then take first result
    embedding = jinaai_embedder.embed_documents([doc.page_content])[0]
    doc.metadata["embedding"] = embedding


# ========== PostgreSQL Vector Database ==========
logger.info("Initializing PostgreSQL with pgvector...")
postgresql_wrapper = PostgreSQLWrapper()

logger.info("Inserting embeddings into PostgreSQL...")
postgresql_wrapper.insert_embeddings(chunked_documents)

# Close the PostgreSQL connection
postgresql_wrapper.close_connection()

logger.info("Pipeline completed successfully!")
