from src.load_config import LoadConfig
from src.logger import logger
from src.components.pineconeWrapper import PineconeWrapper
from src.components.openAIWrapper import OpenAIWrapper
from src.components.jinaaiWrapper import JinaaiWrapper
from src.components.data_preprocessing import (
    load_documents,
    saperate_docs_by_language,
    save_all_chunks_to_json,
    compute_bm25_scores,
    save_random_chunks_to_json,
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

# ========== Data saperation ==========
logger.info("Separating documents by language...")
english_docs, german_docs = saperate_docs_by_language(chunked_documents)

# Print English and German documents
logger.info(f"Loaded {len(chunked_documents)} document chunks")
logger.info(f"Number of English documents: {len(english_docs)}")
logger.info(f"Number of German documents: {len(german_docs)}")

save_all_chunks_to_json(
    german_docs, config_loader.deutsche_chunks_dict, "german", max_chunks_per_file=500
)
save_all_chunks_to_json(
    english_docs, config_loader.english_chunks_dict, "english", max_chunks_per_file=500
)

# ========== jinaai Embedding ==========
jinaai_embedding_model_config = config_loader.get_embedding_model_config("jinaai")
jinaai_embedder = JinaaiWrapper(jinaai_embedding_model_config, config_loader.device)

# Count tokens for German documents in batches
jinaai_token_counts = jinaai_embedder.count_tokens(german_docs, batch_size=16)
logger.info(
    f"The total number of tokens embedded using jinaai of german docs: {jinaai_token_counts}"
)

logger.info("Creating jinaai German embeddings...")
for doc in german_docs:
    embedding = jinaai_embedder.embed_query(doc.page_content)
    doc.metadata["embedding"] = embedding

# ========== Pinecone Database ==========
pinecone_wrapper = PineconeWrapper(
    api_key=config_loader.PINECONE_API_KEY,
    environment=config_loader.pinecone_environment,
    indexes_config=config_loader.pinecone_indexes_config,
)

# Insert German documents into Pinecone
pinecone_wrapper.upsert_documents(
    german_docs,
    index_name=config_loader.get_pinecone_index_config("ecopilot-jinaai-embeddings")[
        "index_name"
    ],
)

# # ========== OpenAI Embedding ==========
openai_embedder = OpenAIWrapper(
    api_key=config_loader.OPENAI_API_KEY,
    embedding_model_name=config_loader.get_embedding_model_config("openai")["name"],
)

# Count tokens for German documents in batches
openai_german_token_counts = openai_embedder.count_tokens(german_docs, batch_size=16)
logger.info(
    f"The total number of tokens embedded using OpenAI of german docs: {openai_german_token_counts}"
)

openai_english_token_counts = openai_embedder.count_tokens(english_docs, batch_size=16)
logger.info(
    f"The total number of tokens embedded using OpenAI of english docs: {openai_english_token_counts}"
)

for doc in german_docs:
    embedding = openai_embedder.embed_documents(doc.page_content)
    doc.metadata["embedding"] = embedding

for doc in english_docs:
    embedding = openai_embedder.embed_documents(doc.page_content)
    doc.metadata["embedding"] = embedding

# Insert German documents into Pinecone
pinecone_wrapper.upsert_documents(
    german_docs,
    index_name=config_loader.get_pinecone_index_config("ecopilot-openai-embeddings")[
        "index_name"
    ],
)

# Insert English documents into Pinecone
pinecone_wrapper.upsert_documents(
    english_docs,
    index_name=config_loader.get_pinecone_index_config("ecopilot-openai-embeddings")[
        "index_name"
    ],
)

# # ========== PostgreSQL Vector Database ==========
# logger.info("Initializing PostgreSQL with pgvector...")
# postgresql_wrapper = PostgreSQLWrapper(
#     host=config_loader.pg_host,
#     port=config_loader.pg_port,
#     database=config_loader.pg_database,
#     user=config_loader.pg_user,
#     password=config_loader.pg_password,
# )

# logger.info("Inserting jinaai German embeddings into PostgreSQL...")
# postgresql_wrapper.insert_embeddings(german_docs, table=config_loader.jinaai_table)

# # Get the total number of records in the table
# total_records = postgresql_wrapper.get_record_count()
# logger.info(f"Total records in the {config_loader.jinaai_table}: {total_records}")

# logger.info("Creating OpenAI German embeddings...")
# for doc in german_docs:
#     embedding = openai_embedder.embed_query(doc.page_content)
#     doc.metadata["embedding"] = embedding

# logger.info("Inserting OpenAI German embeddings into PostgreSQL...")
# postgresql_wrapper.insert_embeddings(german_docs, table=config_loader.openai_table)

# logger.info("Creating openai English embeddings...")
# for doc in english_docs:
#     embedding = openai_embedder.embed_query(doc.page_content)
#     doc.metadata["embedding"] = embedding

# logger.info("Inserting openai English embeddings into PostgreSQL...")
# postgresql_wrapper.insert_embeddings(english_docs, table=config_loader.openai_table)

# # Close the PostgreSQL connection
# postgresql_wrapper.close_connection()

logger.info("Pipeline completed successfully!")
