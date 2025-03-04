import os
import yaml
import torch
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()


class LoadConfig:
    """
    Secure configuration loader that handles:
    - YAML configuration
    - Environment variables
    - Directory management
    - Service initialization
    """

    def __init__(self, config_path: str = None) -> None:
        # Load YAML configuration
        config_path = "configs/app_configs.yml"
        with open(config_path) as cfg:
            self.app_config = yaml.safe_load(cfg)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_env_vars()
        self._init_data_sources()
        self._init_directories_vars()
        self._init_data_preprocessing_vars()
        self._init_embeddings_vars()

        if self.app_config["vector_db"]["active"] == "pinecone":
            self._init_pinecone_vars()
        if self.app_config["vector_db"]["active"] == "postgresql":
            self._init_postgresql_vars()


    def _init_env_vars(self) -> None:
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

    def _init_data_sources(self) -> None:
        """Load data sources from config file."""
        sources_config = self.app_config.get("data_sources", {})

        self.data_sources = []

        # Ensure directories exist and are a list
        directories = sources_config.get("directories", [])
        if isinstance(directories, list):
            self.data_sources.extend(directories)

        # Ensure links exist and are a list
        links = sources_config.get("links", [])
        if isinstance(links, list):
            self.data_sources.extend(links)

    def _init_directories_vars(self) -> None:
        """Create required directories"""
        self.deutsche_chunks_dict = self.app_config["directories"][
            "deutsche_chunks_dict"
        ]
        self.english_chunks_dict = self.app_config["directories"]["english_chunks_dict"]

    def _init_data_preprocessing_vars(self) -> None:
        """Initialize data preprocessing configuration"""
        data_config = self.app_config["data_preprocessing"]
        self.chunk_size = data_config["chunking"]["chunk_size"]
        self.chunk_overlap = data_config["chunking"]["chunk_overlap"]

    def _init_embeddings_vars(self) -> None:
        """Initialize embedding models configuration"""
        emb_config = self.app_config["embedding_models"]
        self.embedding_active_models = emb_config["active_models"]
        self.embedding_models = {}
        for model, model_key in self.embedding_active_models.items():
            self.embedding_models[model] = emb_config["models"][model_key]

    def get_embedding_model_config(self, provider: str):
        """Return the embedding model config for a given provider."""
        return self.app_config["embedding_models"]["models"].get(provider, None)

    def _init_pinecone_vars(self) -> None:
        """Initialize Pinecone vector database configuration if active."""
        vectorDB_config = self.app_config["vector_db"]

        if vectorDB_config.get("active") == "pinecone":
            pinecone_config = vectorDB_config["providers"]["pinecone"]
            self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
            self.pinecone_environment = pinecone_config["environment"]

            self.pinecone_indexes_config = {}
            for index_alias, index_details in pinecone_config["indexes"].items():
                self.pinecone_indexes_config[index_details["index_name"]] = (
                    index_details  # Use actual index name
                )

    def get_pinecone_index_config(self, index_name: str):
        """Return the Pinecone index config for a given index name."""
        return self.pinecone_indexes_config.get(index_name, None)

    def _init_postgresql_vars(self) -> None:
        """Initialize PostgreSQL (pgvector) configuration if active."""
        vectorDB_config = self.app_config["vector_db"]

        if vectorDB_config.get("active") == "postgresql":
            pg_config = vectorDB_config["providers"]["postgresql"]
            self.pg_host = os.getenv("POSTGRES_HOST", pg_config["host"])
            self.pg_port = int(os.getenv("POSTGRES_PORT", pg_config["port"]))
            self.pg_database = os.getenv("POSTGRES_DB", pg_config["database"])
            self.pg_user = os.getenv("POSTGRES_USER", pg_config["user"])
            self.pg_password = os.getenv("POSTGRES_PASSWORD", pg_config["password"])
            self.jinaai_table = pg_config["jinaai_table"]
            self.openai_table = pg_config["openai_table"]



if __name__ == "__main__":
    config_loader = LoadConfig()

    print(config_loader.get_llm_model_config("llama3-german"))
