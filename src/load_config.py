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
        self._init_azure_vars()

        if self.app_config["vector_db"]["active"] == "pinecone":
            self._init_pinecone_vars()
        if self.app_config["vector_db"]["active"] == "postgresql":
            self._init_postgresql_vars()

        self._init_embeddings_vars()
        self._init_llm_vars()

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
        self.chunks_dict = self.app_config["directories"]["chunks_dict"]

    def _init_data_preprocessing_vars(self) -> None:
        """Initialize data preprocessing configuration"""
        data_config = self.app_config["data_preprocessing"]
        self.chunk_size = data_config["chunking"]["chunk_size"]
        self.chunk_overlap = data_config["chunking"]["chunk_overlap"]

    def _init_azure_vars(self) -> None:
        """Initialize Azure configuration if active."""
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
            
    def _init_embeddings_vars(self) -> None:
        """Initialize embedding model configuration"""
        emb_config = self.app_config["embedding_models"]
        self.embedding_model = emb_config["name"]

    def _init_pinecone_vars(self) -> None:
        """Initialize Pinecone vector database configuration if active."""
        vectorDB_config = self.app_config["vector_db"]

        if vectorDB_config.get("active") == "pinecone":
            pinecone_config = vectorDB_config["providers"]["pinecone"]
            self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
            self.pinecone_environment = pinecone_config["environment"]
            self.pinecone_index = pinecone_config["index_name"]
            self.pinecone_index_dimensions = pinecone_config["dimension"]

    def _init_postgresql_vars(self) -> None:
        """Initialize PostgreSQL (pgvector) configuration if active."""
        vectorDB_config = self.app_config["vector_db"]

        if vectorDB_config.get("active") == "postgresql":
            pg_config = vectorDB_config["postgresql"]
            self.pg_host = os.getenv("POSTGRES_HOST", pg_config["host"])
            self.pg_port = int(os.getenv("POSTGRES_PORT", pg_config["port"]))
            self.pg_database = os.getenv("POSTGRES_DB", pg_config["database"])
            self.pg_user = os.getenv("POSTGRES_USER", pg_config["user"])
            self.pg_password = os.getenv(
                "POSTGRES_PASSWORD", pg_config["password"])
            self.pg_table = pg_config["table"]

    def _init_llm_vars(self) -> None:
        """Initialize Language Model configuration."""
        llm_config = self.app_config["language_models"]

        active_model = llm_config["active_model"]
        active_model_configs = llm_config["models"][active_model]
        
        self.llm_model_name = active_model_configs["model_name"]
        self.llm_api_key = active_model_configs["api_key"]
        self.llm_temperature = active_model_configs["temperature"]
        self.llm_max_tokens = active_model_configs["max_tokens"]
        self.llm_top_p = active_model_configs["top_p"]
        
        self.llm_models_config = {}
        self.llm_system_role = llm_config["system_prompt"]
        for model, model_key in llm_config["models"].items():
            self.llm_models_config[model] = model_key

    def get_llm_model_config(self, provider: str):
        """Return the Language Model config for a given provider."""
        return self.llm_models_config.get(provider, None)


if __name__ == "__main__":
    config_loader = LoadConfig()

    print(config_loader.pinecone_api_key)
