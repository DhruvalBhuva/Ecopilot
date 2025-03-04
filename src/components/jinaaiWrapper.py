import torch
from src.load_config import LoadConfig
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


class JinaaiWrapper:
    def __init__(self, model_config, device):
        """
        Initialize the embedding model using configuration.

        :param model_config: Dictionary containing model name and settings.
        """
        # Load model name from config
        model_name = model_config["name"]

        # Set device
        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load embedding model on the correct device
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_documents(self, texts):
        """Embed a list of documents into dense vectors."""
        embeddings = self.model.encode(
            texts, convert_to_tensor=True, device=self.device
        )
        # embeddings = self.extend_to_1536(embeddings)
        return embeddings

    def embed_query(self, text):
        """Embed a single query text."""
        return self.model.encode([text], convert_to_tensor=True, device=self.device)[0]

    def extend_to_1536(self, embedding, target_dim=1536):
        """Extend the given embedding to the target dimension by appending zeros."""
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.unsqueeze(0)  # Add batch dimension if needed
        extended_embedding = torch.zeros(
            (embedding.size(0), target_dim), device=embedding.device
        )
        extended_embedding[:, : embedding.size(1)] = embedding
        return extended_embedding.squeeze(0)

    def count_tokens(self, texts, batch_size=16):
        """Count tokens for a list of texts using batch encoding."""
        token_counts = []

        # Ensure all documents are strings
        texts = [str(text) for text in texts]

        # Split texts into smaller batches to avoid overloading the tokenizer
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Get tokenized inputs for the batch
            tokenized = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )
            # Count tokens for this batch
            batch_token_counts = [len(tokens) for tokens in tokenized["input_ids"]]
            token_counts.extend(batch_token_counts)

        return sum(token_counts)


if __name__ == "__main__":
    # Load config
    config_loader = LoadConfig()

    # Get model config for German embeddings
    jinaai_model_config = config_loader.get_embedding_model_config("jinaai")

    # Initialize embedding model using config
    jinaai_embedder = JinaaiWrapper(jinaai_model_config, config_loader.device)

    # Test embedding
    print("Query Embedding:", jinaai_embedder.embed_query("Hallo, Welt!"))
    print(
        "Token Counts:",
        jinaai_embedder.count_tokens(["Hallo, Welt!", "Wie geht es dir?"]),
    )
