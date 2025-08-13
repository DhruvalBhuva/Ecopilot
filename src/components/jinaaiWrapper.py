from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Union
import torch
from sklearn.preprocessing import normalize


class JinaaiWrapper:
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        normalize_embeddings: bool = True,
        device: str = None
    ):
        """
        Wrapper for Jina Embeddings v4 with GPU support.
        
        Args:
            model_name: Hugging Face model name or path
            normalize_embeddings: Whether to L2 normalize output
            device: "cuda", "cpu", or None (auto-detect)
        """
        # Detect device if not specified
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.normalize_embeddings = normalize_embeddings

        print(
            f"✅ Jina Embedder initialized on {self.device.upper()} "
            f"(normalization={'ON' if normalize_embeddings else 'OFF'})"
        )

    def _process_embeddings(
        self, embeddings: Union[torch.Tensor, List[torch.Tensor]]
    ) -> np.ndarray:
        """
        Convert tensor to numpy array and normalize if required.
        Always returns a numpy array.
        """
        if isinstance(embeddings, list):
            embeddings_np = torch.stack(embeddings).float().cpu().numpy()
        else:
            embeddings_np = embeddings.float().cpu().numpy()

        # Ensure 2D shape
        if embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)

        if self.normalize_embeddings:
            embeddings_np = normalize(embeddings_np, norm="l2")

        return embeddings_np

    def embed_documents(
        self, texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Embed one or more documents.
        Returns:
            - Single document → List[float]
            - Multiple documents → List[List[float]]
        """
        if isinstance(texts, str):
            texts = [texts]  # Convert to list

        with torch.no_grad():
            embeddings = self.model.encode_text(
                texts=texts, task="retrieval", prompt_name="passage"
            )

        result = self._process_embeddings(embeddings)

        # Return flat vector if only one text
        if len(result) == 1:
            return result[0].tolist()
        return result.tolist()

    def embed_query(
        self, queries: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Embed one or more queries.
        Returns:
            - Single query → List[float]
            - Multiple queries → List[List[float]]
        """
        if isinstance(queries, str):
            queries = [queries]

        with torch.no_grad():
            embeddings = self.model.encode_text(
                texts=queries, task="retrieval", prompt_name="query"
            )

        result = self._process_embeddings(embeddings)

        # Return flat vector if only one query
        if len(result) == 1:
            return result[0].tolist()
        return result.tolist()

    def count_tokens(self, texts, batch_size=16) -> int:
        """
        Count tokens in a list of texts using the tokenizer.
        """
        if isinstance(texts, str):
            texts = [texts]

        token_counts = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_token_counts = [len(self.tokenizer.encode(text)) for text in batch]
            token_counts.extend(batch_token_counts)

        return sum(token_counts)


# 🧪 Example usage
if __name__ == "__main__":
    embedder = JinaaiWrapper()

    # Example documents
    docs = [
        "Climate change impacts on coastal cities include rising sea levels.",
        "Urban planning must adapt to increased flooding risks.",
    ]

    # Embed multiple docs
    print("\nEmbedding multiple documents...")
    doc_embeds = embedder.embed_documents(docs)
    print(f"→ Embeddings shape: ({len(doc_embeds)}, {len(doc_embeds[0])})")

    # Embed single doc
    print("\nEmbedding single document...")
    single_doc_embed = embedder.embed_documents(docs[0])
    print(f"→ Single embedding shape: ({len(single_doc_embed)},)")

    # Example query
    query = "How does climate change affect coastal cities?"
    print("\nEmbedding single query...")
    query_embed = embedder.embed_query(query)
    print(f"→ Query embedding shape: ({len(query_embed)},)")

    # Token count
    print("\nCounting tokens...")
    token_count = embedder.count_tokens(docs)
    print(f"→ Total tokens: {token_count}")
