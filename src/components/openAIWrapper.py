import torch
import tiktoken
import openai
from src.logger import logger
from openai import OpenAI
from src.load_config import LoadConfig


class OpenAIWrapper:

    def __init__(
        self,
        api_key,
        embedding_model_name="text-embedding-ada-002",
        llm_model_name="o1-mini",
        temperature=0.7,
        return_torch=True,
    ):
        """
        Initialize the OpenAI embedding model.

        :param api_key: OpenAI API key.
        :param embedding_model_name: Name of the OpenAI embedding model (default: "text-embedding-ada-002").
        :param return_torch: If True, return embeddings as PyTorch tensors (default: False).
                             Note: This is deprecated and will be removed in the future.
        """
        self.client = OpenAI(api_key=api_key)
        self.embedding_model_name = embedding_model_name
        self.return_torch = return_torch  # Deprecated, kept for backward compatibility
        self.tokenizer = tiktoken.encoding_for_model(embedding_model_name)
        self.llm_model_name = llm_model_name
        self.temperature = temperature

    def embed_documents(self, texts):
        """
        Embed a list of documents into dense vectors using OpenAI's embedding model.

        :param texts: List of texts to embed.
        :return: List of embeddings (always as lists of floats).
        """
        try:
            response = self.client.embeddings.create(
                input=texts, model=self.embedding_model_name
            )
            embeddings = [embedding.embedding for embedding in response.data]

            # convert to float
            embeddings = [[float(x) for x in embedding] for embedding in embeddings]

            if self.return_torch:
                return torch.tensor(embeddings)[0]

            return embeddings[0]

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []

    def embed_query(self, text):
        """
        Embed a single query text using OpenAI's embedding model.

        :param text: Query text to embed.
        :return: Embedding vector (always as a list of floats).
        """
        try:
            response = self.client.embeddings.create(
                input=[text], model=self.embedding_model_name
            )
            embedding = response.data[0].embedding

            # Ensure embedding is a list of floats
            embedding = list(map(float, embedding))

            return embedding  # Return as a list of floats

        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return []

    def count_tokens(self, texts, batch_size=16):
        """
        Count tokens for a list of texts using OpenAI's tokenizer.

        :param texts: List of texts to count tokens for.
        :param batch_size: Number of texts to process at once (default: 16).
        :return: Total number of tokens.
        """
        token_counts = []

        # Ensure all documents are strings
        texts = [str(text) for text in texts]

        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Tokenize the batch
            batch_token_counts = [len(self.tokenizer.encode(text)) for text in batch]
            token_counts.extend(batch_token_counts)

        return sum(token_counts)

    def text_generator(self, prompt, max_tokens=100):
        """
        Generate text based on a prompt using OpenAI's language model.

        :param prompt: The prompt text.
        :param max_tokens: The maximum number of tokens to generate (default: 100).
        :return: The generated text.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": "Say this is a test",
                    },
                    # {
                    #     "role": "assistant",
                    #     "content": prompt,
                    # },
                ],
                model=self.llm_model_name,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return None


if __name__ == "__main__":
    # Load config
    config_loader = LoadConfig()

    # Initialize OpenAI embedding model with PyTorch support
    openai_embedder = OpenAIWrapper(
        api_key=config_loader.OPENAI_API_KEY,
        embedding_model_name=config_loader.get_embedding_model_config("openai")["name"],
    )

    # Test embedding
    # print("Query Embedding (Torch):", openai_embedder.embed_query("Hello, world!"))
    # print(
    #     "Document Embeddings (Torch):",
    #     openai_embedder.embed_documents(["This is a test document."]),
    # )
    # print(
    #     "Token Count:",
    #     openai_embedder.count_tokens(["This is a test document."]),
    # )

    # prompt = "Explain the role of OeMAG in promoting renewable energy in Austria."
    # generated_text = openai_embedder.text_generator(prompt)
    # print("Generated Text:", generated_text)
