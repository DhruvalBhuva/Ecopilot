import os
import re
import html
import json
import random
import tiktoken
import unicodedata
from src.logger import logger
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredURLLoader,
)
from src.load_config import LoadConfig
from langchain.docstore.document import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load config
config_loader = LoadConfig()


def load_documents(sources):
    """Loads and processes documents from URLs, PDF file paths, or folders."""
    documents = []

    for source in sources:
        try:
            if source.startswith("http"):
                loader = UnstructuredURLLoader(urls=[source])
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = source
                    doc.page_content = clean_text(doc.page_content)

                chunked_docs = chunk_documents(docs)
                documents.extend(chunked_docs)

            else:
                source_path = os.path.abspath(source)

                if os.path.isdir(source_path):  # Handle directories
                    for filename in os.listdir(source_path):
                        if filename.lower().endswith(".pdf"):
                            filepath = os.path.join(source_path, filename)
                            docs = load_pdf_with_langchain(filepath)
                            chunked_docs = chunk_documents(docs)  # Chunking PDFs
                            documents.extend(chunked_docs)

                elif source_path.lower().endswith(".pdf"):  # Handle single PDF files
                    docs = load_pdf_with_langchain(source_path)
                    chunked_docs = chunk_documents(docs)
                    documents.extend(chunked_docs)

                else:
                    logger.info(
                        f"Skipping invalid source: {source_path}. Must be a URL, PDF file, or directory."
                    )

        except Exception as e:
            print(f"Error loading {source}: {e}")

    return documents


def load_pdf_with_langchain(pdf_path):
    """Loads PDFs using LangChain's UnstructuredPDFLoader."""
    try:
        loader = UnstructuredPDFLoader(pdf_path)
        docs = loader.load()

        # Extract only the filename (not the full path)
        file_name = os.path.basename(pdf_path)

        # Update metadata
        for doc in docs:
            doc.metadata["source"] = file_name  # Only store the file name
            doc.page_content = clean_text(doc.page_content)

        return docs

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return []



def extract_keywords_tfidf(text, num_keywords=15):
    """
    Extracts top `num_keywords` based on TF-IDF scores for mixed English and German text.
    """
    try:
        # Combine English and German stopwords
        en_stopwords = set(stopwords.words("english")).union(["https"])
        de_stopwords = set(stopwords.words("german")).union(["speaker2", "speaker1", "ähm", "äh", "ja", "nein", "okay"])
        stop_words = en_stopwords.union(de_stopwords)

        # Initialize TF-IDF Vectorizer with combined stopwords
        vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=1000)

        # Fit and transform the text
        tfidf_matrix = vectorizer.fit_transform([text])

        # Get feature names (words) and their TF-IDF scores
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        # Sort words by TF-IDF scores and extract top keywords
        top_keyword_indices = tfidf_scores.argsort()[-num_keywords:][::-1]
        top_keywords = [feature_names[i] for i in top_keyword_indices]

        return top_keywords

    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

# Use OpenAI's tokenizer (same for text-embedding-3-small)
encoding = tiktoken.get_encoding("cl100k_base")
def token_len(text):
    return len(encoding.encode(text))

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Chunks documents into smaller sections and adds chunk number to metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config_loader.chunk_size,
        chunk_overlap=config_loader.chunk_overlap,
        length_function=token_len,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_num"] = idx + 1
            # chunk.metadata["keywords"] = extract_keywords_tfidf(chunk.page_content)
        all_chunks.extend(chunks)
    return all_chunks

def clean_text(text, words_to_remove=None):
    """
    Cleans and normalizes text by fixing Unicode issues, removing unwanted patterns,
    joining broken hyphenated words, and improving readability.
    """
    try:
        import re, html, unicodedata

        # 1. Decode escaped Unicode sequences like "\u2013"
        text = bytes(text, "utf-8").decode("unicode_escape", "ignore")

        # 2. Decode HTML entities like "&amp;" → "&"
        text = html.unescape(text)

        # 3. Re-decode misencoded Unicode characters (e.g., "Ã¼" → "ü")
        text = text.encode('latin1').decode('utf-8', "ignore")

        # 4. Normalize characters (e.g., full-width → ASCII, ligatures)
        text = unicodedata.normalize("NFKC", text)

        # 5. Fix hyphenated line-breaks: "influ-\nenced" → "influenced"
        text = re.sub(r'(\w+)-\s*\n?\s*(\w+)', r'\1\2', text)

        # 6. Replace escaped newlines: "\\n" → actual newline
        text = re.sub(r'\\n', '\n', text)

        # 7. Remove long dotted sequences: ". . . . ." → " "
        text = re.sub(r"\s*\.\s*(?:\.\s*)+", " ", text)

        # 8. Remove page number ranges like "[117–125]", "[25]", "[ ]"
        text = re.sub(r"\[\s*\d+(?:[,-]\d+)*\s*\]|\[\s*\]", "", text)

        # 9. Remove unnecessary placeholders like "<EOS> <pad>"
        text = re.sub(r'\s*<EOS>\s*<pad>\s*', ' ', text)

        # 10. Fix split letters (common in OCR): "E n e r g y" → "Energy"
        text = re.sub(
            r"\b([A-ZÄÖÜa-zäöüß])(?:\s+([A-ZÄÖÜa-zäöüß]))+\b",
            lambda m: m.group(0).replace(" ", ""),
            text,
        )

        # 11. Remove unwanted words (if specified)
        if words_to_remove:
            pattern = r"\b(" + "|".join(re.escape(w) for w in words_to_remove) + r")\b"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # 12. Final whitespace normalization
        text = re.sub(r"\s+", " ", text).strip()

        return text

    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text


def save_all_chunks_to_json(
    documents, directory, max_chunks_per_file=500
):
    """Save all document chunks to JSON files, ensuring the target directory exists."""
    if not documents:
        print("No documents available to save.")
        return

    os.makedirs(directory, exist_ok=True)

    total_chunks = len(documents)
    num_files = (total_chunks // max_chunks_per_file) + (
        1 if total_chunks % max_chunks_per_file > 0 else 0
    )

    for i in range(num_files):
        start_idx = i * max_chunks_per_file
        end_idx = start_idx + max_chunks_per_file
        chunk_subset = documents[start_idx:end_idx]

        chunk_data = [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "Unknown Source"),
                "chunk_num": doc.metadata.get("chunk_num", 0),
                # "embeddings": doc.metadata.get("embedding", []),
            }
            for doc in chunk_subset
        ]

        filename = os.path.join(directory, f"chunk_{i+1}.json")

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=4)

    logger.info(f"Saved {total_chunks} chunks across {num_files} file(s).")


if __name__ == "__main__":
    sources = config_loader.data_sources
    chunked_documents = load_documents(sources)

    # Print chunked documents
    # for doc in chunked_documents[:1]:
    #     print(doc)
    #     print("-" * 20)
