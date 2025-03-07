import os
import re
import html
import json
import random
import unicodedata
from langdetect import detect
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

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Chunks documents into smaller sections and adds chunk number to metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config_loader.chunk_size,
        chunk_overlap=config_loader.chunk_overlap,
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
    Cleans and normalizes text by fixing Unicode issues, removing unwanted patterns, and improving readability.
    """
    try:
        # 1. Normalize Unicode characters (fix ambiguous characters)
        # Example: Converts full-width characters to standard width, ligatures to normal letters
        # "ＡＢＣ" → "ABC", "ﬁ" → "fi"
        text = unicodedata.normalize("NFKC", text)
        
        # 2. Decode escaped Unicode sequences
        # Example: Converts "\u2013" to "–" (en-dash), "\n" to actual newline
        text = bytes(text, "utf-8").decode("unicode_escape")
        
        # 3. Replace escaped newlines with actual newlines
        # Example: "Hello\\nWorld" → "Hello\nWorld"
        text = re.sub(r'\\n', '\n', text)

        # 4. Remove unnecessary placeholders like "<EOS> <pad>"
        # Example: "Hello <EOS> <pad> World" → "Hello World"
        text = re.sub(r'\s*<EOS>\s*<pad>\s*', ' ', text)

        # 5. Decode HTML entities
        # Example: "Tom &amp; Jerry" → "Tom & Jerry"
        text = html.unescape(text)

        # 6. Fix encoding issues (double encoding)
        # Example: "MÃ¼ller" → "Müller"
        text = text.encode('latin1').decode('utf-8', 'ignore')

        # 7. Remove long sequences of dots (3 or more in a row)
        # Example: "Sektor . . . . . . . . . . 117–118" → "Sektor 117–118"
        text = re.sub(r"\s*\.\s*(?:\.\s*)+", " ", text)

        # 8. Remove page number ranges (e.g., "[117–118]", "[25]", [ ])
        # Example: "Technisch unvermeidbare Abwärme [121–125]" → "Technisch unvermeidbare Abwärme"
        text = re.sub(r"\[\s*\d+(?:[,-]\d+)*\s*\]|\[\s*\]", "", text)


        # 9. Fix words that have been incorrectly split by spaces (common OCR issue)
        # Example: "E n e r g y   M a n a g e m e n t" → "EnergyManagement"
        text = re.sub(
            r"\b([A-ZÄÖÜa-zäöüß])(?:\s+([A-ZÄÖÜa-zäöüß]))+\b",
            lambda m: m.group(0).replace(" ", ""),  # Merge split words
            text,
        )

        # 10. Remove specified words (if a list of words is provided)
        # Example: If words_to_remove = ["Technisch", "Abwärme"]
        # "Technisch unvermeidbare Abwärme" → "unvermeidbare"
        if words_to_remove:
            pattern = (
                r"\b(" + "|".join(re.escape(word) for word in words_to_remove) + r")\b"
            )
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # 11. Remove excessive spaces and newlines
        # Example: "Sektor    Technisch    unvermeidbare" → "Sektor Technisch unvermeidbare"
        text = re.sub(r"\s+", " ", text).strip()

        return text
    
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text
        


def detect_language(text):
    """Detects the language of a given text."""
    try:
        return detect(text)
    except:
        return "unknown"


def compute_bm25_scores(documents):
    """
    Compute BM25 scores for each document chunk relative to all other chunks.
    """
    # Tokenize document chunks
    tokenized_corpus = [doc.page_content.lower().split() for doc in documents]

    # Initialize BM25 model
    bm25 = BM25Okapi(tokenized_corpus)

    # Assign BM25 scores to metadata
    for i, doc in enumerate(documents):
        # Compute BM25 scores for the current document against all others
        bm25_scores = bm25.get_scores(tokenized_corpus[i])

        # Store the BM25 score (taking the average for simplicity)
        doc.metadata["bm25_score"] = float(sum(bm25_scores) / len(bm25_scores))

    return documents


def save_all_chunks_to_json(
    documents, directory, base_filename="chunks", max_chunks_per_file=500
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
                "bm25_score": doc.metadata.get("bm25_score", 0),
                "language": doc.metadata.get("language", "en"),
                "keywords": doc.metadata.get("keywords", []),
            }
            for doc in chunk_subset
        ]

        filename = os.path.join(directory, f"{base_filename}_chunk_{i+1}.json")

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=4)

    logger.info(f"Saved {total_chunks} chunks across {num_files} file(s).")


def save_random_chunks_to_json(documents, filename="random_chunks.json", num_samples=5):
    """Save randomly selected document chunks to a JSON file."""
    if not documents:
        print("No documents available to save.")
        return

    sampled_docs = random.sample(documents, min(num_samples, len(documents)))

    chunk_data = [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source", "Unknown Source"),
            "chunk_num": doc.metadata.get("chunk_num", 0),
            "bm25_score": doc.metadata.get("bm25_score", 0),
            "language": doc.metadata.get("language", "en"),
        }
        for doc in sampled_docs
    ]

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=4)

    logger.info(f"Saved {len(sampled_docs)} random chunks to {filename}")


def saperate_docs_by_language(chunked_documents):
    english_docs = []
    german_docs = []

    for doc in chunked_documents:
        lang = detect_language(doc.page_content)
        if lang == "en":
            doc.metadata["language"] = "en"
            english_docs.append(doc)
        elif lang == "de":
            doc.metadata["language"] = "de"
            german_docs.append(doc)
        else:
            doc.metadata["language"] = "en"
            english_docs.append(doc)
            logger.info(
                f"Unknown language chunk: {lang} of doc '{doc.metadata.get("source", "Unknown Source")}'. Defaulting to English."
            )

    return english_docs, german_docs


if __name__ == "__main__":
    sources = [
        "artifacts/data/Deutsche",
        "artifacts/data/English",
    ]

    chunked_documents = load_documents(sources)
    english_docs, german_docs = saperate_docs_by_language(chunked_documents)

    # Print English and German documents
    print(f"Loaded {len(chunked_documents)} document chunks")
    print(f"Number of English documents: {len(english_docs)}")
    print(f"Number of German documents: {len(german_docs)}")
    print("-" * 20)

    # Print chunked documents
    # for doc in chunked_documents[:1]:
    #     print(doc)
    #     print("-" * 20)

    # for doc in english_docs[:1]:
    #     print(doc)
    #     print("-" * 20)

    # for doc in german_docs[:1]:
    #     print(doc)
    #     print("-" * 20)

    save_random_chunks_to_json(
        english_docs, "artifacts/random_english_chunks.json", num_samples=50
    )
    save_random_chunks_to_json(
        german_docs, "artifacts/random_german_chunks.json", num_samples=50
    )
