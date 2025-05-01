import re
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logger
logger = logging.getLogger('text_processor')

def process_text(text: str, model: SentenceTransformer) -> List[str]:
    """
    Process raw text by cleaning and chunking it using semantic-based chunking.

    Args:
        text (str): Raw text to process.
        model (SentenceTransformer): Pre-loaded SentenceTransformer model.

    Returns:
        List[str]: List of semantically coherent text chunks.
    """
    logger.info(f"Starting semantic text processing for {len(text)} characters")

    # Clean text
    logger.debug("Cleaning text")
    cleaned_text = clean_text(text)
    logger.debug(f"Text cleaned, new length: {len(cleaned_text)} characters")

    # Split text into semantic chunks
    logger.debug("Splitting text into semantic chunks")
    chunks = semantic_chunk_text(cleaned_text, model)
    logger.debug(f"Text split into {len(chunks)} initial semantic chunks")

    # Filter chunks (remove very short ones)
    initial_chunk_count = len(chunks)
    filtered_chunks = [chunk for chunk in chunks if len(chunk) >= 50]
    logger.info(f"Text processed into {len(filtered_chunks)} chunks (removed {initial_chunk_count - len(filtered_chunks)} short chunks)")

    # Log some chunk statistics
    if filtered_chunks:
        chunk_lengths = [len(chunk) for chunk in filtered_chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths)
        min_length = min(chunk_lengths)
        max_length = max(chunk_lengths)
        logger.debug(f"Chunk statistics - Avg: {avg_length:.1f}, Min: {min_length}, Max: {max_length} characters")

    return filtered_chunks

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace, special characters, etc.

    Args:
        text (str): Text to clean.

    Returns:
        str: Cleaned text.
    """
    logger.debug("Starting text cleaning")
    original_length = len(text)

    # Replace multiple newlines and tabs with single space
    logger.debug("Replacing newlines and tabs")
    cleaned = re.sub(r'[\n\t\r]+', ' ', text)

    # Replace multiple spaces with single space
    logger.debug("Normalizing whitespace")
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()

    new_length = len(cleaned)
    logger.debug(f"Text cleaning complete: {original_length} → {new_length} characters ({original_length - new_length} removed)")

    return cleaned

def get_sentence_embeddings(sentences: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Get embeddings for a list of sentences using the provided SentenceTransformer model.

    Args:
        sentences (List[str]): List of sentences to embed.
        model (SentenceTransformer): Pre-loaded SentenceTransformer model.

    Returns:
        np.ndarray: Array of sentence embeddings.
    """
    try:
        logger.debug(f"Generating embeddings for {len(sentences)} sentences")
        embeddings = model.encode(sentences, show_progress_bar=False)
        logger.debug(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating sentence embeddings: {str(e)}")
        # Fallback to zero embeddings if there's an issue
        return np.zeros((len(sentences), 384))  # 384 is the embedding size for all-MiniLM-L6-v2

def calculate_semantic_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.

    Args:
        embedding1 (np.ndarray): First embedding.
        embedding2 (np.ndarray): Second embedding.

    Returns:
        float: Cosine similarity score.
    """
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(embedding1, embedding2) / (norm1 * norm2)

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex patterns that handle common abbreviations.

    Args:
        text (str): Text to split into sentences.

    Returns:
        List[str]: List of sentences.
    """
    # Define regex pattern for sentence splitting that handles common abbreviations
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s'
    sentences = re.split(pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    logger.debug(f"Split text into {len(sentences)} sentences")
    return sentences

def merge_similar_sentences(sentences: List[str], embeddings: np.ndarray, similarity_threshold: float = 0.6, max_chunk_size: int = 800) -> List[str]:
    """
    Merge sentences with high semantic similarity into chunks.

    Args:
        sentences (List[str]): List of sentences.
        embeddings (np.ndarray): Sentence embeddings.
        similarity_threshold (float): Threshold for combining sentences.
        max_chunk_size (int): Maximum size of a chunk in characters.

    Returns:
        List[str]: List of merged text chunks.
    """
    if not sentences:
        return []

    chunks = []
    current_chunk = [sentences[0]]
    current_embedding = embeddings[0].reshape(1, -1)
    current_length = len(sentences[0])

    for i in range(1, len(sentences)):
        sentence = sentences[i]
        sentence_embedding = embeddings[i].reshape(1, -1)
        sentence_length = len(sentence)

        # Calculate average embedding for current chunk
        avg_embedding = np.mean(current_embedding, axis=0)

        # Calculate similarity between sentence and current chunk
        similarity = calculate_semantic_similarity(avg_embedding, sentence_embedding[0])

        # Decide whether to add to current chunk or start a new one
        if similarity >= similarity_threshold and current_length + sentence_length <= max_chunk_size:
            current_chunk.append(sentence)
            current_embedding = np.vstack([current_embedding, sentence_embedding])
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_embedding = sentence_embedding
            current_length = sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.debug(f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences")
    return chunks

def semantic_chunk_text(text: str, model: SentenceTransformer) -> List[str]:
    """
    Split text into semantically coherent chunks using the provided model.

    Args:
        text (str): Text to split.
        model (SentenceTransformer): Pre-loaded SentenceTransformer model.

    Returns:
        List[str]: List of semantically coherent text chunks.
    """
    logger.debug("Beginning semantic chunking")

    try:
        sentences = split_into_sentences(text)
        if not sentences:
            logger.warning("No sentences identified, falling back to basic chunking")
            return fallback_chunk_text(text)

        if len(sentences) == 1:
            logger.debug("Only one sentence found, returning as single chunk")
            return [text]

        embeddings = get_sentence_embeddings(sentences, model)
        chunks = merge_similar_sentences(sentences, embeddings)
        if not chunks:
            logger.warning("Semantic chunking produced no chunks, falling back to basic chunking")
            return fallback_chunk_text(text)

        logger.debug(f"Semantic chunking completed successfully with {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logger.error(f"Error during semantic chunking: {str(e)}")
        logger.warning("Falling back to basic RecursiveCharacterTextSplitter")
        return fallback_chunk_text(text)

def fallback_chunk_text(text: str) -> List[str]:
    """
    Fallback method to split text using RecursiveCharacterTextSplitter.

    Args:
        text (str): Text to split.

    Returns:
        List[str]: List of text chunks.
    """
    logger.debug("Using fallback chunking method: RecursiveCharacterTextSplitter")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    logger.debug(f"Splitting text of length {len(text)} characters")
    chunks = splitter.split_text(text)
    logger.debug(f"Text successfully split into {len(chunks)} chunks using fallback method")

    if chunks and logger.isEnabledFor(logging.DEBUG):
        sample = chunks[0][:100] + "..." if len(chunks[0]) > 100 else chunks[0]
        logger.debug(f"Sample chunk: \"{sample}\"")

    return chunks