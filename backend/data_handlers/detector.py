import os
import json
import numpy as np
import time
import logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logger
logger = logging.getLogger('detector')

def detect_misinformation(chunks: List[str], model: SentenceTransformer) -> List[Dict[str, any]]:
    """
    Detect misinformation in text chunks by comparing with known misinformation statements.

    Args:
        chunks (List[str]): List of text chunks to analyze.
        model (SentenceTransformer): Pre-loaded SentenceTransformer model.

    Returns:
        List[Dict[str, any]]: List of dictionaries containing detected misinformation and metadata.
    """
    logger.info(f"Starting misinformation detection on {len(chunks)} text chunks")
    start_time = time.time()

    # Load misinformation statements
    logger.debug("Loading misinformation statements")
    statements_path = os.path.join('data', 'statements', 'misinfo_statements.json')
    try:
        with open(statements_path, 'r') as f:
            misinfo_data = json.load(f)
        logger.debug(f"Loaded misinformation data from {statements_path}")
    except Exception as e:
        logger.error(f"Error loading misinformation statements: {str(e)}")
        raise

    misinfo_statements = misinfo_data['statements']
    logger.info(f"Loaded {len(misinfo_statements)} misinformation statements")

    # Get all statements and their categories
    statements = [item['statement'] for item in misinfo_statements]
    categories = [item['category'] for item in misinfo_statements]

    # Log category distribution
    category_counts = {}
    for category in categories:
        category_counts[category] = category_counts.get(category, 0) + 1
    logger.debug("Misinformation categories distribution:")
    for category, count in category_counts.items():
        logger.debug(f"  - {category}: {count} statements")

    # Define category-specific thresholds
    category_thresholds = {
        "safety_claims": 0.73,
        "health_impact": 0.74,
        "marketing_claims": 0.80,
        "evidence_denial": 0.73,
        "regulation_claims": 0.75,
        "secondhand_exposure": 0.73,
        "chemical_content": 0.77,
        "cessation_claims": 0.73,
        "addiction_claims": 0.73,
        "long_term_effects": 0.75
    }
    default_threshold = 0.80
    logger.debug("Using category-specific similarity thresholds:")
    for category, threshold in category_thresholds.items():
        logger.debug(f"  - {category}: {threshold}")
    logger.debug(f"Default threshold for unlisted categories: {default_threshold}")

    # Encode misinformation statements
    logger.debug(f"Encoding {len(statements)} misinformation statements")
    encode_start = time.time()
    misinfo_embeddings = model.encode(statements, convert_to_tensor=True)
    logger.debug(f"Encoded misinformation statements in {time.time() - encode_start:.2f} seconds")

    # Encode chunks
    logger.debug(f"Encoding {len(chunks)} text chunks")
    encode_start = time.time()
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    logger.debug(f"Encoded text chunks in {time.time() - encode_start:.2f} seconds")

    # Convert tensors to numpy arrays for similarity calculation
    logger.debug("Converting tensors to numpy arrays for similarity calculation")
    misinfo_embeddings_np = misinfo_embeddings.cpu().numpy()
    chunk_embeddings_np = chunk_embeddings.cpu().numpy()

    logger.debug(f"Embeddings shape - chunks: {chunk_embeddings_np.shape}, statements: {misinfo_embeddings_np.shape}")

    # Calculate similarity scores
    logger.debug("Calculating similarity scores using cosine similarity")
    sim_start = time.time()
    similarity_scores = cosine_similarity(chunk_embeddings_np, misinfo_embeddings_np)
    logger.debug(f"Similarity calculation completed in {time.time() - sim_start:.2f} seconds")

    # Find matches that exceed threshold
    logger.debug("Finding matches that exceed threshold")
    results = []
    matches_by_category = {}

    for i, chunk in enumerate(chunks):
        for j, statement in enumerate(statements):
            category = categories[j]
            threshold = category_thresholds.get(category, default_threshold)
            similarity = similarity_scores[i][j]

            # Near miss logging
            if logger.isEnabledFor(logging.DEBUG):
                if similarity >= threshold * 0.9 and similarity < threshold:
                    logger.debug(f"Near miss - Category: {category}, Similarity: {similarity:.4f}, Threshold: {threshold}")
                    logger.debug(f"Chunk: \"{chunk[:100]}...\"")
                    logger.debug(f"Statement: \"{statement}\"")

            # Only add to results if it exceeds the threshold
            if similarity >= threshold:
                # Track matches by category for logging
                matches_by_category[category] = matches_by_category.get(category, 0) + 1
                results.append({
                    'chunk': chunk,
                    'statement': statement,
                    'category': category,
                    'confidence': float(similarity),
                })

    logger.info(f"Found {len(results)} initial matches above threshold")
    # Log matches by category
    for category, count in matches_by_category.items():
        logger.debug(f"  - {category}: {count} matches")

    # Sort results by confidence score (descending)
    logger.debug("Sorting results by confidence score")
    results.sort(key=lambda x: x['confidence'], reverse=True)

    # Remove duplicates (keep highest confidence for each chunk)
    logger.debug("Removing duplicate chunks (keeping highest confidence for each)")
    unique_chunks = set()
    filtered_results = []

    for result in results:
        chunk = result['chunk']
        if chunk not in unique_chunks:
            unique_chunks.add(chunk)
            filtered_results.append(result)

    logger.info(f"Final detection results: {len(filtered_results)} unique chunks with misinformation")

    # Log some high-confidence examples
    if filtered_results and logger.isEnabledFor(logging.DEBUG):
        logger.debug("Sample high-confidence matches:")
        for i, result in enumerate(filtered_results[:3]):  # Log top 3 examples
            truncated_chunk = result['chunk'][:50] + "..." if len(result['chunk']) > 50 else result['chunk']
            logger.debug(f"  {i+1}. {result['category']} ({result['confidence']:.3f}): \"{truncated_chunk}\"")

    logger.info(f"Misinformation detection completed in {time.time() - start_time:.2f} seconds")
    return filtered_results