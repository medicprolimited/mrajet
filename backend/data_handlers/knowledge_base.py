import os
import json
import logging
import time
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# Set up logger
logger = logging.getLogger('knowledge_base')

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def match_counter_arguments(detection_results: List[Dict[str, any]], model: SentenceTransformer) -> List[Dict[str, any]]:
    """
    Match detected misinformation with counter-arguments from the knowledge base using semantic similarity.

    Args:
        detection_results (List[Dict[str, any]]): List of dictionaries containing detected misinformation.
        model (SentenceTransformer): Pre-loaded SentenceTransformer model for computing similarities.

    Returns:
        List[Dict[str, any]]: Enriched detection results with counter-arguments and sources.
    """
    logger.info(f"Starting counter-argument matching for {len(detection_results)} detection results")
    start_time = time.time()

    # Load the knowledge base
    kb_path = os.path.join('data', 'knowledge_base', 'vaping_facts.json')
    try:
        logger.debug(f"Loading knowledge base from {kb_path}")
        with open(kb_path, 'r') as f:
            kb_data = json.load(f)
        logger.debug("Knowledge base loaded successfully")
    except Exception as e:
        logger.error(f"Error loading knowledge base: {str(e)}")
        raise

    # Log available categories in knowledge base
    kb_categories = list(kb_data['categories'].keys())
    logger.debug(f"Knowledge base contains {len(kb_categories)} categories: {', '.join(kb_categories)}")

    # Enrich detection results with counter-arguments and sources
    enriched_results = []
    categories_matched = set()
    categories_missing = set()

    for result_idx, result in enumerate(detection_results):
        category = result['category']
        statement = result.get('statement', '')
        chunk = result.get('chunk', '')[:100] + '...' if len(result.get('chunk', '')) > 100 else result.get('chunk', '')

        logger.debug(f"Processing result #{result_idx+1} - Category: {category}")
        logger.debug(f"  Statement: \"{statement}\"")
        logger.debug(f"  Chunk: \"{chunk}\"")

        # If we have counter-arguments for this category
        if category in kb_data['categories']:
            categories_matched.add(category)
            category_data = kb_data['categories'][category]
            counter_args = category_data.get('counter_arguments', [])
            sources = category_data.get('sources', [])

            counter_args_count = len(counter_args)
            sources_count = len(sources)
            logger.debug(f"Matching category '{category}': found {counter_args_count} counter-arguments and {sources_count} sources")

            # Rank counter-arguments by semantic similarity (model is always provided)
            if counter_args and statement:
                try:
                    # Log all available counter-arguments before ranking
                    logger.debug(f"===== BEFORE SEMANTIC RANKING - All counter-arguments for {category} =====")
                    for i, counter_arg in enumerate(counter_args):
                        logger.debug(f"  {i+1}. \"{counter_arg[:100]}...\"")

                    # Encode the statement and counter-arguments
                    logger.debug(f"Computing semantic similarity for statement: \"{statement}\"")
                    statement_embedding = model.encode([statement])[0]
                    counter_arg_embeddings = model.encode(counter_args)

                    # Calculate similarities
                    similarities = []
                    logger.debug("===== SIMILARITY SCORES =====")
                    for i, counter_arg_embedding in enumerate(counter_arg_embeddings):
                        similarity = cosine_similarity(statement_embedding, counter_arg_embedding)
                        similarities.append((i, similarity))
                        logger.debug(f"  Counter-argument #{i+1}: {similarity:.4f} - \"{counter_args[i][:100]}...\"")

                    # Sort by similarity (descending)
                    similarities.sort(key=lambda x: x[1], reverse=True)

                    # Get the top 3 (or fewer if there are less) most relevant counter-arguments
                    top_count = min(3, len(counter_args))
                    top_indices = [idx for idx, _ in similarities[:top_count]]
                    ranked_counter_args = [counter_args[idx] for idx in top_indices]

                    # Log the selected counter-arguments
                    logger.debug("===== AFTER SEMANTIC RANKING - Selected counter-arguments =====")
                    for i, (idx, similarity) in enumerate(similarities[:top_count]):
                        logger.debug(f"  {i+1}. Score: {similarity:.4f} - \"{counter_args[idx][:100]}...\"")

                    logger.debug(f"Semantic ranking complete. Top similarity: {similarities[0][1]:.4f}")

                    # Add ranked counter-arguments and sources to the result
                    enriched_result = {
                        **result,
                        'counter_arguments': ranked_counter_args,
                        'sources': sources
                    }
                except Exception as e:
                    logger.error(f"Error during semantic ranking: {str(e)}")
                    # Fall back to using all counter-arguments if semantic ranking fails
                    logger.debug("Falling back to using all counter-arguments")
                    enriched_result = {
                        **result,
                        'counter_arguments': counter_args,
                        'sources': sources
                    }
            else:
                # Use all counter-arguments if no statement or counter-arguments are available
                logger.debug("Using all counter-arguments (no ranking)")
                enriched_result = {
                    **result,
                    'counter_arguments': counter_args,
                    'sources': sources
                }

            enriched_results.append(enriched_result)
        else:
            categories_missing.add(category)
            logger.warning(f"No counter-arguments found for category: {category}")
            # No counter-arguments found for this category, pass through original result
            enriched_results.append(result)

    # Log summary
    logger.info(f"Matched {len(categories_matched)} unique categories with counter-arguments")
    if categories_missing:
        logger.warning(f"Missing counter-arguments for {len(categories_missing)} categories: {', '.join(categories_missing)}")
    logger.info(f"Counter-argument matching completed in {time.time() - start_time:.2f} seconds")

    return enriched_results