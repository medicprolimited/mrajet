import os
import logging
import time
from sentence_transformers import SentenceTransformer
from typing import Optional

# Set up logger
logger = logging.getLogger('model_loader')

def load_model(fine_tuned_path: str, base_model_name: str) -> SentenceTransformer:
    """
    Load the fine-tuned SentenceTransformer model or fall back to the base model.

    Args:
        fine_tuned_path (str): Path to the fine-tuned model directory.
        base_model_name (str): Name of the base model to fall back to (e.g., 'all-mpnet-base-v2').

    Returns:
        SentenceTransformer: Loaded model instance.
    """
    logger.info("Starting model loading process")
    start_time = time.time()

    logger.info(f"Checking for fine-tuned model at: {fine_tuned_path}")

    try:
        # Try to load fine-tuned model
        if os.path.exists(fine_tuned_path):
            logger.info(f"Fine-tuned model directory found. Checking for required files.")

            # Check for specific model files
            required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(fine_tuned_path, f))]

            if missing_files:
                logger.warning(f"Fine-tuned model directory exists but missing files: {missing_files}")
                logger.warning(f"Falling back to {base_model_name}")
                model_load_start = time.time()
                model = SentenceTransformer(base_model_name)
                logger.info(f"Base model loaded in {time.time() - model_load_start:.2f} seconds")
            else:
                logger.info("All required model files present, loading fine-tuned model")
                model_load_start = time.time()
                model = SentenceTransformer(fine_tuned_path)
                logger.info(f"Fine-tuned model loaded in {time.time() - model_load_start:.2f} seconds")
        else:
            # Fall back to base model
            logger.info(f"Fine-tuned model not found. Falling back to {base_model_name}")
            model_load_start = time.time()
            model = SentenceTransformer(base_model_name)
            logger.info(f"Base model loaded in {time.time() - model_load_start:.2f} seconds")

        # Log model info
        logger.info(f"Model loaded successfully: {type(model).__name__}")
        logger.info(f"Model dimensions: {model.get_sentence_embedding_dimension()}")
        logger.info(f"Total model loading time: {time.time() - start_time:.2f} seconds")

        return model

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(f"Falling back to base model: {base_model_name}")
        model_load_start = time.time()
        model = SentenceTransformer(base_model_name)
        logger.info(f"Base model loaded in {time.time() - model_load_start:.2f} seconds")
        logger.info(f"Total model loading time after fallback: {time.time() - start_time:.2f} seconds")
        return model