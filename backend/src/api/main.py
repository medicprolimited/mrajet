from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import logging
import os
import time
import traceback
from data_handlers.article_scraper import extract_article_text
from data_handlers.text_processor import process_text
from data_handlers.detector import detect_misinformation
from data_handlers.knowledge_base import match_counter_arguments
from utils.report_generator import generate_report
from models.model_loader import load_chunking_model, load_detection_model

app = FastAPI()

# CORS configuration for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://apps.medicpro.london"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create logs directory if it doesn’t exist
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler(os.path.join('logs', 'app.log'), mode='a')  # Output to file
    ]
)

# Set specific loggers to DEBUG for detailed output
logging.getLogger('detector').setLevel(logging.DEBUG)
logging.getLogger('knowledge_base').setLevel(logging.DEBUG)

logger = logging.getLogger('fastapi_app')

# Load models once at startup
chunking_model = load_chunking_model()
detection_model = load_detection_model()

@app.post("/analyze")
async def analyze_url(request: Request):
    start_time = time.time()
    logger.info("=== Starting new analysis request ===")
    try:
        # Get URL from request body
        data = await request.json()
        url = data.get('url')
        logger.info(f"Received POST request for URL: {url}")
        if not url:
            logger.warning("No URL provided in request")
            return {"error": "Please provide a URL"}

        # Fetch HTML content
        try:
            logger.info(f"Fetching HTML content from URL: {url}")
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10)
                response.raise_for_status()  # Raise exception for 4XX/5XX responses
            html_content = response.text
            logger.info(f"Successfully fetched HTML content: {len(html_content)} bytes")
        except httpx.RequestError as e:
            logger.error(f"Failed to fetch URL: {str(e)}")
            return {"error": f"Failed to fetch URL: {str(e)}"}

        # Extract article text
        logger.info("Extracting article text using BeautifulSoup")
        article_text = extract_article_text(html_content)
        if not article_text or len(article_text) < 100:
            logger.warning(f"Insufficient text extracted: {len(article_text) if article_text else 0} characters")
            return {"error": "Could not extract meaningful text from the URL"}
        logger.info(f"Successfully extracted article text: {len(article_text)} characters")

        # Process text (chunk and clean)
        logger.info("Processing and chunking text")
        chunks = process_text(article_text, chunking_model)
        logger.info(f"Text processed into {len(chunks)} chunks")

        # Detect misinformation
        logger.info("Starting misinformation detection")
        detection_results = detect_misinformation(chunks, detection_model)
        logger.info(f"Detected {len(detection_results)} potential misinformation instances")

        # Match counter arguments
        logger.info("Matching counter-arguments from knowledge base")
        enriched_results = match_counter_arguments(detection_results, detection_model)
        logger.info(f"Enriched {len(enriched_results)} results with counter-arguments")

        # Generate report
        logger.info("Generating report")
        report = generate_report(enriched_results, url)
        logger.info(f"Report generated with {len(report)} characters")

        # Prepare response
        processing_time = time.time() - start_time
        response_data = {
            "result": enriched_results,
            "report": report,
            "url": url,
            "processing_time": processing_time
        }
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        logger.info("=== Analysis request completed successfully ===")
        return response_data

    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        logger.error(traceback.format_exc())  # Log full stack trace
        logger.error("=== Analysis request failed ===")
        return {"error": f"An error occurred: {str(e)}"}