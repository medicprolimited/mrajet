from bs4 import BeautifulSoup
import re
import logging

# Set up logger
logger = logging.getLogger('article_scraper')

def extract_article_text(html_content: str) -> str:
    """
    Extract main article text from HTML content using BeautifulSoup.

    Args:
        html_content (str): Raw HTML content of the webpage.

    Returns:
        str: Cleaned article text, or empty string if extraction fails.
    """
    logger.info(f"Starting article extraction from {len(html_content)} bytes of HTML")

    try:
        # Create BeautifulSoup object
        logger.debug("Creating BeautifulSoup object with lxml parser")
        soup = BeautifulSoup(html_content, 'lxml')

        # Remove script, style, and boilerplate elements
        logger.debug("Removing script, style, and boilerplate elements")
        elements_before = len(list(soup.descendants))
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'iframe']):
            script_or_style.decompose()

        # Remove metadata-related elements by class
        metadata_classes = [
            'author', 'byline', 'mol-publication-date',
            'social-details', 'share-actions', 'feed-shared-social-action-bar'  # LinkedIn-specific
        ]
        for class_name in metadata_classes:
            for elem in soup.find_all(class_=re.compile(class_name, re.I)):
                elem.decompose()

        elements_after = len(list(soup.descendants))
        logger.debug(f"Removed {elements_before - elements_after} elements")

        # Look for article or main content containers
        article_candidates = []
        logger.debug("Searching for main content containers")
        for container in [
            'article', 'main', '.post-content', '.entry-content',
            '.article-body', '.content', 'feed-shared-update-v2__description'
        ]:
            elements = soup.select(container)
            if elements:
                logger.debug(f"Found {len(elements)} elements matching '{container}'")
                article_candidates.extend(elements)

        # Fallback to paragraph elements if no containers found
        if not article_candidates:
            logger.debug("No specific content containers found, falling back to all paragraphs")
            article_candidates = soup.find_all('p')
            logger.debug(f"Found {len(article_candidates)} paragraph elements")

        # Extract and clean text from candidates
        logger.debug("Extracting and cleaning text from candidate elements")
        extracted_text = []
        too_short_count = 0
        for candidate in article_candidates:
            text = candidate.get_text(separator=' ', strip=True)
            if text and len(text) > 50:  # Ignore very short snippets
                extracted_text.append(text)
            elif text:
                too_short_count += 1

        logger.debug(f"Extracted {len(extracted_text)} text segments (ignored {too_short_count} too-short segments)")

        # Combine extracted text
        combined_text = ' '.join(extracted_text)
        logger.debug(f"Combined text length: {len(combined_text)} characters")

        # Final text cleanup
        logger.debug("Performing final text cleanup")
        cleaned_text = re.sub(r'\s+', ' ', combined_text)  # Normalize whitespace
        cleaned_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', cleaned_text)  # Remove control chars
        cleaned_text = re.sub(
            r'https?://\S+|#\w+|\b(Like|Comment|Share|Copy|LinkedIn|Facebook|Twitter|'
            r'Report this post|1w|Reply|Reactions|Reaction|See more comments|Report this|'
            r'\d{1,3}(?:,\d{3})*)\b',
            '', cleaned_text, flags=re.IGNORECASE
        )  # Remove LinkedIn-specific patterns

        logger.info(f"Extraction complete. Final text length: {len(cleaned_text)} characters")
        return cleaned_text

    except Exception as e:
        logger.error(f"Error extracting article text: {str(e)}")
        return ""