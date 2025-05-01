# Vaping Misinformation Detection Tool

A modern web application that uses AI to detect and analyze potential misinformation in vaping-related content. The tool provides evidence-based analysis and reliable information to help users distinguish between factual and misleading content about vaping.

## Features

- **Content Analysis**: Analyze text content for potential misinformation
- **Real-time Processing**: Quick and efficient analysis of submitted content
- **Evidence-based Results**: Provides counter-arguments and supporting evidence
- **Modern UI**: Clean, responsive design with intuitive user interface
- **Detailed Reports**: Comprehensive analysis with detailed breakdowns

## Tech Stack

### Frontend
- HTML5
- CSS3 (Modern styling with CSS variables)
- JavaScript (Vanilla JS)
- Responsive design
- Font Awesome icons

### Backend
- FastAPI (Python web framework)
- Machine Learning models for text analysis
- Natural Language Processing
- BeautifulSoup4 for text extraction
- Sentence Transformers for semantic analysis

## Project Structure

```
├── backend/
│   ├── data/                   # Data files
│   ├── data_handlers/          # Data processing modules
│   │   ├── article_scraper.py
│   │   ├── text_processor.py
│   │   ├── detector.py
│   │   └── knowledge_base.py
│   ├── logs/                   # Application logs
│   ├── models/                 # ML model storage
│   │   └── fine_tuned/
│   │       └── vaping_misinfo_model/
│   ├── src/
│   │   └── api/
│   │       └── main.py         # FastAPI application
│   ├── utils/
│   │   └── report_generator.py
│   └── models/
│       └── model_loader.py
├── frontend/
│   ├── index.html             # Main HTML page
│   ├── scripts/
│   │   └── app.js            # Frontend logic
│   └── styles/
│       └── main.css          # Styling
└── requirements.txt          # Python dependencies
```

## Setup and Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd mrajet
```

2. Set up the Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the backend server:
```bash
cd backend/src/api
uvicorn main:app --reload
```

5. Serve the frontend:
   - Use any static file server (e.g., Python's built-in server):
```bash
cd frontend
python -m http.server 3000
```

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Enter or paste the content you want to analyze in the text area
3. Click "Analyze Content"
4. View the results and detailed analysis

## API Endpoints

- `POST /api/analyze`: Analyze content for misinformation
  - Request body: `{ "content": "text to analyze" }`
  - Response: Analysis results with counter-arguments

- `POST /api/contact`: Submit contact form
  - Request body: `{ "email": "user@example.com", "message": "message content" }`
  - Response: Success/error message

## Development

### Frontend Development
- The frontend uses vanilla JavaScript for simplicity and performance
- CSS is organized using CSS variables for easy theming
- Responsive design ensures compatibility across devices

### Backend Development
- FastAPI provides automatic API documentation at `/docs`
- Logging is configured for debugging and monitoring
- Models are loaded at startup for optimal performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Specify your license here]

## Contact

[Add contact information] 