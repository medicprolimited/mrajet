document.addEventListener('DOMContentLoaded', () => {
    // Select DOM elements
    const form = document.querySelector('form');
    const urlInput = document.getElementById('url-input');
    const resultDiv = document.getElementById('results');
    const loadingIndicator = document.getElementById('loading');

    // Handle form submission
    form.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission

        const url = urlInput.value.trim();
        if (!url) {
            alert('Please enter a URL');
            return;
        }

        // Show loading indicator
        loadingIndicator.style.display = 'block';
        resultDiv.innerHTML = ''; // Clear previous results

        try {
            // Send POST request to backend URL
            const response = await fetch(`${window.BACKEND_URL}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: url }),
            });

            // Check if response is successful
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();

            // Handle backend errors
            if (data.error) {
                resultDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
            } else {
                displayResults(data); // Display successful results
            }
        } catch (error) {
            console.error('Error:', error);
            resultDiv.innerHTML = `<p style="color: red;">An error occurred while analyzing the URL</p>`;
        } finally {
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
        }
    });

    // Function to display analysis results
    function displayResults(data) {
        let html = `
            <h2>Analysis Results</h2>
            <p><strong>URL:</strong> ${data.url}</p>
            <p><strong>Processing Time:</strong> ${data.processing_time.toFixed(2)} seconds</p>
            <div class="misinformation-header">Detected Misinformation</div>
            <div class="misinformation-list">
        `;

        // Loop through misinformation results
        data.result.forEach(item => {
            html += `
                <div class="misinformation-item">
                    <div class="statement-label">Statement:</div>
                    <div class="statement-text">${item.statement}</div>
                    <div class="category-label">Category: ${item.category}</div>
                    <div class="confidence-label">Confidence: ${(item.confidence * 100).toFixed(1)}%</div>
                    <div class="counter-args-label">Counter Arguments:</div>
                    <ul class="counter-args-list">
                        ${item.counter_arguments.map(arg => `<li>${arg}</li>`).join('')}
                    </ul>
            `;
    
            if (item.sources && item.sources.length > 0) {
                html += `
                    <div class="sources-label">Sources:</div>
                    <ul class="sources-list">
                        ${item.sources.map(source => `<li>${source}</li>`).join('')}
                    </ul>
                `;
            }

            html += `</div>`;
        });
    
        html += `</div>
            <div class="report-container">
                <div class="report-toggle">
                    <button type="button" class="primary-btn" onclick="toggleReport()">Toggle Markdown Report</button>
                </div>
                <pre id="report" class="report" style="display: none;">${data.report || 'No report available.'}</pre>
            </div>
        `;
    
        resultDiv.innerHTML = html;
    }

    // Function to toggle report visibility
    function toggleReport() {
        const report = document.getElementById('report');
        if (report) {
            report.style.display = report.style.display === 'none' ? 'block' : 'none';
        }
    }
});