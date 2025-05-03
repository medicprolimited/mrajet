document.addEventListener('DOMContentLoaded', () => {
    // Select DOM elements
    const form = document.querySelector('form');
    const urlInput = document.getElementById('url-input');
    const resultDiv = document.getElementById('result');
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
            // Send POST request to ngrok URL
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
            <h3>Detected Misinformation</h3>
            <ul>
        `;

        // Loop through misinformation results
        data.result.forEach(item => {
            html += `
                <li>
                    <strong>Statement:</strong> ${item.statement}<br>
                    <strong>Category:</strong> ${item.category}<br>
                    <strong>Confidence:</strong> ${(item.confidence * 100).toFixed(1)}%<br>
                    <strong>Counter Arguments:</strong>
                    <ul>
                        ${item.counter_arguments.map(arg => `<li>${arg}</li>`).join('')}
                    </ul>
                </li>
            `;
        });

        html += `</ul><h3>Report</h3><pre>${data.report}</pre>`;
        resultDiv.innerHTML = html;
    }
});