document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById('fileInput');
    const responseMessage = document.getElementById('responseMessage');
    const chartsContainer = document.getElementById('chartsContainer');
    const beforeChart = document.getElementById('beforeChart');
    const afterChart = document.getElementById('afterChart');

    // Reset visual feedback
    responseMessage.innerText = '';
    responseMessage.style.color = '#e74c3c'; // Default error color
    chartsContainer.style.display = 'none';
    beforeChart.style.display = 'none';
    afterChart.style.display = 'none';

    // Check if a file is selected
    if (fileInput.files.length === 0) {
        responseMessage.innerText = 'Error: No file selected.';
        responseMessage.classList.add('fade-in'); // Simple fade-in animation for better feedback
        return;
    }

    formData.append('file', fileInput.files[0]);

    // Display loading state
    responseMessage.innerText = 'Uploading...';
    responseMessage.style.color = '#007bff'; // Blue to indicate progress

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok: ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        // Handle chart display
        if (data.beforeChart && data.afterChart) {
            beforeChart.src = `/static/${data.beforeChart}`;
            afterChart.src = `/static/${data.afterChart}`;
            beforeChart.style.display = 'block';
            afterChart.style.display = 'block';
            chartsContainer.style.display = 'block';

            beforeChart.classList.add('fade-in');
            afterChart.classList.add('fade-in');

        } else {
            responseMessage.innerText = 'Error: Charts not available.';
        }

        // Handle the processed file download link
        if (data.filename) {
            const link = document.createElement('a');
            link.href = `/processed/${data.filename}`;
            link.innerText = 'Download processed file';
            link.target = '_blank';
            responseMessage.innerHTML = ''; // Clear previous messages
            responseMessage.appendChild(link);
            responseMessage.style.color = '#28a745'; // Success color
        } else {
            responseMessage.innerText = 'Error: Processed file not available.';
        }

    })
    .catch(error => {
        responseMessage.innerText = 'Error: ' + error.message;
        responseMessage.classList.add('fade-in');
    });
});
