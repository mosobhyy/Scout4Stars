function updateProgress() {
    // Make an AJAX request to the server to get the progress information
    fetch('/progress/status')
        .then(response => response.json())
        .then(data => {
            // Update the progress information in the HTML
            const progressPlaceholder = document.getElementById('progress-placeholder');
            progressPlaceholder.innerHTML = '';

            for (const filename in data) {
                const progress = data[filename];
                const progressText = document.createElement('p');
                progressText.textContent = `${filename}: ${progress}%`;

                progressPlaceholder.appendChild(progressText);
            }
        });
}

// Periodically update the progress every 2 seconds
setInterval(updateProgress, 2000);
