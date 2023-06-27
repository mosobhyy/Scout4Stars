function updateProgress() {
    // Make an AJAX request to the server to get the progress information
    fetch('/progress_status')
        .then(response => response.json())
        .then(data => {

            // Update multiple elements on the client-side based on the data returned by the server
            const progressBarElements = document.getElementsByClassName('progress_role');
            
            // Loop over the keys and values of the dictionary
            for (const i in data) {
                const progressPlaceholder = progressBarElements[i];
                progressPlaceholder.style.setProperty('--value', data[i]);
            }
        });
}

// Periodically update the progress every 2 seconds
setInterval(updateProgress, 2000);