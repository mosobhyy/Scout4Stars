function checkProgress() {
    // Make an AJAX request to the server to get the progress information
    fetch('/progress_status')
        .then(response => response.json())
        .then(data => {
            
            let allValuesAre100 = true;
            // Loop over the keys and values of the dictionary
            for (const i in data) {
                if (data[i] !== 100) {
                    allValuesAre100 = false;
                    break;
                }
            }
            if (allValuesAre100) {
                setTimeout(() => {
                    const buttonToEnable = document.getElementById('next-button');
                    buttonToEnable.disabled = false;
                }, 3000); // sleep for 5 seconds before enabling the button
            }
        });
}

setInterval(checkProgress, 2000); // Call checkProgress every 2 seconds