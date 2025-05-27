javascript:(function(){
    // Extract scores from RT page
    try {
        const data = {
            title: document.querySelector('h1').innerText,
            tomatometer: parseInt(document.querySelector('[data-qa="tomatometer"]').innerText),
            audience: parseInt(document.querySelector('[data-qa="audience-score"]').innerText)
        };

        // Validate scores exist
        if (!data.tomatometer && !data.audience) {
            alert('Could not find any scores on page');
            return;
        }

        // Send to local proxy
        fetch('http://localhost:3000/submit-scores', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            if (result.status === 'success') {
                alert('Scores saved successfully!');
            } else {
                throw new Error(result.message || 'Unknown error');
            }
        })
        .catch(error => {
            alert(`Error saving scores: ${error.message}\nMake sure the proxy server is running.`);
        });
    } catch (error) {
        alert(`Error extracting scores: ${error.message}`);
    }
})();
