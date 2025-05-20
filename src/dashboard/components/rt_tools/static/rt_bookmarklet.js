// RT score extraction bookmarklet
(function(){
    // Extract data from RT page
    var title = document.querySelector('h1')?.textContent?.trim();
    var scores = Array.from(document.querySelectorAll('.critics-score')).map(e => e.textContent);
    var seasons = Array.from(document.querySelectorAll('.seasonHeader')).map(e => e.textContent);
    var episodes = Array.from(document.querySelectorAll('.episodeHeader')).map(e => e.textContent);
    var labels = [...seasons, ...episodes];
    
    // Show overlay
    var d = document.createElement('div');
    d.style.cssText = 'position:fixed;top:0;left:0;background:white;padding:20px;z-index:9999;border:2px solid black';
    d.innerHTML = '<h3>Found Scores for ' + title + ':</h3>' + 
        scores.map((score, i) => (labels[i] || '') + ' ' + score).join('<br>');
    document.body.appendChild(d);
    
    // Send data via postMessage
    window.opener.postMessage({
        type: 'rt_scores',
        data: {
            title: title,
            scores: scores,
            labels: labels
        }
    }, '*');
    
    // Auto-close after 2 seconds
    setTimeout(() => window.close(), 2000);
})();
