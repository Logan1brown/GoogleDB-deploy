// RT score extraction bookmarklet
(function(){
    // Extract data from RT page
    var title = document.querySelector('h1')?.textContent?.trim();
    
    // Get main show scores
    var tomatometer = document.querySelector('[data-qa="tomatometer"]')?.textContent?.trim();
    var audience = document.querySelector('[data-qa="audience-score"]')?.textContent?.trim();
    
    // Convert to numbers
    tomatometer = tomatometer ? parseInt(tomatometer) : null;
    audience = audience ? parseInt(audience) : null;
    
    if (!title || (!tomatometer && !audience)) {
        alert('Could not find show scores. Make sure you are on a show\'s main page.');
        return;
    }
    
    // Show overlay
    var d = document.createElement('div');
    d.style.cssText = 'position:fixed;top:0;left:0;background:white;padding:20px;z-index:9999;border:2px solid black';
    d.innerHTML = `
        <div style="font-family:sans-serif">
            <h3>${title}</h3>
            <p>Tomatometer: ${tomatometer}%</p>
            <p>Audience: ${audience}%</p>
            <button onclick="this.parentElement.parentElement.remove()">Close</button>
        </div>
    `;
    document.body.appendChild(d);
    
    // Send data via postMessage
    window.opener.postMessage({
        type: 'rt_scores',
        data: {
            title: title,
            tomatometer: tomatometer,
            audience: audience
        }
    }, '*');
})();
