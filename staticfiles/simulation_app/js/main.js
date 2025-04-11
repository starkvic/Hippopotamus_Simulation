// Example: polling simulation status from a (future) API endpoint
function fetchSimulationStatus() {
    fetch('/api/simulation-status/')
        .then(response => response.json())
        .then(data => {
            document.getElementById('simulation-status').innerText =
                "Status: " + data.status + " (" + data.progress + "%)";
        })
        .catch(error => console.error('Error fetching simulation status:', error));
}

// For now, simulate periodic updates
setInterval(fetchSimulationStatus, 5000);
