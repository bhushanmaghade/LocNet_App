document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const loading = document.getElementById('loading');
    const resultsArea = document.getElementById('results-area');
    let map = null;

    // --- Drag & Drop ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
    });

    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    fileInput.addEventListener('change', function () {
        handleFiles(this.files);
    });

    // --- Main Logic ---
    function handleFiles(files) {
        if (files.length > 0) {
            uploadFile(files[0]);
        }
    }

    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        showLoading(true, "Uploading & Processing...");
        resultsArea.classList.add('hidden');

        try {
            // 1. Upload
            const upRes = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const upData = await upRes.json();

            if (!upRes.ok) throw new Error(upData.error || "Upload failed");

            // 2. Predict
            showLoading(true, "Optimizing Coordinates...");
            const predRes = await fetch(`/predict/${upData.upload_id}`);
            const predData = await predRes.json();

            if (!predRes.ok) throw new Error(predData.error || "Prediction failed");

            displayResults(predData);
            showToast("Optimization Complete!", "success");

        } catch (error) {
            showToast(error.message, "error");
        } finally {
            showLoading(false);
        }
    }

    function displayResults(data) {
        // Update Text Metrics
        updateMetric('res-median', data.median);
        updateMetric('res-kalman', data.kalman);
        updateMetric('res-lstm', data.lstm);
        updateMetric('res-cnn', data.cnn);

        resultsArea.classList.remove('hidden');
        resultsArea.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Init Map if needed
        if (!map) {
            map = L.map('map').setView([0, 0], 2);
            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; OpenStreetMap &copy; CARTO'
            }).addTo(map);
        }

        // Clear existing layers
        map.eachLayer((layer) => {
            if (layer instanceof L.Marker || layer instanceof L.Polyline) {
                map.removeLayer(layer);
            }
        });

        // Collect points for bounds
        const points = [];
        const colors = {
            'median': '#10b981', // Emerald
            'kalman': '#3b82f6', // Blue
            'lstm': '#a855f7',   // Purple
            'cnn': '#ef4444'     // Red
        };

        // Plot Markers
        for (const [key, val] of Object.entries(data)) {
            if (val && val.lat && val.lon) {
                L.circleMarker([val.lat, val.lon], {
                    color: colors[key] || 'gray',
                    radius: 8,
                    fillOpacity: 0.9,
                    weight: 2
                }).addTo(map).bindPopup(`<div style="color:black"><b>${key.toUpperCase()}</b><br>Lat: ${val.lat.toFixed(6)}<br>Lon: ${val.lon.toFixed(6)}</div>`);
                points.push([val.lat, val.lon]);
            }
        }

        // Fit Bounds
        if (points.length > 0) {
            const group = new L.featureGroup(points.map(p => L.marker(p)));
            map.fitBounds(group.getBounds().pad(0.2));
            setTimeout(() => map.invalidateSize(), 100); // Fix map resize glitch
        }
    }

    function updateMetric(id, val) {
        const el = document.getElementById(id);
        if (val) {
            el.innerHTML = `<span style="display:block; color:var(--text-secondary); font-size:0.75rem;">LATITUDE</span>${val.lat.toFixed(5)}<br><span style="display:block; color:var(--text-secondary); font-size:0.75rem; margin-top:4px;">LONGITUDE</span>${val.lon.toFixed(5)}`;
        } else {
            el.textContent = "N/A";
        }
    }

    function showLoading(show, text) {
        loading.className = show ? "" : "hidden";
        if (text) document.getElementById('loading-text').textContent = text;
    }

    function showToast(message, type = "info") {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `<span>${type === 'success' ? '✅' : '⚠️'}</span> ${message}`;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = "slideIn 0.3s reverse";
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
});
