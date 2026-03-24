document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const errorMessage = document.getElementById('error-message');
    const budgetInput = document.getElementById('budget-input');
    const latInput = document.getElementById('lat-input');
    const lonInput = document.getElementById('lon-input');
    const locateBtn = document.getElementById('locate-btn');

    // UI elements to update
    const totalDetections = document.getElementById('total-detections');
    const totalMatches = document.getElementById('total-matches');

    // All 6 garment categories mapped to their DOM elements
    const categories = {
        shirts:  { container: document.getElementById('shirts-container'),  grid: document.getElementById('shirts-grid') },
        pants:   { container: document.getElementById('pants-container'),   grid: document.getElementById('pants-grid') },
        shoes:   { container: document.getElementById('shoes-container'),   grid: document.getElementById('shoes-grid') },
        jackets: { container: document.getElementById('jackets-container'), grid: document.getElementById('jackets-grid') },
        dresses: { container: document.getElementById('dresses-container'), grid: document.getElementById('dresses-grid') },
        skirts:  { container: document.getElementById('skirts-container'),  grid: document.getElementById('skirts-grid') },
    };

    const cardTemplate = document.getElementById('product-card-template');

    // --- Geolocation ---
    if (locateBtn) {
        locateBtn.addEventListener('click', () => {
            if (!navigator.geolocation) {
                alert('Geolocation is not supported by your browser.');
                return;
            }
            locateBtn.textContent = 'Locating…';
            locateBtn.disabled = true;
            navigator.geolocation.getCurrentPosition(
                (pos) => {
                    latInput.value = pos.coords.latitude.toFixed(6);
                    lonInput.value = pos.coords.longitude.toFixed(6);
                    locateBtn.textContent = '✓ Location set';
                    locateBtn.disabled = false;
                },
                () => {
                    alert('Unable to retrieve your location.');
                    locateBtn.textContent = 'Use My Location';
                    locateBtn.disabled = false;
                }
            );
        });
    }

    // --- Drag and Drop Handling ---
    dropZone.addEventListener('click', () => fileInput.click());

    ['dragover', 'dragenter'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'dragend', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            processImage(file);
        } else {
            showError('Please upload a valid image file (JPG, PNG, WEBP).');
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            processImage(e.target.files[0]);
        }
    });

    // --- Core Logic ---
    async function processImage(file) {
        // Reset UI
        hideError();
        results.classList.add('hidden');
        dropZone.classList.add('hidden');
        loading.classList.remove('hidden');

        // Clear previous grids
        Object.values(categories).forEach(cat => {
            cat.grid.innerHTML = '';
            cat.container.classList.add('hidden');
        });

        // Build multipart form data
        const formData = new FormData();
        formData.append('file', file);

        const budget = budgetInput && budgetInput.value ? parseFloat(budgetInput.value) : null;
        if (budget && budget > 0) {
            formData.append('budget', budget);
        }

        const lat = latInput && latInput.value ? parseFloat(latInput.value) : null;
        const lon = lonInput && lonInput.value ? parseFloat(lonInput.value) : null;
        if (lat !== null && lon !== null) {
            formData.append('user_latitude', lat);
            formData.append('user_longitude', lon);
        }

        try {
            const response = await fetch('/api/v1/pipeline/recommend', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                if (response.status === 429) {
                    throw new Error('Too many requests. Please try again in a minute.');
                }
                throw new Error(`Server returned ${response.status}`);
            }

            const data = await response.json();
            renderResults(data);

        } catch (error) {
            console.error(error);
            showError(error.message || 'An error occurred while connecting to the AI service.');
            loading.classList.add('hidden');
            dropZone.classList.remove('hidden');
        }
    }

    function renderResults(data) {
        totalDetections.textContent = data.total_detections;
        totalMatches.textContent = data.total_matches;

        renderCategoryGrid(data.shirts,  categories.shirts);
        renderCategoryGrid(data.pants,   categories.pants);
        renderCategoryGrid(data.shoes,   categories.shoes);
        renderCategoryGrid(data.jackets, categories.jackets);
        renderCategoryGrid(data.dresses, categories.dresses);
        renderCategoryGrid(data.skirts,  categories.skirts);

        loading.classList.add('hidden');
        dropZone.classList.remove('hidden');
        results.classList.remove('hidden');
        results.scrollIntoView({ behavior: 'smooth' });
    }

    function renderCategoryGrid(categoryResults, uiElements) {
        if (!categoryResults || categoryResults.length === 0) return;

        let hasMatches = false;
        categoryResults.forEach(detected => {
            if (detected.matches && detected.matches.length > 0) {
                hasMatches = true;
                detected.matches.forEach(match => {
                    uiElements.grid.appendChild(createCard(match));
                });
            }
        });

        if (hasMatches) {
            uiElements.container.classList.remove('hidden');
        }
    }

    function createCard(match) {
        const clone = cardTemplate.content.cloneNode(true);
        const a = clone.querySelector('a');
        const img = clone.querySelector('img');
        const badge = clone.querySelector('.match-badge');
        const brand = clone.querySelector('.brand-name');
        const name = clone.querySelector('.product-name');
        const price = clone.querySelector('.price');

        const meta = match.metadata || {};

        a.href = meta.product_url || '#';
        img.src = meta.image_url || 'https://via.placeholder.com/300x400?text=No+Image';

        const confidence = Math.round(match.score * 100);
        badge.textContent = `${confidence}% Match`;

        brand.textContent = meta.brand || 'Unknown Brand';
        name.textContent = meta.name || 'Unnamed Product';

        const currency = meta.currency || '$';
        const priceVal = meta.price != null ? Number(meta.price).toFixed(2) : '0.00';
        price.textContent = `${currency}${priceVal}`;

        return clone;
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorMessage.classList.remove('hidden');
    }

    function hideError() {
        errorMessage.classList.add('hidden');
    }
});
