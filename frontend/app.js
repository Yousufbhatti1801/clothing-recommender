document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const errorMessage = document.getElementById('error-message');

    // UI elements to update
    const totalDetections = document.getElementById('total-detections');
    const totalMatches = document.getElementById('total-matches');

    // Categories UI
    const categories = {
        'shirt': { container: document.getElementById('shirts-container'), grid: document.getElementById('shirts-grid') },
        'pants': { container: document.getElementById('pants-container'), grid: document.getElementById('pants-grid') },
        'shoes': { container: document.getElementById('shoes-container'), grid: document.getElementById('shoes-grid') }
    };

    const cardTemplate = document.getElementById('product-card-template');

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
            showError("Please upload a valid image file (JPG, PNG, WEBP).");
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

        // Prepare request
        const formData = new FormData();
        formData.append('file', file);
        
        // We can pass optional form parameters if needed (e.g., budget, limit)
        // formData.append('limit_per_category', 4);

        try {
            // Note: Since the static files and API are served from the same origin,
            // we can use relative paths.
            const response = await fetch('/api/v1/pipeline/recommend', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                if (response.status === 429) throw new Error("Too many requests. Please try again in a minute.");
                throw new Error(`Server returned ${response.status}`);
            }

            const data = await response.json();
            renderResults(data);

        } catch (error) {
            console.error(error);
            showError(error.message || "An error occurred while connecting to the AI service.");
            
            // Bring drop zone back on failure
            loading.classList.add('hidden');
            dropZone.classList.remove('hidden');
        }
    }

    function renderResults(data) {
        // Set stats
        totalDetections.textContent = data.total_detections;
        totalMatches.textContent = data.total_matches;

        // Render grids
        renderCategoryGrid(data.shirts, categories.shirt);
        renderCategoryGrid(data.pants, categories.pants);
        renderCategoryGrid(data.shoes, categories.shoes);

        // Swap UI
        loading.classList.add('hidden');
        dropZone.classList.remove('hidden'); // allow uploading another
        results.classList.remove('hidden');
        results.scrollIntoView({ behavior: 'smooth' });
    }

    function renderCategoryGrid(categoryResults, uiElements) {
        if (!categoryResults || categoryResults.length === 0) return;
        
        // We only take the first set of matches for the category
        // (If there are multiple people/shirts detected, this MVP just dumps all matches flat)
        let hasMatches = false;

        categoryResults.forEach(detected => {
            if (detected.matches && detected.matches.length > 0) {
                hasMatches = true;
                detected.matches.forEach(match => {
                    const card = createCard(match);
                    uiElements.grid.appendChild(card);
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

        // Extract metadata safely
        const meta = match.metadata || {};

        a.href = meta.product_url || '#';
        img.src = meta.image_url || 'https://via.placeholder.com/300x400?text=No+Image';
        
        // Convert score to percentage
        const confidence = Math.round(match.score * 100);
        badge.textContent = `${confidence}% Match`;
        
        brand.textContent = meta.brand || 'Unknown Brand';
        name.textContent = meta.name || 'Unnamed Product';
        
        // Format price
        const currency = meta.currency || '$';
        price.textContent = `${currency}${meta.price || '0.00'}`;

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
