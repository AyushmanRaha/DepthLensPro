const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const fileQueue = document.getElementById('fileQueue');
const processBtn = document.getElementById('processBtn');
const stopBtn = document.getElementById('stopBtn');
const refreshBtn = document.getElementById('refreshBtn');
const modelSelect = document.getElementById('modelSelect');
const loadingState = document.getElementById('loadingState');
const progressBar = document.getElementById('progressBar');
const processingText = document.getElementById('processingText');
const progressTimer = document.getElementById('progressTimer');
const resultsGallery = document.getElementById('resultsGallery');

let selectedFiles = []; 
const MAX_FILES = 5;
let progressInterval = null;
let currentController = null; 

// Chart.js setup
const ctx = document.getElementById('performanceChart').getContext('2d');
let performanceChart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Inference Time (s)', data: [], borderColor: '#6366f1', backgroundColor: 'rgba(99, 102, 241, 0.1)', fill: true, tension: 0.3, pointBackgroundColor: '#6366f1' }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, grid: { color: '#2d3748' } }, x: { grid: { display: false } } } }
});

// Drag & Drop handlers
['dragenter', 'dragover'].forEach(eventName => { dropZone.addEventListener(eventName, e => { e.preventDefault(); dropZone.classList.add('highlight'); }); });
['dragleave', 'drop'].forEach(eventName => { dropZone.addEventListener(eventName, e => { e.preventDefault(); dropZone.classList.remove('highlight'); }); });
dropZone.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
fileInput.addEventListener('change', e => handleFiles(e.target.files));

function handleFiles(files) {
    const validFiles = Array.from(files).filter(f => ['image/jpeg', 'image/png'].includes(f.type));
    if (selectedFiles.length + validFiles.length > MAX_FILES) return alert(`Max ${MAX_FILES} images allowed.`);
    
    validFiles.forEach(file => {
        if(file.size > 10 * 1024 * 1024) return alert(`Skipping ${file.name} (> 10MB).`);
        selectedFiles.push({ file: file, url: URL.createObjectURL(file) });
    });
    updateFileQueueVisuals();
}

function updateFileQueueVisuals() {
    fileQueue.innerHTML = '';
    if (selectedFiles.length === 0) {
        processBtn.disabled = true;
        processBtn.textContent = 'Generate';
        return;
    }
    processBtn.disabled = false;
    processBtn.textContent = `Generate (${selectedFiles.length})`;
    selectedFiles.forEach((fileObj, index) => {
        const item = document.createElement('div');
        item.className = 'queue-item';
        item.innerHTML = `<span>${fileObj.file.name}</span><button class="btn-remove" onclick="removeFileFromQueue(${index})">&times;</button>`;
        fileQueue.appendChild(item);
    });
}

function removeFileFromQueue(index) {
    URL.revokeObjectURL(selectedFiles[index].url);
    selectedFiles.splice(index, 1);
    updateFileQueueVisuals();
}

// STOP Functionality
stopBtn.addEventListener('click', () => {
    if (currentController) {
        currentController.abort(); 
        setProcessingState(false);
        processingText.textContent = "Process aborted by user.";
        loadingState.classList.remove('hidden'); 
        setTimeout(() => loadingState.classList.add('hidden'), 3000);
    }
});

// REFRESH Functionality
refreshBtn.addEventListener('click', () => {
    if (currentController) currentController.abort();
    setProcessingState(false);
    selectedFiles.forEach(obj => URL.revokeObjectURL(obj.url));
    selectedFiles = [];
    updateFileQueueVisuals();
    resultsGallery.innerHTML = '';
    
    // Reset Chart and Metrics
    performanceChart.data.labels = [];
    performanceChart.data.datasets[0].data = [];
    performanceChart.update();
    document.getElementById('metricTime').textContent = '--';
    document.getElementById('metricRes').textContent = '--';
});

// GENERATE Functionality
processBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) return;

    currentController = new AbortController();
    const formData = new FormData();
    selectedFiles.forEach(obj => formData.append('files', obj.file));
    formData.append('model_name', modelSelect.value);

    setProcessingState(true, selectedFiles.length);

    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            body: formData,
            signal: currentController.signal 
        });

        if (!response.ok) throw new Error(response.statusText);
        const data = await response.json();
        
        setProcessingState(false);
        renderResultsGallery(data, selectedFiles);
        updateDashboard(data);

        selectedFiles.forEach(obj => URL.revokeObjectURL(obj.url));
        selectedFiles = [];
        updateFileQueueVisuals();

    } catch (error) {
        setProcessingState(false);
        if (error.name === 'AbortError') {
            console.log("Fetch aborted successfully.");
        } else {
            alert("ERROR: Cannot connect to backend. Is FastAPI running?");
            console.error(error);
        }
    }
});

function setProcessingState(isProcessing, fileCount = 0) {
    if (isProcessing) {
        loadingState.classList.remove('hidden');
        resultsGallery.innerHTML = '';
        processBtn.classList.add('hidden');
        stopBtn.classList.remove('hidden');
        
        processingText.textContent = `Analyzing ${fileCount} image${fileCount > 1 ? 's' : ''}...`;
        
        let expectedTimeMs = 15000; 
        if (fileCount > 2) expectedTimeMs += 10000;
        if (modelSelect.value === 'DPT_Large') expectedTimeMs *= 1.5; 

        let remainingTime = expectedTimeMs;
        let progress = 0;
        const tickRate = 100; 
        
        progressBar.style.width = '0%';
        progressTimer.textContent = `Estimated time: ${Math.ceil(remainingTime/1000)}s`;

        clearInterval(progressInterval);
        progressInterval = setInterval(() => {
            remainingTime -= tickRate;
            progress = ((expectedTimeMs - remainingTime) / expectedTimeMs) * 100;
            
            if(remainingTime <= 0) { remainingTime = 0; progress = 99; }
            
            progressBar.style.width = `${Math.min(progress, 99)}%`;
            progressTimer.textContent = `Estimated time: ${Math.ceil(remainingTime/1000)}s`;
        }, tickRate);

    } else {
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        
        setTimeout(() => {
            loadingState.classList.add('hidden');
            progressBar.style.width = '0%';
            stopBtn.classList.add('hidden');
            processBtn.classList.remove('hidden');
            currentController = null;
        }, 300); 
    }
}

// Optimized Layout Renderer for Details
function renderResultsGallery(apiData, localFileObjects) {
    resultsGallery.innerHTML = ''; 
    apiData.results.forEach((result, index) => {
        const localObj = localFileObjects[index];
        const card = document.createElement('div');
        card.className = 'result-card';
        
        card.innerHTML = `
            <div class="image-comparison">
                <img src="${localObj.url}" alt="Original">
                <img src="data:image/png;base64,${result.depth_map}" alt="Depth Map">
            </div>
            <div class="card-content">
                <h4>${result.filename}</h4>
                <div class="card-details">
                    <span><strong>Model:</strong> ${apiData.model_used}</span>
                    <span><strong>Res:</strong> ${result.resolution}</span>
                    <span><strong>Time:</strong> ${result.inference_time}s</span>
                </div>
                <a href="data:image/png;base64,${result.depth_map}" download="depth_${result.filename}" class="btn-download">↓ Download Depth Map</a>
            </div>
        `;
        resultsGallery.appendChild(card);
    });
}

function updateDashboard(apiData) {
    const count = apiData.results.length;
    const totalTime = apiData.results.reduce((sum, res) => sum + res.inference_time, 0).toFixed(3);
    document.getElementById('metricTime').textContent = `${totalTime}s`;
    document.getElementById('metricRes').textContent = `${count}`;

    const runLabel = `${apiData.model_used} (${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'})})`;
    performanceChart.data.labels.push(runLabel);
    performanceChart.data.datasets[0].data.push(totalTime);
    
    if(performanceChart.data.labels.length > 5) {
        performanceChart.data.labels.shift();
        performanceChart.data.datasets[0].data.shift();
    }
    performanceChart.update();
}