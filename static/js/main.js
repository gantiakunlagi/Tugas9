// Global Variables
let scatterChart = null;
let currentClassificationData = null;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    // Setup drag and drop for upload zone
    setupDragAndDrop();
});

// 1. Switch between Slider and Upload Input Modes
function switchInputMode(mode) {
    document.getElementById("input-mode").value = mode;
    
    // Toggle active state for tab buttons
    const btnSlider = document.getElementById("tab-btn-slider");
    const btnUpload = document.getElementById("tab-btn-upload");
    
    const sectionSlider = document.getElementById("slider-section");
    const sectionUpload = document.getElementById("upload-section");
    
    if (mode === 'slider') {
        btnSlider.classList.add("active");
        btnUpload.classList.remove("active");
        sectionSlider.classList.add("active");
        sectionUpload.classList.remove("active");
    } else {
        btnSlider.classList.remove("active");
        btnUpload.classList.add("active");
        sectionSlider.classList.remove("active");
        sectionUpload.classList.add("active");
    }
}

// 2. Update displayed values for range inputs
function updateParamVal(id) {
    const input = document.getElementById(id);
    const display = document.getElementById(`${id}-val`);
    if (input && display) {
        display.textContent = input.value;
    }
}

// 3. Handle File Selector and Drag & Drop
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        showFilePreview(file);
    }
}

function showFilePreview(file) {
    if (!file.type.startsWith('image/')) {
        alert("Harap unggah file berupa citra gambar saja.");
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewContainer = document.getElementById("preview-container");
        const imgPreview = document.getElementById("img-preview");
        const dropZone = document.getElementById("drop-zone");
        
        imgPreview.src = e.target.result;
        previewContainer.style.display = "block";
        dropZone.style.display = "none";
    };
    reader.readAsDataURL(file);
}

function removeSelectedFile() {
    const fileInput = document.getElementById("file-input");
    const previewContainer = document.getElementById("preview-container");
    const dropZone = document.getElementById("drop-zone");
    const imgPreview = document.getElementById("img-preview");
    
    fileInput.value = "";
    imgPreview.src = "#";
    previewContainer.style.display = "none";
    dropZone.style.display = "block";
}

function setupDragAndDrop() {
    const dropZone = document.getElementById("drop-zone");
    if (!dropZone) return;
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });
    
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            const fileInput = document.getElementById("file-input");
            fileInput.files = files;
            showFilePreview(files[0]);
        }
    }, false);
}

// 4. Handle AJAX Classification Submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    const submitBtn = document.getElementById("submit-btn");
    const form = document.getElementById("classify-form");
    const welcomeCard = document.getElementById("welcome-card");
    
    const resultHero = document.getElementById("result-hero");
    const visCard = document.getElementById("visualization-card");
    const probCard = document.getElementById("prob-card");
    const neighborsCard = document.getElementById("neighbors-card");
    
    // Disable button & animate status
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Memproses Analisis...';
    
    const formData = new FormData(form);
    
    try {
        const response = await fetch("/api/classify", {
            method: "POST",
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || "Terjadi kegagalan komunikasi dengan server.");
        }
        
        // Save classification data globally
        currentClassificationData = data;
        
        // Hide welcome state
        if (welcomeCard) welcomeCard.style.display = "none";
        
        // Show results containers
        resultHero.style.display = "block";
        visCard.style.display = "block";
        probCard.style.display = "block";
        neighborsCard.style.display = "block";
        
        // 5. Update Prediction Hero Card Details
        const label = data.prediction.label;
        const name = data.prediction.name;
        const cause = data.prediction.cause;
        const symptoms = data.prediction.symptoms;
        const solutions = data.prediction.solutions;
        const severity = data.prediction.severity;
        
        // UI color mapping for the hero card
        let diseaseClass = 'healthy';
        if (label === 0) diseaseClass = 'blight';
        else if (label === 1) diseaseClass = 'brown';
        else if (label === 2) diseaseClass = 'smut';
        
        // Update elements
        const badgeText = document.getElementById("res-badge-text");
        const resBadge = document.getElementById("res-badge");
        const resTitle = document.getElementById("res-title");
        const resCause = document.getElementById("res-cause");
        const resSymptoms = document.getElementById("res-symptoms");
        const resSolutions = document.getElementById("res-solutions");
        
        badgeText.textContent = name;
        resBadge.className = `prediction-badge ${diseaseClass}`;
        
        // Change icon inside badge
        let iconClass = 'fa-shield-virus';
        if (label === 3) iconClass = 'fa-leaf';
        resBadge.querySelector('i').className = `fa-solid ${iconClass}`;
        
        resTitle.textContent = name;
        resCause.innerHTML = `<i class="fa-solid fa-circle-info"></i> Penyebab: <strong>${cause}</strong> <span style="margin-left: 1rem; color: var(--text-muted)">(${severity})</span>`;
        
        // Populate lists
        resSymptoms.innerHTML = symptoms.map(s => `<li>${s}</li>`).join('');
        resSolutions.innerHTML = solutions.map(s => `<li>${s}</li>`).join('');
        
        // Calculate Confidence based on class probability
        const pctConfidence = data.probabilities[label.toString()];
        document.getElementById("confidence-val").textContent = `${pctConfidence}%`;
        document.getElementById("confidence-fill").style.width = `${pctConfidence}%`;
        
        // 6. Update Probability Bars
        document.getElementById("prob-blight-pct").textContent = `${data.probabilities['0']}%`;
        document.getElementById("prob-blight-bar").style.width = `${data.probabilities['0']}%`;
        
        document.getElementById("prob-brown-pct").textContent = `${data.probabilities['1']}%`;
        document.getElementById("prob-brown-bar").style.width = `${data.probabilities['1']}%`;
        
        document.getElementById("prob-smut-pct").textContent = `${data.probabilities['2']}%`;
        document.getElementById("prob-smut-bar").style.width = `${data.probabilities['2']}%`;
        
        document.getElementById("prob-healthy-pct").textContent = `${data.probabilities['3']}%`;
        document.getElementById("prob-healthy-bar").style.width = `${data.probabilities['3']}%`;
        
        // 7. Update Neighbors Table
        const tableBody = document.getElementById("neighbors-table").querySelector("tbody");
        tableBody.innerHTML = "";
        
        data.neighbors.forEach((neigh, index) => {
            let nClass = 'healthy';
            if (neigh.label === 0) nClass = 'blight';
            else if (neigh.label === 1) nClass = 'brown';
            else if (neigh.label === 2) nClass = 'smut';
            
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>
                    <div style="display: flex; flex-direction: column;">
                        <strong>Tetangga #${index + 1}</strong>
                        <span class="distance-badge" style="width: fit-content; margin-top: 0.25rem;">d = ${neigh.distance}</span>
                    </div>
                </td>
                <td>
                    <div class="disease-dot-container">
                        <span class="dot ${nClass}"></span>
                        <strong>${neigh.disease_name}</strong>
                    </div>
                </td>
                <td>${neigh.features.Greenness.toFixed(4)}</td>
                <td>${neigh.features.BrownSpotDensity.toFixed(4)}</td>
                <td>${neigh.features.BlackSpotDensity.toFixed(4)}</td>
                <td>${neigh.features.LesionArea.toFixed(4)}</td>
            `;
            tableBody.appendChild(tr);
        });
        
        // 8. Render interactive scatter plot!
        renderKNNScatterPlot();
        
        // Scroll to results smoothly
        resultHero.scrollIntoView({ behavior: 'smooth' });
        
    } catch (error) {
        console.error("Classification error:", error);
        alert(`Gagal memproses data: ${error.message}`);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Jalankan Klasifikasi KNN';
    }
}

// 9. Generate and Redraw the Scatter Plot using Chart.js
function renderKNNScatterPlot() {
    if (!currentClassificationData) return;
    
    const xAxisFeat = document.getElementById("sumbu-x").value;
    const yAxisFeat = document.getElementById("sumbu-y").value;
    
    const ctx = document.getElementById("knnScatterPlot").getContext("2d");
    
    // Destroy previous chart if exists
    if (scatterChart !== null) {
        scatterChart.destroy();
    }
    
    // Group all training points by disease for scatter series
    const blightPoints = [];
    const brownPoints = [];
    const smutPoints = [];
    const healthyPoints = [];
    
    currentClassificationData.all_points.forEach(pt => {
        const pointObj = { x: pt[xAxisFeat], y: pt[yAxisFeat], id: pt.id };
        if (pt.Label === 0) blightPoints.push(pointObj);
        else if (pt.Label === 1) brownPoints.push(pointObj);
        else if (pt.Label === 2) smutPoints.push(pointObj);
        else if (pt.Label === 3) healthyPoints.push(pointObj);
    });
    
    // User query point
    const userQuery = {
        x: currentClassificationData.features[xAxisFeat],
        y: currentClassificationData.features[yAxisFeat]
    };
    
    // Prepare datasets
    const datasets = [
        {
            label: "Hawar Daun Bakteri",
            data: blightPoints,
            backgroundColor: "rgba(231, 76, 60, 0.55)",
            borderColor: "rgba(231, 76, 60, 0.8)",
            borderWidth: 1,
            pointRadius: 5.5,
            pointHoverRadius: 8
        },
        {
            label: "Bercak Cokelat",
            data: brownPoints,
            backgroundColor: "rgba(211, 84, 0, 0.55)",
            borderColor: "rgba(211, 84, 0, 0.8)",
            borderWidth: 1,
            pointRadius: 5.5,
            pointHoverRadius: 8
        },
        {
            label: "Smut Daun",
            data: smutPoints,
            backgroundColor: "rgba(241, 196, 15, 0.55)",
            borderColor: "rgba(241, 196, 15, 0.8)",
            borderWidth: 1,
            pointRadius: 5.5,
            pointHoverRadius: 8
        },
        {
            label: "Sehat",
            data: healthyPoints,
            backgroundColor: "rgba(46, 204, 113, 0.55)",
            borderColor: "rgba(46, 204, 113, 0.8)",
            borderWidth: 1,
            pointRadius: 5.5,
            pointHoverRadius: 8
        }
    ];
    
    // Draw connection lines to nearest neighbors
    currentClassificationData.neighbors.forEach((neigh, index) => {
        const neighborPoint = {
            x: neigh.features[xAxisFeat],
            y: neigh.features[yAxisFeat]
        };
        
        // Add a line dataset from query to neighbor
        datasets.push({
            label: `Koneksi Tetangga #${index + 1}`,
            data: [userQuery, neighborPoint],
            type: 'line',
            showLine: true,
            borderColor: "rgba(255, 255, 255, 0.35)",
            borderWidth: 1.5,
            borderDash: [5, 5],
            pointRadius: 0,
            pointHoverRadius: 0,
            fill: false,
            legend: {
                display: false
            }
        });
    });
    
    // User Query Point Dataset (Large and highlighted)
    datasets.push({
        label: "Data Input (Query)",
        data: [userQuery],
        backgroundColor: "#ffffff",
        borderColor: "#2ecc71",
        borderWidth: 3,
        pointRadius: 10,
        pointStyle: 'triangle',
        pointHoverRadius: 12,
        shadowBlur: 15,
        shadowColor: "#2ecc71"
    });
    
    // Config Chart
    scatterChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#9ca3af',
                        font: { family: 'Inter', size: 11 },
                        filter: function(item, chart) {
                            // Filter out line datasets from the legend
                            return !item.text.includes("Koneksi");
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.dataset.label;
                            if (label.includes("Koneksi")) return null;
                            return `${label}: (${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: getFeatureLabel(xAxisFeat),
                        color: '#f3f4f6',
                        font: { family: 'Outfit', size: 12, weight: 'bold' }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' },
                    min: 0,
                    max: 1.05
                },
                y: {
                    title: {
                        display: true,
                        text: getFeatureLabel(yAxisFeat),
                        color: '#f3f4f6',
                        font: { family: 'Outfit', size: 12, weight: 'bold' }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' },
                    min: 0,
                    max: 1.05
                }
            }
        }
    });
}

function redrawChart() {
    renderKNNScatterPlot();
}

function getFeatureLabel(feat) {
    const labels = {
        'Greenness': 'Greenness Index (Tingkat Kehijauan)',
        'BrownSpotDensity': 'Brown Spot Density (Bercak Cokelat)',
        'BlackSpotDensity': 'Black Spot Density (Bercak Hitam)',
        'LesionArea': 'Lesion Area Ratio (Luas Lesi)'
    };
    return labels[feat] || feat;
}
