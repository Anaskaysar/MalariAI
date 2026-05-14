// MalariAI Phase 4 Web App Logic

document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const runBtn = document.getElementById('runBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');
    
    // Result elements
    const smearImg = document.getElementById('smearImg');
    const cellGrid = document.getElementById('cellGrid');
    const cellCountNote = document.getElementById('cellCountNote');
    
    // Stats
    const statDetected = document.getElementById('statDetected');
    const statInfected = document.getElementById('statInfected');
    const statHealthy = document.getElementById('statHealthy');
    const statRate = document.getElementById('statRate');
    const statLeuko = document.getElementById('statLeuko');
    
    // Report
    const statusBanner = document.getElementById('statusBanner');
    const statusText = document.getElementById('statusText');
    const statusSub = document.getElementById('statusSub');
    const dominantStage = document.getElementById('dominantStage');
    const ringSummary = document.getElementById('ringSummary');
    const schizontSummary = document.getElementById('schizontSummary');
    
    // Bars
    const bars = {
        'red blood cell': document.getElementById('barRbc'),
        'trophozoite': document.getElementById('barTroph'),
        'ring': document.getElementById('barRing'),
        'schizont': document.getElementById('barSchiz'),
        'gametocyte': document.getElementById('barGam'),
        'leukocyte': document.getElementById('barLeuko')
    };
    const barCounts = {
        'red blood cell': document.getElementById('countRbc'),
        'trophozoite': document.getElementById('countTroph'),
        'ring': document.getElementById('countRing'),
        'schizont': document.getElementById('countSchiz'),
        'gametocyte': document.getElementById('countGam'),
        'leukocyte': document.getElementById('countLeuko')
    };

    // Grad-CAM
    const gcImg = document.getElementById('gcImg');
    const gcClass = document.getElementById('gcClass');
    const gcNote = document.getElementById('gcNote');

    // UI State
    let analysisData = null;

    // Upload handling
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = '#1d4ed8';
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.style.borderColor = '#bfdbfe';
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = '#bfdbfe';
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect();
        }
    });

    function handleFileSelect() {
        if (fileInput.files.length) {
            const file = fileInput.files[0];
            const reader = new FileReader();
            reader.onload = (e) => {
                smearImg.src = e.target.result;
                runBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    }

    runBtn.addEventListener('click', async () => {
        if (!fileInput.files.length) return;
        
        const formData = new FormData();
        formData.append('image', fileInput.files[0]);
        
        loadingOverlay.classList.remove('hidden');
        runBtn.disabled = true;
        
        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) throw new Error('Analysis failed');
            
            const data = await response.json();
            analysisData = data;
            renderResults(data);
            
        } catch (err) {
            alert('Error: ' + err.message);
        } finally {
            loadingOverlay.classList.add('hidden');
            runBtn.disabled = false;
        }
    });

    function renderResults(data) {
        // 1. Smear Image
        smearImg.src = `data:image/jpeg;base64,${data.smear_image}`;
        
        // 2. Stats
        statDetected.textContent = data.metrics.total_cells;
        statInfected.textContent = data.metrics.infected_cells;
        statHealthy.textContent = data.metrics.healthy_rbcs;
        statRate.textContent = data.metrics.infection_rate + '%';
        statLeuko.textContent = data.metrics.leukocytes;
        
        // 3. Status
        if (data.metrics.infected_cells > 0) {
            statusBanner.className = 'badge badge-danger';
            if (statusText) statusText.textContent = 'Malaria Detected';
            if (statusSub) {
                statusSub.textContent = `· ${data.metrics.infection_rate}% parasitaemia`;
                statusSub.classList.remove('hidden');
            }
        } else {
            statusBanner.className = 'badge badge-success';
            if (statusText) statusText.textContent = 'No Malaria Detected';
            if (statusSub) {
                statusSub.textContent = '· Screening negative';
                statusSub.classList.remove('hidden');
            }
        }
        
        if (dominantStage) dominantStage.textContent = data.metrics.dominant_stage;
        
        // Per-class summary
        const ringCount = data.class_counts['ring'] || 0;
        if (ringSummary) ringSummary.textContent = ringCount > 0 ? `${ringCount} cells detected` : 'None detected';
        
        const schizCount = data.class_counts['schizont'] || 0;
        if (schizontSummary) schizontSummary.textContent = schizCount > 0 ? `${schizCount} cell detected` : 'None detected';

        // 4. Bar Chart
        Object.keys(bars).forEach(cls => {
            const count = data.class_counts[cls] || 0;
            const pct = data.metrics.total_cells > 0 ? (count / data.metrics.total_cells * 100) : 0;
            if (bars[cls]) bars[cls].style.width = pct + '%';
            if (barCounts[cls]) barCounts[cls].textContent = count;
        });

        // 5. Gallery
        cellGrid.innerHTML = '';
        data.cells.forEach((cell, i) => {
            const thumb = document.createElement('div');
            const classKey = cell.label.split(' ')[0].toLowerCase();
            thumb.className = `cell-thumb t-${classKey}`;
            if (i === 0) thumb.classList.add('selected');

            thumb.innerHTML = `
                <img src="data:image/jpeg;base64,${cell.crop}" class="cell-img">
                <div class="cell-tag2">${getShort(cell.label)}</div>
                <div class="cell-conf2">${Math.round(cell.confidence * 100)}%</div>
            `;
            
            thumb.onclick = () => {
                document.querySelectorAll('.cell-thumb').forEach(t => t.classList.remove('selected'));
                thumb.classList.add('selected');
                updateGradCam(cell);
            };
            
            cellGrid.appendChild(thumb);
        });
        
        cellCountNote.textContent = `Showing ${data.cells.length} of ${data.metrics.total_cells} cells`;

        // Default Grad-CAM to first cell
        if (data.cells.length > 0) {
            updateGradCam(data.cells[0]);
        }
    }

    function updateGradCam(cell) {
        if (cell.gradcam) {
            gcImg.src = `data:image/jpeg;base64,${cell.gradcam}`;
            gcNote.textContent = 'Model activated on parasite signature.';
        } else {
            gcImg.src = `data:image/jpeg;base64,${cell.crop}`;
            gcNote.textContent = 'No Grad-CAM available for this cell type.';
        }
        gcClass.textContent = `${cell.label} · ${Math.round(cell.confidence * 100)}%`;
    }

    function getShort(label) {
        const map = {
            'red blood cell': 'RBC',
            'trophozoite': 'Troph',
            'ring': 'Ring',
            'schizont': 'Schiz',
            'gametocyte': 'Gam',
            'leukocyte': 'WBC'
        };
        return map[label] || label;
    }
});
