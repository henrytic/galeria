const state = {
    currentTab: 'upload'
};

document.addEventListener('DOMContentLoaded', () => {
    setupDragAndDrop();
});

function switchTab(tabId) {
    state.currentTab = tabId;

    // UI Updates
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });

    document.querySelectorAll('.tab-content').forEach(section => {
        section.classList.toggle('active', section.id === `${tabId}-section`);
    });

    // Content Load
    if (tabId === 'gallery') {
        loadGallery();
    } else if (tabId === 'people') {
        loadPeople();
    } else if (tabId === 'metrics') {
        loadMetrics();
    }
}

function loadMetrics() {
    fetch('/api/metrics')
        .then(res => res.json())
        .then(data => {
            document.getElementById('m-total-faces').textContent = data.total_faces;
            document.getElementById('m-total-people').textContent = data.total_people;
            document.getElementById('m-noise').textContent = data.noise_faces;
            document.getElementById('m-silhouette').textContent = data.silhouette_score;
            if (document.getElementById('m-db')) document.getElementById('m-db').textContent = data.davies_bouldin;
            if (document.getElementById('m-ch')) document.getElementById('m-ch').textContent = data.calinski_harabasz;
            document.getElementById('m-confidence').textContent = data.avg_confidence + '%';
        });

    fetch('/api/scatter')
        .then(res => res.json())
        .then(data => {
            if (!data || data.length === 0) return;

            // Group data by cluster for Plotly traces
            const clusters = {};
            data.forEach(pt => {
                const cid = pt.cluster;
                if (!clusters[cid]) clusters[cid] = { x: [], y: [], text: [], name: pt.label, marker: { size: 10 } };
                clusters[cid].x.push(pt.x);
                clusters[cid].y.push(pt.y);
                clusters[cid].text.push(pt.label);
            });

            const traces = [];
            for (const [cid, pts] of Object.entries(clusters)) {
                traces.push({
                    x: pts.x,
                    y: pts.y,
                    mode: 'markers',
                    type: 'scatter',
                    name: pts.name,
                    text: pts.text,
                    marker: { size: 12 }
                });
            }

            const layout = {
                title: 'Distribución de Rostros (PCA)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#f8fafc' },
                xaxis: { showgrid: false, zeroline: false, showticklabels: false },
                yaxis: { showgrid: false, zeroline: false, showticklabels: false },
                hovermode: 'closest'
            };

            Plotly.newPlot('cluster-plot', traces, layout);
        });
}

// Upload Handling
function setupDragAndDrop() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

    dropZone.addEventListener('click', () => fileInput.click()); // Allow click to open dialog if not clicking button

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
}

function handleFiles(files) {
    if (files.length === 0) return;

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files[]', files[i]);
    }

    const uploadArea = document.querySelector('.upload-area');
    const statusArea = document.getElementById('processing-status');

    uploadArea.style.display = 'none';
    statusArea.style.display = 'block';

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data);
            alert(`Procesado exitosamente! SE encontraron ${data.people.length} personas.`);
            uploadArea.style.display = 'flex';
            statusArea.style.display = 'none';
            switchTab('people');
        })
        .catch((error) => {
            console.error('Error:', error);
            alert('Ocurrió un error al subir las imágenes.');
            uploadArea.style.display = 'flex';
            statusArea.style.display = 'none';
        });
}

// Gallery Handling
function loadGallery() {
    fetch('/api/gallery')
        .then(res => res.json())
        .then(data => {
            const grid = document.getElementById('gallery-grid');
            document.getElementById('gallery-count').textContent = data.images.length;
            grid.innerHTML = '';
            data.images.forEach(src => {
                const div = document.createElement('div');
                div.className = 'gallery-item';
                div.innerHTML = `<img src="${src}" loading="lazy" alt="Image">`;
                grid.appendChild(div);
            });
        });
}

// People Handling
function loadPeople() {
    fetch('/api/people')
        .then(res => res.json())
        .then(data => {
            const grid = document.getElementById('people-grid');
            document.getElementById('people-count').textContent = data.length;
            grid.innerHTML = '';

            if (data.length === 0) {
                grid.innerHTML = '<p>No se han encontrado personas aún. Sube algunas fotos.</p>';
                return;
            }

            data.forEach(person => {
                const div = document.createElement('div');
                div.className = 'person-card';
                // Pass both ID and Name to the modal opener
                div.onclick = () => openPersonModal(person.id, person.name);
                div.innerHTML = `
                <img src="${person.face_url}" class="person-face" alt="${person.name}">
                <div class="person-info">
                    <h3>${person.name}</h3>
                    <p>${person.images.length} Fotos</p>
                </div>
            `;
                grid.appendChild(div);
            });
        });
}

// Modal Handling
let currentPersonId = null;

function openPersonModal(id, name) {
    currentPersonId = id;
    const modal = document.getElementById('person-modal');

    // Update modal header with name
    const display_name = name || `Persona ${id}`;
    document.getElementById('modal-person-name').textContent = display_name;

    // Reset/Set input field
    const input = document.getElementById('rename-input');
    if (input) {
        input.value = display_name.startsWith('Persona ') ? '' : display_name;
    }

    modal.style.display = 'block';

    fetch(`/api/person/${id}`)
        .then(res => res.json())
        .then(data => {
            const grid = document.getElementById('person-gallery-grid');
            grid.innerHTML = '';
            data.images.forEach(src => {
                const div = document.createElement('div');
                div.className = 'gallery-item';
                div.innerHTML = `<img src="${src}" loading="lazy" alt="Image">`;
                grid.appendChild(div);
            });
        });
}

function saveName() {
    const nameInput = document.getElementById('rename-input');
    const name = nameInput.value;

    if (!name || !currentPersonId) return;

    fetch(`/api/person/${currentPersonId}/rename`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name })
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                document.getElementById('modal-person-name').textContent = name;
                loadPeople(); // Refresh list to update name there too
                loadMetrics(); // Refresh chart labels
                alert('Nombre actualizado correctamente');
            } else {
                alert('Error al guardar el nombre');
            }
        });
}

function closeModal() {
    document.getElementById('person-modal').style.display = 'none';
}

// Close modal if clicking outside
window.onclick = function (event) {
    const modal = document.getElementById('person-modal');
    if (event.target == modal) {
        modal.style.display = "none";
    }
}
