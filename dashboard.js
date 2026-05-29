// Global State
let allSteps = [];
let filteredSteps = [];
let currentIdx = 0;
let currentFilter = 'all';
let searchQuery = '';
let isPlaying = false;
let playInterval = null;

// Initialize Dashboard
window.addEventListener('DOMContentLoaded', () => {
  const progressEl = document.getElementById('preloader-progress');
  const statusEl = document.getElementById('preloader-status');
  const preloader = document.getElementById('preloader');

  function updateLoadProgress(pct, text) {
    if (progressEl) progressEl.style.width = `${pct}%`;
    if (statusEl) statusEl.textContent = `${text} (${pct}%)`;
  }

  // Simulation of V8 memory heap initialization to show visual progress
  setTimeout(() => {
    updateLoadProgress(25, "Parsing token sequences & embeddings...");
    
    setTimeout(() => {
      updateLoadProgress(55, "Scanning 12 SSM layer projection variables...");
      
      setTimeout(() => {
        updateLoadProgress(80, "Indexing convolutional recurrence states...");
        
        setTimeout(() => {
          if (typeof H3_INDEX_DATA !== 'undefined' && Array.isArray(H3_INDEX_DATA)) {
            allSteps = H3_INDEX_DATA;
          } else {
            showEmptyState("No H3 mathematical trace datasets detected. Please run the generation script first.");
            if (preloader) preloader.style.display = 'none';
            return;
          }

          filteredSteps = [...allSteps];
          updateLoadProgress(100, "Trace environment ready!");
          
          setTimeout(() => {
            if (preloader) {
              preloader.style.opacity = '0';
              setTimeout(() => {
                preloader.style.display = 'none';
              }, 400);
            }
            updateStepBadges();
            renderIndex();
            loadStep(0);
          }, 300);
        }, 300);
      }, 300);
    }, 300);
  }, 150);
});

// Helper: Empty State UI
function showEmptyState(msg) {
  document.getElementById('viewer-area').innerHTML = `
    <div class="empty-state">
      <svg fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"></path>
      </svg>
      <h3>No Data Loaded</h3>
      <p>${msg}</p>
    </div>
  `;
}

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// Render a provenance badge + description block for tensor cards
function renderProvenance(provenance) {
  if (!provenance) return '';

  // Detect source type from the provenance prefix
  let sourceType = 'UNKNOWN';
  let badgeColor = 'rgba(120,120,120,0.25)';
  let badgeBorder = 'rgba(120,120,120,0.4)';
  let badgeText = '#aaa';
  let icon = '🔍';

  if (provenance.startsWith('MODEL')) {
    sourceType = 'MODEL'; badgeColor = 'rgba(139,92,246,0.18)'; badgeBorder = 'rgba(139,92,246,0.45)';
    badgeText = '#c4b5fd'; icon = '🏛️';
  } else if (provenance.startsWith('COMPUTED')) {
    sourceType = 'COMPUTED'; badgeColor = 'rgba(6,182,212,0.15)'; badgeBorder = 'rgba(6,182,212,0.4)';
    badgeText = '#67e8f9'; icon = '⚙️';
  } else if (provenance.startsWith('PREVIOUS_STEP')) {
    sourceType = 'PREVIOUS STEP'; badgeColor = 'rgba(245,158,11,0.15)'; badgeBorder = 'rgba(245,158,11,0.4)';
    badgeText = '#fcd34d'; icon = '🔗';
  } else if (provenance.startsWith('INPUT')) {
    sourceType = 'INPUT'; badgeColor = 'rgba(16,185,129,0.15)'; badgeBorder = 'rgba(16,185,129,0.4)';
    badgeText = '#6ee7b7'; icon = '📥';
  }

  // Bold keywords in the description for readability
  let desc = escapeHtml(provenance)
    .replace(/(Formula:)/g, '<strong style="color:var(--secondary)">$1</strong>')
    .replace(/(Inputs:)/g, '<strong style="color:var(--secondary)">$1</strong>')
    .replace(/(MODEL|COMPUTED|PREVIOUS_STEP|INPUT|PREVIOUS STEP)/g, `<strong style="color:${badgeText}">$1</strong>`);

  return `
    <div class="provenance-block" style="
      margin: 12px 0 16px 0;
      padding: 12px 16px;
      background: ${badgeColor};
      border: 1px solid ${badgeBorder};
      border-radius: 8px;
      font-size: 0.82rem;
      line-height: 1.65;
      color: var(--text-muted);
    ">
      <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
        <span style="font-size:1rem;">${icon}</span>
        <span style="
          font-size: 0.7rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          color: ${badgeText};
          background: ${badgeBorder};
          padding: 2px 9px;
          border-radius: 4px;
          text-transform: uppercase;
        ">${sourceType}</span>
        <span style="font-size:0.78rem; color:var(--text-muted); opacity:0.7;">Where does this value come from?</span>
      </div>
      <div style="color: rgba(255,255,255,0.75);">${desc}</div>
    </div>
  `;
}

function renderMarkdownText(text) {
  if (!text) return '';
  let html = escapeHtml(text);
  html = html.replace(/^###\s*(.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^##\s*(.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^\s*-\s+(.+)$/gm, '<li>$1</li>');
  if (html.includes('<li>')) {
    html = html.replace(/(?:<br>\s*)?<li>/g, '<li>');
    html = `<ul>${html}</ul>`;
  }
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\n{2,}/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');
  html = `<p>${html}</p>`;
  html = html.replace(/<p>\s*<ul>/g, '<ul>');
  html = html.replace(/<\/ul>\s*<\/p>/g, '</ul>');
  return html;
}

// Step filter options
function filterSteps(category) {
  currentFilter = category;
  document.querySelectorAll('.filter-pills .pill').forEach(p => p.classList.remove('active'));
  document.getElementById(`pill-${category}`).classList.add('active');
  
  applyFilters();
}

// Search input handler
function handleSearch() {
  searchQuery = document.getElementById('search-input').value.toLowerCase();
  applyFilters();
}

// Combine filters & search queries
function applyFilters() {
  filteredSteps = allSteps.filter(step => {
    // Category check
    if (currentFilter !== 'all') {
      if (currentFilter === 'section' && step.type !== 'section') return false;
      if (currentFilter === 'equation' && step.type !== 'equation') return false;
      if (currentFilter === 'tensor' && step.type !== 'tensor') return false;
      if (currentFilter === 'text' && step.type !== 'text') return false;
    }
    
    // Search query check
    if (searchQuery !== '') {
      let matchesSearch = false;
      if (step.title && step.title.toLowerCase().includes(searchQuery)) matchesSearch = true;
      if (step.description && step.description.toLowerCase().includes(searchQuery)) matchesSearch = true;
      if (step.content && step.content.toLowerCase().includes(searchQuery)) matchesSearch = true;
      if (step.name && step.name.toLowerCase().includes(searchQuery)) matchesSearch = true;
      if (step.chunk_url && step.chunk_url.toLowerCase().includes(searchQuery)) matchesSearch = true;
      if (step.values && Array.isArray(step.values)) {
        // Fast loop search instead of expensive JSON serialization block
        for (let i = 0; i < Math.min(step.values.length, 100); i++) {
          if (String(step.values[i]).toLowerCase().includes(searchQuery)) {
            matchesSearch = true;
            break;
          }
        }
      }
      return matchesSearch;
    }
    
    return true;
  });

  renderIndex();
  
  // Load first item of filtered list or reset
  if (filteredSteps.length > 0) {
    loadStep(0);
  } else {
    showEmptyState("No calculations matched the active search filters.");
  }
}

// Update count labels
function updateStepBadges() {
  document.getElementById('step-count-badge').textContent = `${allSteps.length} Total Steps`;
}

// Sidebar Index Rendering
function renderIndex() {
  const listContainer = document.getElementById('index-list');
  listContainer.innerHTML = '';
  
  filteredSteps.forEach((step, idx) => {
    const item = document.createElement('div');
    item.className = `index-item ${idx === currentIdx ? 'active' : ''}`;
    item.id = `index-item-${idx}`;
    item.onclick = () => jumpToStep(idx);
    
    // Step title logic
    let title = 'Step';
    let badgeClass = 'badge-text';
    
    if (step.type === 'section') {
      title = step.title;
      badgeClass = 'badge-section';
    } else if (step.type === 'subsection') {
      title = step.title;
      badgeClass = 'badge-subsection';
    } else if (step.type === 'equation') {
      title = step.description || 'Equation';
      badgeClass = 'badge-equation';
    } else if (step.type === 'tensor') {
      title = step.name || 'Tensor Parameter';
      badgeClass = 'badge-tensor';
    } else if (step.type === 'text') {
      title = step.content.substring(0, 30) + '...';
      badgeClass = 'badge-text';
    }
    
    item.innerHTML = `
      <span class="step-badge ${badgeClass}">${step.type}</span>
      <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1;">${title}</span>
    `;
    
    listContainer.appendChild(item);
  });
}

// Load active step details in the main window
function loadStep(idx) {
  if (filteredSteps.length === 0) return;
  currentIdx = idx;
  activeTensorValues = null; // Reset current tensor cache for copying
  
  // Update sidebar active selection style
  document.querySelectorAll('.index-list .index-item').forEach(item => item.classList.remove('active'));
  const activeSidebarItem = document.getElementById(`index-item-${idx}`);
  if (activeSidebarItem) {
    activeSidebarItem.classList.add('active');
    activeSidebarItem.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  
  // Update Controls labels
  document.getElementById('current-step-label').textContent = `Step ${currentIdx + 1} of ${filteredSteps.length}`;
  const percentage = Math.round(((currentIdx + 1) / filteredSteps.length) * 100);
  document.getElementById('completion-percentage').textContent = `${percentage}% Complete`;
  document.getElementById('progress-bar').style.width = `${percentage}%`;
  
  // Disable nav buttons if boundaries reached
  document.getElementById('btn-prev').disabled = currentIdx === 0;
  document.getElementById('btn-next').disabled = currentIdx === filteredSteps.length - 1;
  
  // Render details card
  const step = filteredSteps[currentIdx];
  const viewer = document.getElementById('viewer-area');
  
  let stepTypeLabel = step.type.toUpperCase();
  let stepTitle = "";
  let cardBody = "";
  
  if (step.type === 'section') {
    stepTitle = "Execution Section Marker";
    cardBody = `
      <div class="section-view">
        <h2>${step.title}</h2>
        <p>H3 block execution has transitioned to this operation sequence phase.</p>
      </div>
    `;
  } else if (step.type === 'subsection') {
    stepTitle = "Sub-execution Phase";
    cardBody = `
      <div class="section-view">
        <h3 style="font-size: 1.6rem; color: var(--secondary);">${step.title}</h3>
        <p>Intermediate parameters and sub-equations evaluated below.</p>
      </div>
    `;
  } else if (step.type === 'equation') {
    stepTitle = step.description;
    cardBody = `
      <div class="equation-view">
        <div class="math-block">
          <!-- Equation to render via KaTeX -->
          <div class="latex-render">$$ ${step.latex_eq} $$</div>
        </div>
        <div class="math-desc">
          Mathematical relationship resolving input variables. Verify exact shape matchings and element-wise computations with these properties.
        </div>
      </div>
    `;
  } else if (step.type === 'text') {
    stepTitle = "Process Information & Metrics";
    cardBody = `
      <div class="text-view">${renderMarkdownText(step.content)}</div>
    `;
  } else if (step.type === 'tensor') {
    stepTitle = step.name;
    
    if (step.chunk_url && !step.values) {
      // Large tensor values stored in dynamic JSON chunk
      cardBody = `
        <div class="tensor-view">
          ${renderProvenance(step.provenance)}
          <div class="tensor-stats">
            <div class="stat-card primary-stat">
              <span class="stat-label">Tensor Shape</span>
              <span class="stat-value">[${step.shape.join(', ')}]</span>
            </div>
            <div class="stat-card">
              <span class="stat-label">Mean Activation</span>
              <span class="stat-value">${typeof step.mean === 'number' ? step.mean.toFixed(6) : step.mean}</span>
            </div>
            <div class="stat-card">
              <span class="stat-label">Standard Deviation</span>
              <span class="stat-value">${typeof step.std === 'number' ? step.std.toFixed(6) : step.std}</span>
            </div>
          </div>
          
          <div class="tensor-values-container">
            <div class="tensor-values-header">
              <span id="tensor-load-status">Value Array Dump (Fetching Raw Chunk...)</span>
              <span style="font-family: monospace; font-size: 0.75rem;">Total Size: ${step.shape.reduce((a, b) => a * b, 1)} float elements</span>
            </div>
            <div class="tensor-grid" id="tensor-grid-values" style="display: flex; align-items: center; justify-content: center; min-height: 120px; color: var(--text-muted);">
              <div class="spinner-container" style="display: flex; flex-direction: column; align-items: center; gap: 0.75rem;">
                <div class="spinner" style="width: 28px; height: 28px; border: 2.5px solid rgba(255, 255, 255, 0.08); border-top-color: var(--secondary); border-radius: 50%; animation: spin 0.8s linear infinite;"></div>
                <span style="font-size: 0.85rem;">Retrieving un-omitted values...</span>
              </div>
            </div>
          </div>
        </div>
      `;
      
      // Trigger asynchronous background load
      fetch(step.chunk_url)
        .then(res => res.json())
        .then(values => {
          // Cache values on the step object so we don't have to fetch them again next time
          step.values = values;
          
          if (currentIdx === idx) {
            activeTensorValues = values; // Cache for easy copying
          }
          if (currentIdx !== idx) return; // Prevent render if user navigated away
          
          let gridHtml = '';
          const maxDisplay = 2000;
          const displayValues = values.slice(0, maxDisplay);
          displayValues.forEach(val => {
            const numVal = parseFloat(val);
            const signClass = numVal >= 0 ? 'positive' : 'negative';
            gridHtml += `<span class="tensor-val-item ${signClass}">${val}</span>`;
          });
          
          if (values.length > maxDisplay) {
            gridHtml += `<span class="tensor-val-item-truncated" style="grid-column: 1 / -1; text-align: center; color: var(--text-muted); padding: 12px; font-style: italic; background: rgba(255, 255, 255, 0.02); border-radius: 4px; border: 1px dashed rgba(255, 255, 255, 0.05); margin-top: 8px;">... and ${values.length - maxDisplay} more values. Use the Copy button to copy all elements.</span>`;
          }
          
          const gridEl = document.getElementById('tensor-grid-values');
          const statusEl = document.getElementById('tensor-load-status');
          if (gridEl) {
            gridEl.style.display = 'grid';
            gridEl.innerHTML = gridHtml;
          }
          if (statusEl) {
            statusEl.textContent = 'Value Array Dump';
          }
        })
        .catch(err => {
          console.error(err);
          const gridEl = document.getElementById('tensor-grid-values');
          if (gridEl) {
            gridEl.innerHTML = `<span style="color: var(--accent); font-size: 0.875rem;">Failed to retrieve values: ${err.message}</span>`;
          }
        });
    } else {
      // Small tensors embedded directly in the index file or already fetched chunk tensors
      activeTensorValues = step.values; // Cache for easy copying
      let gridHtml = '';
      if (Array.isArray(step.values)) {
        const maxDisplay = 2000;
        const displayValues = step.values.slice(0, maxDisplay);
        displayValues.forEach(val => {
          const numVal = parseFloat(val);
          const signClass = numVal >= 0 ? 'positive' : 'negative';
          gridHtml += `<span class="tensor-val-item ${signClass}">${val}</span>`;
        });
        
        if (step.values.length > maxDisplay) {
          gridHtml += `<span class="tensor-val-item-truncated" style="grid-column: 1 / -1; text-align: center; color: var(--text-muted); padding: 12px; font-style: italic; background: rgba(255, 255, 255, 0.02); border-radius: 4px; border: 1px dashed rgba(255, 255, 255, 0.05); margin-top: 8px;">... and ${step.values.length - maxDisplay} more values. Use the Copy button to copy all elements.</span>`;
        }
      } else {
        gridHtml = step.values || 'N/A';
      }
      
      cardBody = `
        <div class="tensor-view">
          ${renderProvenance(step.provenance)}
          <div class="tensor-stats">
            <div class="stat-card primary-stat">
              <span class="stat-label">Tensor Shape</span>
              <span class="stat-value">[${step.shape.join(', ')}]</span>
            </div>
            <div class="stat-card">
              <span class="stat-label">Mean Activation</span>
              <span class="stat-value">${typeof step.mean === 'number' ? step.mean.toFixed(6) : step.mean}</span>
            </div>
            <div class="stat-card">
              <span class="stat-label">Standard Deviation</span>
              <span class="stat-value">${typeof step.std === 'number' ? step.std.toFixed(6) : step.std}</span>
            </div>
          </div>
          
          <div class="tensor-values-container">
            <div class="tensor-values-header">
              <span>Value Array Dump</span>
              <span style="font-family: monospace; font-size: 0.75rem;">Total Size: ${step.shape.reduce((a, b) => a * b, 1)} float elements</span>
            </div>
            <div class="tensor-grid">
              ${gridHtml}
            </div>
          </div>
        </div>
      `;
    }
  }
  
  viewer.innerHTML = `
    <div class="card">
      <div class="card-header-step" style="display: flex; align-items: center; justify-content: space-between; gap: 16px; width: 100%;">
        <div style="display: flex; align-items: center; gap: 12px; min-width: 0; flex: 1;">
          <span class="card-title-step" style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${stepTitle}</span>
          <span class="step-badge badge-${step.type}">${stepTypeLabel}</span>
        </div>
        <button class="btn-copy" onclick="copyActiveSlide(this)" title="Copy slide contents to clipboard" style="flex-shrink: 0;">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 5px;"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
          <span>Copy values</span>
        </button>
      </div>
      ${cardBody}
    </div>
  `;
  
  // Render LaTeX dynamically inside elements using KaTeX
  if (typeof renderMathInElement === 'function') {
    renderMathInElement(viewer, {
      delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false}
      ],
      throwOnError: false
    });
  }
}

let activeTensorValues = null;

function copyActiveSlide(buttonEl) {
  const step = filteredSteps[currentIdx];
  let copyText = "";
  
  if (step.type === 'tensor') {
    if (activeTensorValues) {
      copyText = activeTensorValues.join(', ');
    } else if (step.values && Array.isArray(step.values)) {
      copyText = step.values.join(', ');
    } else {
      alert("Values are still loading. Please try again in a moment.");
      return;
    }
  } else if (step.type === 'equation') {
    copyText = step.latex_eq || "";
  } else if (step.type === 'text') {
    copyText = step.content || "";
  } else if (step.type === 'section' || step.type === 'subsection') {
    copyText = step.title || "";
  }

  navigator.clipboard.writeText(copyText).then(() => {
    const span = buttonEl.querySelector('span');
    const originalText = span.textContent;
    span.textContent = "Copied!";
    buttonEl.classList.add('copied');
    
    setTimeout(() => {
      span.textContent = originalText;
      buttonEl.classList.remove('copied');
    }, 1500);
  }).catch(err => {
    console.error('Failed to copy content: ', err);
  });
}

// Navigation Step
function navigateStep(direction) {
  const nextIdx = currentIdx + direction;
  if (nextIdx >= 0 && nextIdx < filteredSteps.length) {
    loadStep(nextIdx);
  }
}

// Jump Directly to Step
function jumpToStep(idx) {
  if (idx >= 0 && idx < filteredSteps.length) {
    loadStep(idx);
  }
}

// Slideshow player options
function togglePlay() {
  isPlaying = !isPlaying;
  
  const playIcon = document.getElementById('play-icon');
  const pauseIcon = document.getElementById('pause-icon');
  const playBtn = document.getElementById('btn-play');
  
  if (isPlaying) {
    playIcon.style.display = 'none';
    pauseIcon.style.display = 'block';
    playBtn.style.borderColor = 'var(--secondary)';
    
    playInterval = setInterval(() => {
      if (currentIdx < filteredSteps.length - 1) {
        navigateStep(1);
      } else {
        togglePlay(); // Pause if end reached
      }
    }, 3000);
  } else {
    playIcon.style.display = 'block';
    pauseIcon.style.display = 'none';
    playBtn.style.borderColor = 'var(--border-color)';
    
    clearInterval(playInterval);
  }
}
