/**
 * GMP Regulatory Intelligence - Frontend JavaScript
 * Handles query submission, results display, and interactions
 */

// DOM Elements
const queryInput = document.getElementById('queryInput');
const topKSelect = document.getElementById('topK');
const submitBtn = document.getElementById('submitBtn');
const querySection = document.getElementById('querySection');
const resultsSection = document.getElementById('resultsSection');
const loadingState = document.getElementById('loadingState');
const resultsContent = document.getElementById('resultsContent');
const errorState = document.getElementById('errorState');
const newQueryBtn = document.getElementById('newQueryBtn');
const retryBtn = document.getElementById('retryBtn');

// Results elements
const queryDisplay = document.getElementById('queryDisplay');
const answerText = document.getElementById('answerText');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill = document.getElementById('confidenceFill');
const queryType = document.getElementById('queryType');
const sourcesCount = document.getElementById('sourcesCount');
const sourcesGrid = document.getElementById('sourcesGrid');
const errorMessage = document.getElementById('errorMessage');

// State
let currentQuery = '';
let isLoading = false;

/**
 * Initialize application
 */
function init() {
    // Event listeners
    submitBtn.addEventListener('click', handleSubmit);
    queryInput.addEventListener('keydown', handleKeyPress);
    newQueryBtn.addEventListener('click', handleNewQuery);
    retryBtn.addEventListener('click', handleRetry);
    
    // Example queries
    document.querySelectorAll('.example-query').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const query = e.target.dataset.query;
            queryInput.value = query;
            queryInput.focus();
        });
    });
    
    // Auto-resize textarea
    queryInput.addEventListener('input', autoResizeTextarea);
    
    console.log('🚀 GMP Regulatory Intelligence initialized');
}

/**
 * Handle form submission
 */
async function handleSubmit(e) {
    if (e) e.preventDefault();
    
    const query = queryInput.value.trim();
    if (!query || isLoading) return;
    
    currentQuery = query;
    const topK = parseInt(topKSelect.value);
    
    // Show loading state
    setLoadingState(true);
    showResultsSection();
    
    try {
        // Call API
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query, top_k: topK })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data.answer);
        } else {
            showError(data.error || 'An error occurred');
        }
        
    } catch (error) {
        console.error('Query error:', error);
        showError('Failed to connect to server. Please check if the application is running.');
    } finally {
        setLoadingState(false);
    }
}

/**
 * Handle Enter key in textarea (Shift+Enter for new line)
 */
function handleKeyPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
    }
}

/**
 * Auto-resize textarea based on content
 */
function autoResizeTextarea() {
    queryInput.style.height = 'auto';
    queryInput.style.height = queryInput.scrollHeight + 'px';
}

/**
 * Set loading state
 */
function setLoadingState(loading) {
    isLoading = loading;
    
    if (loading) {
        submitBtn.classList.add('loading');
        submitBtn.disabled = true;
        queryInput.disabled = true;
    } else {
        submitBtn.classList.remove('loading');
        submitBtn.disabled = false;
        queryInput.disabled = false;
    }
}

/**
 * Show results section
 */
function showResultsSection() {
    querySection.classList.add('hidden');
    resultsSection.classList.remove('hidden');
    loadingState.classList.remove('hidden');
    resultsContent.classList.add('hidden');
    errorState.classList.add('hidden');
}

/**
 * Display query results
 */
function displayResults(answer) {
    // Query display
    queryDisplay.textContent = `"${currentQuery}"`;
    
    // Answer text
    answerText.textContent = answer.text;
    
    // Confidence
    const confidence = Math.round(answer.confidence_score * 100);
    confidenceValue.textContent = `${confidence}%`;
    confidenceFill.style.width = `${confidence}%`;
    
    // Set confidence color
    if (confidence >= 80) {
        confidenceFill.style.background = 'var(--accent-success)';
    } else if (confidence >= 50) {
        confidenceFill.style.background = 'var(--accent-primary)';
    } else {
        confidenceFill.style.background = 'var(--accent-warm)';
    }
    
    // Query metadata (Cypher query used)
    if (answer.cypher_query) {
        queryType.textContent = `Graph Query Executed`;
    } else {
        queryType.textContent = `Hybrid Retrieval`;
    }
    
    // Graph path
    displayGraphPath(answer.graph_path, answer.path_description);
    
    // Cypher query
    displayCypherQuery(answer.cypher_query);
    
    // Sources
    sourcesCount.textContent = answer.sources.length;
    displaySources(answer.sources);
    
    // Show results
    loadingState.classList.add('hidden');
    resultsContent.classList.remove('hidden');
    
    // Scroll to results
    setTimeout(() => {
        resultsContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

/**
 * Display graph traversal path
 */
function displayGraphPath(graphPath, description) {
    const pathDescElement = document.getElementById('pathDescription');
    const graphPathElement = document.getElementById('graphPath');
    
    pathDescElement.textContent = description || 'Graph traversal completed';
    graphPathElement.innerHTML = '';
    
    if (!graphPath || graphPath.length === 0) {
        graphPathElement.innerHTML = '<p class="no-data">No graph path available</p>';
        return;
    }
    
    graphPath.forEach((node, index) => {
        const nodeElement = document.createElement('div');
        nodeElement.className = 'graph-node';
        nodeElement.style.animationDelay = `${index * 0.05}s`;
        
        nodeElement.innerHTML = `
            <div class="node-header">
                <span class="node-number">${index + 1}</span>
                <span class="node-type">${escapeHtml(node.node_type)}</span>
            </div>
            <div class="node-text">${escapeHtml(node.text || node.node_id || 'N/A')}</div>
        `;
        
        graphPathElement.appendChild(nodeElement);
    });
}

/**
 * Display Cypher query
 */
function displayCypherQuery(query) {
    const cypherElement = document.getElementById('cypherQuery');
    
    if (!query || query.trim() === '') {
        cypherElement.textContent = 'No Cypher query executed (used vector/keyword retrieval)';
        cypherElement.classList.add('no-query');
    } else {
        cypherElement.textContent = query;
        cypherElement.classList.remove('no-query');
    }
}

/**
 * Display source documents
 */
function displaySources(sources) {
    sourcesGrid.innerHTML = '';
    
    sources.forEach((source, index) => {
        const card = document.createElement('div');
        card.className = 'source-card';
        card.style.animationDelay = `${index * 0.05}s`;
        
        card.innerHTML = `
            <div class="source-header">
                <div class="source-doc">${escapeHtml(source.source_doc)}</div>
                <div class="source-id">${escapeHtml(source.requirement_id)}</div>
            </div>
            <div class="source-text">${escapeHtml(source.text)}</div>
            <div class="source-category">${escapeHtml(source.category)}</div>
        `;
        
        sourcesGrid.appendChild(card);
    });
}

/**
 * Show error state
 */
function showError(error) {
    errorMessage.textContent = error;
    loadingState.classList.add('hidden');
    resultsContent.classList.add('hidden');
    errorState.classList.remove('hidden');
}

/**
 * Handle new query button
 */
function handleNewQuery() {
    querySection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    queryInput.value = '';
    queryInput.focus();
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * Handle retry button
 */
function handleRetry() {
    handleSubmit();
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Format confidence percentage
 */
function formatConfidence(value) {
    return Math.round(value * 100);
}

// Initialize on DOM load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
