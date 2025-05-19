// Main application logic for RAG Pipeline Explorer
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const presetSelect = document.getElementById('presetSelect');
    const useVectorCheckbox = document.getElementById('useVector');
    const useBM25Checkbox = document.getElementById('useBM25');
    const useRerankerCheckbox = document.getElementById('useReranker');
    const useLLMRerankerCheckbox = document.getElementById('useLLMReranker');
    const useQueryDecompositionCheckbox = document.getElementById('useQueryDecomposition');
    const topKSlider = document.getElementById('topK');
    const topKValue = document.getElementById('topKValue');
    const minSimilaritySlider = document.getElementById('minSimilarity');
    const minSimilarityValue = document.getElementById('minSimilarityValue');
    const queryInput = document.getElementById('queryInput');
    const submitQueryBtn = document.getElementById('submitQuery');
    const resultsContainer = document.getElementById('resultsContainer');
    const resultsContent = document.getElementById('resultsContent');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const historyList = document.getElementById('historyList');
    const clearHistoryBtn = document.getElementById('clearHistory');
    const leftSidebar = document.querySelector('.left-sidebar');
    const rightSidebar = document.querySelector('.right-sidebar');
    const leftSidebarToggle = document.querySelector('.left-toggle');
    const rightSidebarToggle = document.querySelector('.right-toggle');
    
    // Templates
    const historyItemTemplate = document.getElementById('historyItemTemplate');
    const documentTemplate = document.getElementById('documentTemplate');
    
    // State management
    let queryHistory = loadHistory();
    
    // Initialize UI
    initializeUI();
    renderHistory();
    loadSidebarState();
    
    // Event listeners
    presetSelect.addEventListener('change', handlePresetChange);
    topKSlider.addEventListener('input', updateTopKValue);
    minSimilaritySlider.addEventListener('input', updateMinSimilarityValue);
    submitQueryBtn.addEventListener('click', handleQuerySubmit);
    clearHistoryBtn.addEventListener('click', clearHistory);
    
    // Sidebar toggle event listeners
    leftSidebarToggle.addEventListener('click', toggleLeftSidebar);
    rightSidebarToggle.addEventListener('click', toggleRightSidebar);
    
    // Keyboard shortcuts for sidebar toggles
    document.addEventListener('keydown', (e) => {
        // Ctrl+B for left sidebar (history)
        if (e.ctrlKey && e.key === 'b') {
            e.preventDefault();
            toggleLeftSidebar();
            // Show a temporary tooltip to inform user
            showShortcutToast('History sidebar toggled (Ctrl+B)');
        }
        // Ctrl+Shift+B for right sidebar (configuration)
        if (e.ctrlKey && e.shiftKey && e.key === 'B') {
            e.preventDefault();
            toggleRightSidebar();
            // Show a temporary tooltip to inform user
            showShortcutToast('Configuration sidebar toggled (Ctrl+Shift+B)');
        }
    });
    
    // Helper function to show a temporary toast notification for keyboard shortcuts
    function showShortcutToast(message) {
        // Create toast element if it doesn't exist
        let toast = document.getElementById('shortcutToast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'shortcutToast';
            toast.className = 'shortcut-toast';
            document.body.appendChild(toast);
        }
        
        // Set message and show
        toast.textContent = message;
        toast.classList.add('show');
        
        // Hide after a delay
        setTimeout(() => {
            toast.classList.remove('show');
        }, 2000);
    }
    
    // Additional checkbox event listeners
    useBM25Checkbox.addEventListener('change', updateCustomConfig);
    useRerankerCheckbox.addEventListener('change', updateCustomConfig);
    useLLMRerankerCheckbox.addEventListener('change', updateCustomConfig);
    useQueryDecompositionCheckbox.addEventListener('change', updateCustomConfig);
    
    // Toggle left sidebar with smooth transition
    function toggleLeftSidebar() {
        leftSidebar.classList.toggle('collapsed');
        
        // Update icon based on current state (will be applied after toggle)
        const isCollapsed = leftSidebar.classList.contains('collapsed');
        
        // Update ARIA attributes for accessibility
        leftSidebarToggle.setAttribute('aria-expanded', !isCollapsed);
        
        // If we're expanding from a collapsed state, we need to fix the button position
        if (!isCollapsed) {
            // Small delay to let the CSS transition start
            setTimeout(() => {
                leftSidebarToggle.style.position = '';
                leftSidebarToggle.style.left = '';
                leftSidebarToggle.style.top = '';
            }, 50);
        }
        
        saveSidebarState();
    }
    
    // Toggle right sidebar with smooth transition
    function toggleRightSidebar() {
        rightSidebar.classList.toggle('collapsed');
        
        // Update icon based on current state (will be applied after toggle)
        const isCollapsed = rightSidebar.classList.contains('collapsed');
        
        // Update ARIA attributes for accessibility
        rightSidebarToggle.setAttribute('aria-expanded', !isCollapsed);
        
        // If we're expanding from a collapsed state, we need to fix the button position
        if (!isCollapsed) {
            // Small delay to let the CSS transition start
            setTimeout(() => {
                rightSidebarToggle.style.position = '';
                rightSidebarToggle.style.right = '';
                rightSidebarToggle.style.top = '';
            }, 50);
        }
        
        saveSidebarState();
    }
    
    // Save sidebar state to localStorage
    function saveSidebarState() {
        const state = {
            leftSidebarCollapsed: leftSidebar.classList.contains('collapsed'),
            rightSidebarCollapsed: rightSidebar.classList.contains('collapsed')
        };
        localStorage.setItem('sidebarState', JSON.stringify(state));
    }
    
    // Load sidebar state from localStorage
    function loadSidebarState() {
        try {
            const savedState = localStorage.getItem('sidebarState');
            if (savedState) {
                const state = JSON.parse(savedState);
                
                // Update left sidebar
                if (state.leftSidebarCollapsed) {
                    leftSidebar.classList.add('collapsed');
                    // Update ARIA attributes
                    leftSidebarToggle.setAttribute('aria-expanded', 'false');
                } else {
                    // Ensure ARIA attributes are correctly set
                    leftSidebarToggle.setAttribute('aria-expanded', 'true');
                }
                
                // Update right sidebar
                if (state.rightSidebarCollapsed) {
                    rightSidebar.classList.add('collapsed');
                    // Update ARIA attributes
                    rightSidebarToggle.setAttribute('aria-expanded', 'false');
                } else {
                    // Ensure ARIA attributes are correctly set
                    rightSidebarToggle.setAttribute('aria-expanded', 'true');
                }
            }
        } catch (error) {
            console.error('Error loading sidebar state:', error);
        }
    }
    
    // Initialize UI based on selected preset
    function initializeUI() {
        handlePresetChange();
        updateTopKValue();
        updateMinSimilarityValue();
        
        // Auto-collapse sidebars on mobile devices
        if (window.innerWidth <= 768) {
            if (!leftSidebar.classList.contains('collapsed')) {
                leftSidebar.classList.add('collapsed');
            }
            if (!rightSidebar.classList.contains('collapsed')) {
                rightSidebar.classList.add('collapsed');
            }
        }
    }
    
    // Handle window resize events
    window.addEventListener('resize', () => {
        // Auto-collapse on mobile if switching from desktop to mobile
        if (window.innerWidth <= 768) {
            if (!leftSidebar.classList.contains('collapsed')) {
                leftSidebar.classList.add('collapsed');
            }
            if (!rightSidebar.classList.contains('collapsed')) {
                rightSidebar.classList.add('collapsed');
            }
        }
    });
    
    // Handle preset selection change
    function handlePresetChange() {
        const preset = presetSelect.value;
        
        // Reset checkboxes
        useVectorCheckbox.checked = true; // Always true
        
        // Set checkbox states based on preset
        switch(preset) {
            case 'vector_only':
                useBM25Checkbox.checked = false;
                useRerankerCheckbox.checked = false;
                useLLMRerankerCheckbox.checked = false;
                useQueryDecompositionCheckbox.checked = false;
                break;
            case 'vector_plus_rerank':
                useBM25Checkbox.checked = false;
                useRerankerCheckbox.checked = true;
                useLLMRerankerCheckbox.checked = false;
                useQueryDecompositionCheckbox.checked = false;
                break;
            case 'vector_plus_bm25':
                useBM25Checkbox.checked = true;
                useRerankerCheckbox.checked = false;
                useLLMRerankerCheckbox.checked = false;
                useQueryDecompositionCheckbox.checked = false;
                break;
            case 'vector_bm25_rerank':
                useBM25Checkbox.checked = true;
                useRerankerCheckbox.checked = true;
                useLLMRerankerCheckbox.checked = false;
                useQueryDecompositionCheckbox.checked = false;
                break;
            case 'vector_bm25_rerank_llm':
                useBM25Checkbox.checked = true;
                useRerankerCheckbox.checked = true;
                useLLMRerankerCheckbox.checked = true;
                useQueryDecompositionCheckbox.checked = false;
                break;
            case 'full_hybrid':
                useBM25Checkbox.checked = true;
                useRerankerCheckbox.checked = true;
                useLLMRerankerCheckbox.checked = true;
                useQueryDecompositionCheckbox.checked = true;
                break;
        }
    }
    
    // When custom config is updated, set preset to "custom"
    function updateCustomConfig() {
        // Create a custom option if it doesn't exist
        let customOption = presetSelect.querySelector('option[value="custom"]');
        if (!customOption) {
            customOption = document.createElement('option');
            customOption.value = 'custom';
            customOption.textContent = 'Custom Configuration';
            presetSelect.appendChild(customOption);
        }
        
        // Set the preset to custom
        presetSelect.value = 'custom';
        
        // Log the change for debugging
        console.log("Switched to custom configuration");
    }
    
    // Update Top K value display
    function updateTopKValue() {
        topKValue.textContent = topKSlider.value;
    }
    
    // Update Min Similarity value display
    function updateMinSimilarityValue() {
        minSimilarityValue.textContent = minSimilaritySlider.value;
    }
    
    // Handle query submission
    async function handleQuerySubmit() {
        const query = queryInput.value.trim();
        if (!query) {
            alert('Please enter a query');
            return;
        }
        
        // Show loading indicator
        loadingIndicator.style.display = 'flex';
        resultsContent.innerHTML = '';
        
        try {
            // Get current configuration
            const config = getCurrentConfig();
            
            // Send query to backend
            const result = await sendQuery(query, config);
            
            // Display results
            displayResults(result, query, config);
            
            // Add to history
            addToHistory(query, config, result);
            
        } catch (error) {
            console.error('Error processing query:', error);
            resultsContent.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
        } finally {
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
        }
    }
    
    // Get current configuration from UI
    function getCurrentConfig() {
        return {
            preset: presetSelect.value,
            use_vector: useVectorCheckbox.checked,
            use_bm25: useBM25Checkbox.checked,
            use_reranker: useRerankerCheckbox.checked,
            use_llm_reranker: useLLMRerankerCheckbox.checked,
            use_query_decomposition: useQueryDecompositionCheckbox.checked,
            top_k: parseInt(topKSlider.value),
            min_similarity_pct: parseInt(minSimilaritySlider.value)
        };
    }
    
    // Send query to backend API
    async function sendQuery(query, config) {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query,
                config
            })
        });
        
        if (!response.ok) {
            let errorText = 'Failed to process query';
            try {
                // FastAPI returns errors in a different format
                const errorData = await response.json();
                errorText = errorData.detail || errorText;
            } catch (e) {
                // Fallback to text if not JSON
                errorText = await response.text();
            }
            throw new Error(errorText);
        }
        
        return await response.json();
    }
    
    // Display query results in the UI
    function displayResults(result, query, config) {
        // Clear previous results
        resultsContent.innerHTML = '';
        
        // Display the answer from the LLM
        const answerHtml = formatAnswer(result.answer);
        
        const answerElement = document.createElement('div');
        answerElement.className = 'answer-box';
        answerElement.innerHTML = answerHtml;
        resultsContent.appendChild(answerElement);
        
        // Display source documents
        const documents = result.retrieved_documents;
        
        if (documents && documents.length > 0) {
            // Create sources header
            const sourcesHeader = document.createElement('div');
            sourcesHeader.className = 'sources-header';
            sourcesHeader.textContent = `Source Documents (${documents.length})`;
            resultsContent.appendChild(sourcesHeader);
            
            // Create documents container
            const documentsContainer = document.createElement('div');
            documentsContainer.className = 'documents-container';
            
            // Add each document
            documents.forEach(doc => {
                const docElement = createDocumentElement(doc);
                documentsContainer.appendChild(docElement);
            });
            
            resultsContent.appendChild(documentsContainer);
        }
    }
    
    // Format answer text with paragraph breaks
    function formatAnswer(answerText) {
        // Convert line breaks to paragraphs
        return answerText.split('\n\n')
            .filter(para => para.trim() !== '')
            .map(para => `<p>${para.replace(/\n/g, '<br>')}</p>`)
            .join('');
    }
    
    // Create a document element from a document object
    function createDocumentElement(doc) {
        const template = documentTemplate.content.cloneNode(true);
        
        // Set document title
        template.querySelector('.document-title').textContent = doc.title;
        
        // Set source and similarity
        template.querySelector('.document-source').textContent = doc.source;
        template.querySelector('.document-similarity').textContent = `${Math.round(doc.similarity * 100)}% match`;
        
        // Set content
        template.querySelector('.document-content').textContent = doc.abstract || doc.content;
        
        // Set authors if available
        const authorsElement = template.querySelector('.document-authors');
        if (doc.authors && doc.authors.length > 0) {
            authorsElement.textContent = `Authors: ${doc.authors.join(', ')}`;
        } else {
            authorsElement.style.display = 'none';
        }
        
        return template;
    }
    
    // Add a query to history
    function addToHistory(query, config, result) {
        const historyItem = {
            query,
            config,
            timestamp: new Date().toISOString(),
            result
        };
        
        // Add to front of history
        queryHistory.unshift(historyItem);
        
        // Limit history size to 20 items
        if (queryHistory.length > 20) {
            queryHistory.pop();
        }
        
        // Save to localStorage
        saveHistory();
        
        // Update UI
        renderHistory();
    }
    
    // Render history list in UI
    function renderHistory() {
        historyList.innerHTML = '';
        
        queryHistory.forEach(historyItem => {
            const historyElement = createHistoryItemElement(historyItem);
            historyList.appendChild(historyElement);
        });
    }
    
    // Create a history item element
    function createHistoryItemElement(historyItem) {
        const template = historyItemTemplate.content.cloneNode(true);
        
        // Set query text (truncate if too long)
        const queryText = historyItem.query.length > 60 
            ? historyItem.query.substring(0, 60) + '...' 
            : historyItem.query;
        template.querySelector('.history-item-query').textContent = queryText;
        
        // Set config info
        template.querySelector('.history-item-config').textContent = formatConfigInfo(historyItem.config);
        
        // Set timestamp
        const date = new Date(historyItem.timestamp);
        template.querySelector('.history-item-timestamp').textContent = 
            date.toLocaleString(undefined, { 
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        
        // Add click handler to load this history item
        const historyItemElement = template.querySelector('.history-item');
        historyItemElement.addEventListener('click', () => {
            loadHistoryItem(historyItem);
        });
        
        return template;
    }
    
    // Format configuration info for display
    function formatConfigInfo(config) {
        const parts = [];
        
        // Add preset name
        parts.push(formatPresetName(config.preset));
        
        // Add enabled features
        if (config.use_bm25) parts.push('BM25');
        if (config.use_reranker) parts.push('Reranker');
        if (config.use_llm_reranker) parts.push('LLM Filter');
        if (config.use_query_decomposition) parts.push('Q-Decomp');
        
        // Add top-k value
        parts.push(`K=${config.top_k}`);
        
        return parts.join(' • ');
    }
    
    // Format preset name for display
    function formatPresetName(preset) {
        switch(preset) {
            case 'vector_only': return 'Vector';
            case 'vector_plus_rerank': return 'Vector+Rerank';
            case 'vector_plus_bm25': return 'Vector+BM25';
            case 'vector_bm25_rerank': return 'V+BM25+Rerank';
            case 'vector_bm25_rerank_llm': return 'V+BM25+Rerank+LLM';
            case 'full_hybrid': return 'Full Hybrid';
            case 'custom': return 'Custom';
            default: return preset;
        }
    }
    
    // Load a history item
    function loadHistoryItem(historyItem) {
        // Set query input
        queryInput.value = historyItem.query;
        
        // Set configuration
        presetSelect.value = historyItem.config.preset;
        useVectorCheckbox.checked = historyItem.config.use_vector;
        useBM25Checkbox.checked = historyItem.config.use_bm25;
        useRerankerCheckbox.checked = historyItem.config.use_reranker;
        useLLMRerankerCheckbox.checked = historyItem.config.use_llm_reranker;
        useQueryDecompositionCheckbox.checked = historyItem.config.use_query_decomposition;
        
        // Set slider values
        topKSlider.value = historyItem.config.top_k;
        minSimilaritySlider.value = historyItem.config.min_similarity_pct;
        
        // Update displayed values
        updateTopKValue();
        updateMinSimilarityValue();
        
        // Display result if available
        if (historyItem.result) {
            displayResults(historyItem.result, historyItem.query, historyItem.config);
        }
    }
    
    // Clear history
    function clearHistory() {
        if (confirm('Are you sure you want to clear all history?')) {
            queryHistory = [];
            saveHistory();
            renderHistory();
        }
    }
    
    // Save history to localStorage
    function saveHistory() {
        try {
            localStorage.setItem('queryHistory', JSON.stringify(queryHistory));
        } catch (e) {
            console.error('Error saving history:', e);
        }
    }
    
    // Load history from localStorage
    function loadHistory() {
        try {
            const saved = localStorage.getItem('queryHistory');
            return saved ? JSON.parse(saved) : [];
        } catch (e) {
            console.error('Error loading history:', e);
            return [];
        }
    }
});