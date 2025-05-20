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
    const sourcesSection = document.getElementById('sourcesSection');
    const sourcesContainer = document.getElementById('sourcesContainer');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const historyList = document.getElementById('historyList');
    const clearHistoryBtn = document.getElementById('clearHistory');
    const configSidebar = document.querySelector('.config-sidebar');
    const configToggle = document.querySelector('.config-toggle');
    const historyButton = document.getElementById('historyButton');
    const historyModal = document.getElementById('historyModal');
    const closeHistoryModal = document.getElementById('closeHistoryModal');
    
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
    configToggle.addEventListener('click', toggleConfigSidebar);
    
    // History modal event listeners
    historyButton.addEventListener('click', openHistoryModal);
    closeHistoryModal.addEventListener('click', closeHistoryModalHandler);
    
    // Allow closing modal by clicking outside of it
    historyModal.addEventListener('click', event => {
        if (event.target === historyModal) {
            closeHistoryModalHandler();
        }
    });
    
    // Add escape key to close modal
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && historyModal.classList.contains('show')) {
            closeHistoryModalHandler();
        }
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl+B for config sidebar
        if (e.ctrlKey && e.key === 'b') {
            e.preventDefault();
            toggleConfigSidebar();
            // Show a temporary tooltip to inform user
            showShortcutToast('Configuration sidebar toggled (Ctrl+B)');
        }
        // Ctrl+H for history modal
        if (e.ctrlKey && e.key === 'h') {
            e.preventDefault();
            toggleHistoryModal();
            // Show a temporary tooltip to inform user
            showShortcutToast('History toggled (Ctrl+H)');
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
    
    // Toggle config sidebar
    function toggleConfigSidebar() {
        configSidebar.classList.toggle('collapsed');
        
        // Update icon based on current state
        const isCollapsed = configSidebar.classList.contains('collapsed');
        
        // Update ARIA attributes for accessibility
        configToggle.setAttribute('aria-expanded', !isCollapsed);
        
        // If we're expanding from a collapsed state, we need to fix the button position
        if (!isCollapsed) {
            // Small delay to let the CSS transition start
            setTimeout(() => {
                configToggle.style.position = '';
                configToggle.style.left = '';
                configToggle.style.top = '';
            }, 50);
        }
        
        saveSidebarState();
    }
    
    // Toggle history modal
    function toggleHistoryModal() {
        if (historyModal.classList.contains('show')) {
            closeHistoryModalHandler();
        } else {
            openHistoryModal();
        }
    }
    
    // Open history modal
    function openHistoryModal() {
        historyModal.classList.add('show');
        // Set focus to close button for accessibility
        setTimeout(() => {
            closeHistoryModal.focus();
        }, 100);
    }
    
    // Close history modal
    function closeHistoryModalHandler() {
        historyModal.classList.remove('show');
        // Return focus to history button for accessibility
        setTimeout(() => {
            historyButton.focus();
        }, 100);
    }
    
    // Save sidebar state to localStorage
    function saveSidebarState() {
        const state = {
            configSidebarCollapsed: configSidebar.classList.contains('collapsed')
        };
        localStorage.setItem('sidebarState', JSON.stringify(state));
    }
    
    // Load sidebar state from localStorage
    function loadSidebarState() {
        try {
            const savedState = localStorage.getItem('sidebarState');
            if (savedState) {
                const state = JSON.parse(savedState);
                
                // Update config sidebar
                if (state.configSidebarCollapsed) {
                    configSidebar.classList.add('collapsed');
                    // Update ARIA attributes
                    configToggle.setAttribute('aria-expanded', 'false');
                } else {
                    // Ensure ARIA attributes are correctly set
                    configToggle.setAttribute('aria-expanded', 'true');
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
        
        // Auto-collapse sidebar on mobile devices
        if (window.innerWidth <= 768) {
            if (!configSidebar.classList.contains('collapsed')) {
                configSidebar.classList.add('collapsed');
            }
        }
        
        // Hide sources section initially
        sourcesSection.style.display = 'none';
    }
    
    // Handle window resize events
    window.addEventListener('resize', () => {
        // Auto-collapse on mobile if switching from desktop to mobile
        if (window.innerWidth <= 768) {
            if (!configSidebar.classList.contains('collapsed')) {
                configSidebar.classList.add('collapsed');
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
        sourcesSection.style.display = 'none';
        sourcesContainer.innerHTML = '';
        
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
        sourcesContainer.innerHTML = '';
        
        // Display the answer from the LLM with markdown rendering
        const answerHtml = renderMarkdown(result.answer);
        
        const answerElement = document.createElement('div');
        answerElement.className = 'answer-box';
        answerElement.innerHTML = answerHtml;
        resultsContent.appendChild(answerElement);
        
        // Display source documents in the dedicated sources section
        const documents = result.retrieved_documents;
        
        if (documents && documents.length > 0) {
            // Show the sources section
            sourcesSection.style.display = 'block';
            
            // Update sources count in title
            const sourcesTitle = sourcesSection.querySelector('h2');
            sourcesTitle.textContent = `Sources (${documents.length})`;
            
            // Add each document
            documents.forEach(doc => {
                const docElement = createDocumentElement(doc);
                sourcesContainer.appendChild(docElement);
            });
        } else {
            // Hide sources section if no documents
            sourcesSection.style.display = 'none';
        }
    }
    
    // Render markdown content
    function renderMarkdown(text) {
        // Check if marked is available
        if (typeof marked !== 'undefined') {
            // Configure marked options
            marked.setOptions({
                breaks: true,          // Add <br> on a single line break
                gfm: true,             // Enable GitHub Flavored Markdown
                headerIds: true,       // Generate IDs for headings
                sanitize: false,       // Allow HTML in the markdown (we trust the server)
                smartLists: true       // Use smarter list behavior than default markdown
            });
            
            return marked.parse(text);
        } else {
            // Fallback to simple paragraph formatting if marked is not available
            return formatAnswer(text);
        }
    }
    
    // Format answer text with paragraph breaks (legacy fallback)
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
        
        if (queryHistory.length === 0) {
            const emptyMessage = document.createElement('div');
            emptyMessage.className = 'history-empty';
            emptyMessage.textContent = 'No query history yet';
            historyList.appendChild(emptyMessage);
            return;
        }
        
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
            // Close the modal after selection
            closeHistoryModalHandler();
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
    
    // Allow pressing Enter in the textarea to submit the query
    queryInput.addEventListener('keydown', function(e) {
        // Check if Enter is pressed without Shift (Shift+Enter allows multiline)
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent the default action (newline)
            handleQuerySubmit();
        }
    });
});