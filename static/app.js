// Main application logic for RAG Pipeline Explorer
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
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
    topKSlider.addEventListener('input', updateTopKValue);
    minSimilaritySlider.addEventListener('input', updateMinSimilarityValue);
    submitQueryBtn.addEventListener('click', handleQuerySubmit);
    clearHistoryBtn.addEventListener('click', clearHistory);
    
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
    
    // Add retrieval method validation
    useVectorCheckbox.addEventListener('change', validateRetrievalMethods);
    useBM25Checkbox.addEventListener('change', validateRetrievalMethods);
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl+H for history modal
        if (e.ctrlKey && e.key === 'h') {
            e.preventDefault();
            toggleHistoryModal();
            // Show a temporary tooltip to inform user
            showShortcutToast('History toggled (Ctrl+H)');
        }
    });
    
    // Validate that at least one retrieval method is selected
    function validateRetrievalMethods() {
        if (!useVectorCheckbox.checked && !useBM25Checkbox.checked) {
            // If both are unchecked, force at least one to be checked
            if (this === useVectorCheckbox) {
                useBM25Checkbox.checked = true;
                showShortcutToast('At least one retrieval method must be enabled');
            } else {
                useVectorCheckbox.checked = true;
                showShortcutToast('At least one retrieval method must be enabled');
            }
        }
    }
    
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
    
    // Save sidebar state to localStorage - kept for compatibility
    function saveSidebarState() {
        // No need to save state anymore as sidebar is always visible
    }
    
    // Initialize UI
    function initializeUI() {
        // Set default values
        useVectorCheckbox.checked = true;
        useBM25Checkbox.checked = false;
        useRerankerCheckbox.checked = false;
        useLLMRerankerCheckbox.checked = false;
        useQueryDecompositionCheckbox.checked = false;
        
        updateTopKValue();
        updateMinSimilarityValue();
        
        // Hide sources section initially
        sourcesSection.style.display = 'none';
    }
    
    // We don't need to load sidebar state anymore since it's always visible
    function loadSidebarState() {
        // Function kept for compatibility but doesn't do anything now
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
            preset: "custom", // Always use custom preset since we removed the selector
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

            // Add event listeners for collapsible documents after all are added
            setupCollapsibleDocuments();
        } else {
            // Hide sources section if no documents
            sourcesSection.style.display = 'none';
        }
    }
    
    // Setup collapsible document cards
    function setupCollapsibleDocuments() {
        const documentHeaders = document.querySelectorAll('.document-header');
        
        documentHeaders.forEach(header => {
            header.addEventListener('click', function() {
                const document = this.closest('.document');
                document.classList.toggle('document-expanded');
                
                // Toggle ARIA attributes for accessibility
                const isExpanded = document.classList.contains('document-expanded');
                this.setAttribute('aria-expanded', isExpanded);
                
                // Add subtle animation effect using max-height
                const content = document.querySelector('.document-content-wrapper');
                
                if (isExpanded) {
                    // Set max-height to a large value for expansion
                    content.style.maxHeight = content.scrollHeight + 'px';
                } else {
                    // Set max-height to 0 for collapse
                    content.style.maxHeight = '0';
                }
            });
        });
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
    
    // Format date function
    function formatDate(dateString) {
        if (!dateString) return '';
        
        // Try to parse the date
        try {
            const date = new Date(dateString);
            if (isNaN(date.getTime())) return '';
            
            return date.toLocaleDateString(undefined, {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
        } catch (e) {
            return '';
        }
    }
    
    // Truncate authors to show max 5 authors
    function formatAuthors(authors) {
        if (!authors || !authors.length) return '';
        
        if (authors.length <= 5) {
            return authors.join(', ');
        } else {
            return authors.slice(0, 5).join(', ') + '...';
        }
    }
    
    // Create a document element from a document object
    function createDocumentElement(doc) {
        const template = documentTemplate.content.cloneNode(true);
        
        // Set document title
        template.querySelector('.document-title').textContent = doc.title;
        
        // Set source with appropriate class
        const sourceElement = template.querySelector('.document-source');
        
        // Determine source type display text
        let sourceType = 'Source';
        if (doc.source.includes('vector') && doc.source.includes('bm25')) {
            sourceType = 'Vector+BM25';
            sourceElement.classList.add('combined');
        } else if (doc.source.includes('vector')) {
            sourceType = 'Vector';
            sourceElement.classList.add('vector');
        } else if (doc.source.includes('bm25')) {
            sourceType = 'BM25';
            sourceElement.classList.add('bm25');
        }
        
        sourceElement.textContent = sourceType;
        
        // Set similarity percentage
        const similarityPercent = Math.round(doc.similarity * 100);
        template.querySelector('.document-similarity').textContent = `${similarityPercent}%`;
        
        // Set content
        template.querySelector('.document-content').textContent = doc.abstract || doc.content;
        
        // Set authors if available (with truncation)
        const authorsMetaElement = template.querySelector('.document-authors-meta');
        if (doc.authors && doc.authors.length > 0) {
            authorsMetaElement.textContent = `Authors: ${formatAuthors(doc.authors)}`;
        } else {
            authorsMetaElement.style.display = 'none';
        }
        
        // Set publication date if available
        const dateMetaElement = template.querySelector('.document-date-meta');
        if (doc.published) {
            dateMetaElement.textContent = `Published: ${formatDate(doc.published)}`;
        } else {
            dateMetaElement.style.display = 'none';
        }
        
        // Set initial ARIA attributes for accessibility
        template.querySelector('.document-header').setAttribute('aria-expanded', 'false');
        
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
        
        // Add enabled features based on actual configuration
        if (config.use_vector) parts.push('Vector');
        if (config.use_bm25) parts.push('BM25');
        if (config.use_reranker) parts.push('Reranker');
        if (config.use_llm_reranker) parts.push('LLM Filter');
        if (config.use_query_decomposition) parts.push('Q-Decomp');
        
        // Add top-k value
        parts.push(`K=${config.top_k}`);
        
        return parts.join(' • ');
    }
    
    // Load a history item
    function loadHistoryItem(historyItem) {
        // Set query input
        queryInput.value = historyItem.query;
        
        // Set configuration checkboxes - using explicit boolean conversion
        useVectorCheckbox.checked = Boolean(historyItem.config.use_vector);
        useBM25Checkbox.checked = Boolean(historyItem.config.use_bm25);
        useRerankerCheckbox.checked = Boolean(historyItem.config.use_reranker);
        useLLMRerankerCheckbox.checked = Boolean(historyItem.config.use_llm_reranker);
        useQueryDecompositionCheckbox.checked = Boolean(historyItem.config.use_query_decomposition);
        
        // Validate at least one retrieval method is checked
        if (!useVectorCheckbox.checked && !useBM25Checkbox.checked) {
            // Default to BM25 if neither is selected (should not happen with backend validation)
            useBM25Checkbox.checked = true;
        }
        
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

    // We no longer need tooltips for the sidebar as it's now always visible
});