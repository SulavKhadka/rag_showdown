// Main application logic for RAG Pipeline Explorer
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const useVectorRetrievalCheckbox = document.getElementById('useVectorRetrieval');
    const vectorMethodSingleRadio = document.getElementById('singleVector');
    const vectorMethodMultiRadio = document.getElementById('multiVector');
    const vectorMethodSelection = document.getElementById('vectorMethodSelection');
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
    
    // Add vector retrieval method handling
    useVectorRetrievalCheckbox.addEventListener('change', function() {
        updateVectorMethodVisibility();
        validateRetrievalMethods.call(this);
    });
    vectorMethodSingleRadio.addEventListener('change', validateRetrievalMethods);
    vectorMethodMultiRadio.addEventListener('change', validateRetrievalMethods);
    useBM25Checkbox.addEventListener('change', function() {
        validateRetrievalMethods.call(this);
    });
    
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
    
    // Update vector method selection visibility based on vector retrieval toggle
    function updateVectorMethodVisibility() {
        if (useVectorRetrievalCheckbox.checked) {
            vectorMethodSelection.style.display = 'block';
        } else {
            vectorMethodSelection.style.display = 'none';
        }
    }
    
    // Validate that at least one retrieval method is selected
    function validateRetrievalMethods() {
        const isVectorEnabled = useVectorRetrievalCheckbox.checked;
        const isBM25Enabled = useBM25Checkbox.checked;
        
        // If all are unchecked, enable the opposite of what was just unchecked
        if (!isVectorEnabled && !isBM25Enabled) {
            // Determine which toggle triggered this validation
            if (this === useVectorRetrievalCheckbox) {
                // Vector was just turned off, so turn on BM25
                useBM25Checkbox.checked = true;
                showShortcutToast('BM25 enabled: At least one retrieval method must be enabled');
            } else if (this === useBM25Checkbox) {
                // BM25 was just turned off, so turn on Vector
                useVectorRetrievalCheckbox.checked = true;
                vectorMethodSingleRadio.checked = true;
                updateVectorMethodVisibility();
                showShortcutToast('Vector Retrieval enabled: At least one retrieval method must be enabled');
            } else {
                // Default case (initial load or called from elsewhere)
                // Default to enabling Vector as before
                useVectorRetrievalCheckbox.checked = true;
                vectorMethodSingleRadio.checked = true;
                updateVectorMethodVisibility();
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
        // Temporarily disable validation to avoid triggering toggles
        const originalValidateRetrievalMethods = validateRetrievalMethods;
        validateRetrievalMethods = function() {};
        
        try {
            // Set default values
            useVectorRetrievalCheckbox.checked = true;
            vectorMethodSingleRadio.checked = true;
            useBM25Checkbox.checked = false;
            useRerankerCheckbox.checked = false;
            useLLMRerankerCheckbox.checked = false;
            useQueryDecompositionCheckbox.checked = false;
            
            // Initialize vector method visibility
            updateVectorMethodVisibility();
            
            updateTopKValue();
            updateMinSimilarityValue();
            
            // Initialize sources section with minimum height
            sourcesContainer.innerHTML = '<div class="no-sources-message">Submit a query to see sources</div>';
            sourcesSection.style.display = 'block';
            
            // Call layout adjustment on initial load
            setTimeout(adjustLayoutHeights, 100);
        } finally {
            // Restore validation function
            validateRetrievalMethods = originalValidateRetrievalMethods;
        }
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
        // Determine vector retrieval method from UI
        let vector_retrieval_method = "none";
        
        if (useVectorRetrievalCheckbox.checked) {
            vector_retrieval_method = vectorMethodMultiRadio.checked ? "colbert" : "standard";
        }
        
        // For backward compatibility, set use_vector and use_colbert flags
        const useStandardVector = useVectorRetrievalCheckbox.checked && vectorMethodSingleRadio.checked;
        const useColbert = useVectorRetrievalCheckbox.checked && vectorMethodMultiRadio.checked;
        
        return {
            preset: "custom", // Always use custom preset since we removed the selector
            use_vector: useStandardVector,
            use_colbert: useColbert,
            vector_retrieval_method: vector_retrieval_method,
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
    // Clear the results content
    resultsContent.innerHTML = '';
    sourcesContainer.innerHTML = '';
    
    // Display the answer from the LLM with markdown rendering
    const answerHtml = renderMarkdown(result.answer);
    
    const answerElement = document.createElement('div');
    answerElement.className = 'answer-box';
    answerElement.innerHTML = answerHtml;
    resultsContent.appendChild(answerElement);
    
    // After displaying the answer, adjust layout heights
    adjustLayoutHeights();
        
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
                    
                    // Re-adjust layout heights when a document is expanded
                    setTimeout(adjustLayoutHeights, 50); // Small delay to let content render
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
            // Handle custom date format with underscore (e.g., "2024-06-26_10:07:57")
            if (dateString.includes('_')) {
                // Replace underscore with space or 'T' to make it ISO compatible
                dateString = dateString.replace('_', 'T');
            }
            
            const date = new Date(dateString);
            if (isNaN(date.getTime())) return '';
            
            return date.toLocaleDateString(undefined, {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
        } catch (e) {
            console.error("Error formatting date:", e, dateString);
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
        } else if (doc.source.includes('colbert')) {
            sourceType = 'Multi-Vector';
            sourceElement.classList.add('colbert');
        } else if (doc.source.includes('vector')) {
            sourceType = 'Single-Vector';
            sourceElement.classList.add('vector');
        } else if (doc.source.includes('bm25')) {
            sourceType = 'BM25';
            sourceElement.classList.add('bm25');
        }
        
        sourceElement.textContent = sourceType;
        
        // Set similarity percentage, but hide it for BM25-only results
        const similarityElement = template.querySelector('.document-similarity');
        if (doc.source === 'bm25') {
            similarityElement.style.display = 'none';
        } else {
            const similarityPercent = Math.round(doc.similarity * 100);
            similarityElement.textContent = `${similarityPercent}%`;
        }
        
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
        if (config.vector_retrieval_method) {
            if (config.vector_retrieval_method === "standard") {
                parts.push('Single-Vector');
            } else if (config.vector_retrieval_method === "colbert") {
                parts.push('Multi-Vector');
            }
        } else {
            // Backward compatibility
            if (config.use_vector) parts.push('Vector');
            if (config.use_colbert) parts.push('ColBERT');
        }
        
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
        
        // Temporarily disable validation to avoid triggering toggles
        const originalValidateRetrievalMethods = validateRetrievalMethods;
        validateRetrievalMethods = function() {};
        
        try {
            // Set configuration based on vector_retrieval_method if available
            if (historyItem.config.vector_retrieval_method) {
                if (historyItem.config.vector_retrieval_method === "none") {
                    useVectorRetrievalCheckbox.checked = false;
                } else {
                    useVectorRetrievalCheckbox.checked = true;
                    if (historyItem.config.vector_retrieval_method === "colbert") {
                        vectorMethodMultiRadio.checked = true;
                    } else {
                        vectorMethodSingleRadio.checked = true;
                    }
                }
            } else {
                // Backward compatibility
                useVectorRetrievalCheckbox.checked = Boolean(historyItem.config.use_vector);
                vectorMethodSingleRadio.checked = Boolean(historyItem.config.use_vector);
                vectorMethodMultiRadio.checked = Boolean(historyItem.config.use_colbert);
            }
            
            // Update vector method visibility based on the toggle
            updateVectorMethodVisibility();
            
            useBM25Checkbox.checked = Boolean(historyItem.config.use_bm25);
            useRerankerCheckbox.checked = Boolean(historyItem.config.use_reranker);
            useLLMRerankerCheckbox.checked = Boolean(historyItem.config.use_llm_reranker);
            useQueryDecompositionCheckbox.checked = Boolean(historyItem.config.use_query_decomposition);
            
            // After setting all toggles, validate that at least one retrieval method is checked
            if (!useVectorRetrievalCheckbox.checked && !useBM25Checkbox.checked) {
                // Default to enabling vector if nothing is selected
                useVectorRetrievalCheckbox.checked = true;
                vectorMethodSingleRadio.checked = true;
                updateVectorMethodVisibility();
            }
            
            // Set slider values
            topKSlider.value = historyItem.config.top_k;
            minSimilaritySlider.value = historyItem.config.min_similarity_pct;
            
            // Update displayed values
            updateTopKValue();
            updateMinSimilarityValue();
        } finally {
            // Restore validation function
            validateRetrievalMethods = originalValidateRetrievalMethods;
        }
        
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

    // Add this new function to adjust layout heights 
function adjustLayoutHeights() {
    const resultsContent = document.getElementById('resultsContent');
    const answerBox = document.querySelector('.answer-box');
    const sourcesSection = document.getElementById('sourcesSection');
    
    if (!sourcesSection) return;
    
    // Get the viewport height
    const viewportHeight = window.innerHeight;
    const maxAnswerHeight = viewportHeight * 0.75; // 75% of viewport height
    const minSourcesHeight = viewportHeight * 0.25; // 25% of viewport height
    
    // If no answer box yet (initial state or error), give minimum height to sources
    if (!answerBox) {
        sourcesSection.style.height = `${minSourcesHeight}px`;
        return;
    }
    
    // Get the actual content height of the answer
    const answerHeight = answerBox.scrollHeight;
    
    // Adjust heights based on content
    if (answerHeight < maxAnswerHeight) {
        // If answer is smaller than max allowed, let it take its natural height
        resultsContent.style.height = 'auto';
        resultsContent.style.maxHeight = `${maxAnswerHeight}px`;
        
        // Give the remaining space to sources, but ensure at least minSourcesHeight
        const remainingHeight = Math.max(viewportHeight - answerHeight - 100, minSourcesHeight);
        sourcesSection.style.height = `${remainingHeight}px`;
    } else {
        // If answer would exceed max height, cap it and ensure sources gets its minimum
        resultsContent.style.height = `${maxAnswerHeight}px`;
        sourcesSection.style.height = `${minSourcesHeight}px`;
    }
    }

    // Add window resize event listener to adjust heights when window is resized
    window.addEventListener('resize', adjustLayoutHeights);

    // We no longer need tooltips for the sidebar as it's now always visible

    
});