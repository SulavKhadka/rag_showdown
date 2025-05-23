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
    
    // Authentication elements
    const loginForm = document.getElementById('loginForm');
    const authStatus = document.getElementById('authStatus');
    const welcomeText = document.getElementById('welcomeText');
    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');
    const loginBtn = document.getElementById('loginBtn');
    const registerBtn = document.getElementById('registerBtn');
    const logoutBtn = document.getElementById('logoutBtn');
    
    // State management
    let queryHistory = loadHistory();
    let authToken = localStorage.getItem('authToken');
    let currentUser = null;
    
    // Initialize UI
    initializeUI();
    renderHistory();
    loadSidebarState();
    checkAuthStatus();
    
    // Event listeners
    topKSlider.addEventListener('input', updateTopKValue);
    minSimilaritySlider.addEventListener('input', updateMinSimilarityValue);
    submitQueryBtn.addEventListener('click', handleQuerySubmit);
    clearHistoryBtn.addEventListener('click', clearHistory);
    
    // Authentication event listeners
    loginBtn.addEventListener('click', handleLogin);
    registerBtn.addEventListener('click', handleRegister);
    logoutBtn.addEventListener('click', handleLogout);
    
    // Enter key support for login form
    passwordInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleLogin();
        }
    });
    
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
        if (!authToken) {
            throw new Error('You must be logged in to submit queries');
        }
        
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${authToken}`
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
                sanitize: true,        // Sanitize HTML in the markdown
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
    
    // Truncate authors to show max 3 authors
    function formatAuthors(authors) {
        if (!authors || !authors.length) return '';
        
        if (authors.length <= 3) {
            return authors.join(', ');
        } else {
            return authors.slice(0, 3).join(', ') + ' et al';
        }
    }
    
    // Create a document element from a document object
    function createDocumentElement(doc) {
        const template = documentTemplate.content.cloneNode(true);
        
        // Determine source type display text
        let sourceType = 'Source';
        if (doc.source.includes('vector') && doc.source.includes('bm25')) {
            sourceType = 'Vector+BM25';
        } else if (doc.source.includes('colbert')) {
            sourceType = 'Multi-Vector';
        } else if (doc.source.includes('vector')) {
            sourceType = 'Single-Vector';
        } else if (doc.source.includes('bm25')) {
            sourceType = 'BM25';
        }
        
        // Build combined title with inline metadata: "Title • SourceType • 85%"
        let combinedTitle = doc.title;
        combinedTitle += ` • ${sourceType}`;
        if (doc.source !== 'bm25' && doc.similarity) {
            const similarityPercent = Math.round(doc.similarity * 100);
            combinedTitle += ` • ${similarityPercent}%`;
        }
        
        // Set the combined title
        template.querySelector('.document-title').textContent = combinedTitle;
        
        // Hide the separate source info section since it's now inline
        const sourceInfoElement = template.querySelector('.document-source-info');
        if (sourceInfoElement) {
            sourceInfoElement.style.display = 'none';
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

    // === EXPLORATION FUNCTIONALITY ===
    
    // Additional DOM elements for exploration
    const navTabs = document.querySelectorAll('.nav-tab');
    const ragConfig = document.getElementById('ragConfig');
    const exploreConfig = document.getElementById('exploreConfig');
    const ragResults = document.getElementById('ragResults');
    const exploreResults = document.getElementById('exploreResults');
    const exploreSearchInput = document.getElementById('exploreSearchInput');
    const exploreSearchButton = document.getElementById('exploreSearchButton');
    const authorFilter = document.getElementById('authorFilter');
    const yearStartFilter = document.getElementById('yearStartFilter');
    const yearEndFilter = document.getElementById('yearEndFilter');
    const sortFilter = document.getElementById('sortFilter');
    const statsGrid = document.getElementById('statsGrid');
    const documentsGrid = document.getElementById('documentsGrid');
    const paginationInfo = document.getElementById('paginationInfo');
    const paginationControls = document.getElementById('paginationControls');
    const documentModal = document.getElementById('documentModal');
    const closeDocumentModal = document.getElementById('closeDocumentModal');
    const documentModalBody = document.getElementById('documentModalBody');
    
    // Templates for exploration
    const statsCardTemplate = document.getElementById('statsCardTemplate');
    const documentListTemplate = document.getElementById('documentListTemplate');
    
    // Exploration state
    let currentView = 'rag';
    let currentPage = 1;
    let currentSearch = '';
    let currentFilters = {};
    let explorationHistory = [];
    
    // Navigation tab event listeners
    navTabs.forEach(tab => {
        tab.addEventListener('click', () => switchView(tab.dataset.view));
    });
    
    // Debounce function to limit how often a function can be called
    function debounce(func, wait) {
        let timeout;
        return function(...args) {
            const context = this;
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(context, args), wait);
        };
    }
    
    // Exploration search and filter event listeners
    exploreSearchButton.addEventListener('click', performExploreSearch);
    exploreSearchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performExploreSearch();
    });
    
    authorFilter.addEventListener('change', performExploreSearch);
    
    // Apply debounce to year filters with 300ms delay
    const debouncedSearch = debounce(performExploreSearch, 300);
    yearStartFilter.addEventListener('input', debouncedSearch);
    yearEndFilter.addEventListener('input', debouncedSearch);
    
    // Add keyboard support for immediate search on Enter key
    yearStartFilter.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performExploreSearch();
    });
    
    yearEndFilter.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performExploreSearch();
    });
    
    sortFilter.addEventListener('change', performExploreSearch);
    
    // Document modal event listeners
    closeDocumentModal.addEventListener('click', closeDocumentModalHandler);
    documentModal.addEventListener('click', (e) => {
        if (e.target === documentModal) closeDocumentModalHandler();
    });
    
    // Functions for exploration
    function switchView(view) {
        currentView = view;
        
        // Update tab states
        navTabs.forEach(tab => {
            tab.classList.toggle('active', tab.dataset.view === view);
        });
        
        // Show/hide appropriate config panels
        if (view === 'rag') {
            ragConfig.classList.remove('hidden');
            exploreConfig.classList.add('hidden');
            ragResults.classList.remove('hidden');
            exploreResults.classList.add('hidden');
        } else if (view === 'explore') {
            ragConfig.classList.add('hidden');
            exploreConfig.classList.remove('hidden');
            ragResults.classList.add('hidden');
            exploreResults.classList.remove('hidden');
            
            // Initialize exploration view
            initializeExploration();
        }
        
        // Save view state
        localStorage.setItem('currentView', view);
    }
    
    async function initializeExploration() {
        try {
            // Load dataset stats
            await loadDatasetStats();
            
            // Load authors for filter
            await loadAuthors();
            
            // Load initial documents
            await loadDocuments();
        } catch (error) {
            console.error('Error initializing exploration:', error);
            showErrorMessage('Failed to load exploration data');
        }
    }
    
    async function loadDatasetStats() {
        try {
            const response = await fetch('/api/dataset/stats');
            if (!response.ok) throw new Error('Failed to load stats');
            
            const stats = await response.json();
            renderStats(stats);
        } catch (error) {
            console.error('Error loading stats:', error);
            showErrorMessage('Failed to load dataset statistics');
        }
    }
    
    function renderStats(stats) {
        const overviewStats = document.getElementById('overviewStats');
        
        // Handle undefined stats
        if (!stats) {
            overviewStats.innerHTML = '<p>Unable to load statistics</p>';
            return;
        }
        
        // Create compact stats for sidebar
        const compactStats = [
            {
                label: 'Documents',
                value: (stats.total_documents || 0).toLocaleString()
            },
            {
                label: 'Authors',
                value: (stats.top_authors?.length || 0).toLocaleString()
            },
            {
                label: 'Date Range',
                value: `${stats.date_range?.earliest?.slice(0, 4) || 'N/A'} - ${stats.date_range?.latest?.slice(0, 4) || 'N/A'}`
            },
            {
                label: 'Avg Length',
                value: `${Math.round(stats.avg_abstract_length || 0)} chars`
            }
        ];
        
        overviewStats.innerHTML = compactStats.map(stat => `
            <div class="overview-stat">
                <div class="overview-stat-label">${stat.label}</div>
                <div class="overview-stat-value">${stat.value}</div>
            </div>
        `).join('');
    }
    
    async function loadAuthors() {
        try {
            const response = await fetch('/api/dataset/authors');
            if (!response.ok) throw new Error('Failed to load authors');
            
            const authors = await response.json();
            
            // Clear existing options except the first one
            authorFilter.innerHTML = '<option value="">All Authors</option>';
            
            // Add top 50 authors to avoid overwhelming the dropdown
            authors.slice(0, 50).forEach(author => {
                const option = document.createElement('option');
                option.value = author.name;
                option.textContent = `${author.name} (${author.count})`;
                authorFilter.appendChild(option);
            });
        } catch (error) {
            console.error('Error loading authors:', error);
        }
    }
    
    async function loadDocuments(page = 1) {
        try {
            const params = new URLSearchParams({
                page: page.toString(),
                limit: '20'
            });
            
            if (currentSearch) params.append('search', currentSearch);
            if (currentFilters.author) params.append('author', currentFilters.author);
            if (currentFilters.yearStart) params.append('year_start', currentFilters.yearStart);
            if (currentFilters.yearEnd) params.append('year_end', currentFilters.yearEnd);
            if (currentFilters.sort) params.append('sort', currentFilters.sort);
            
            const response = await fetch(`/api/documents?${params}`);
            if (!response.ok) {
                const errorText = await response.text();
                console.error('API Error:', response.status, errorText);
                throw new Error(`Failed to load documents: ${response.status}`);
            }
            
            const data = await response.json();
            renderDocuments(data);
            renderPagination(data);
            
            currentPage = page;
        } catch (error) {
            console.error('Error loading documents:', error);
            showErrorMessage('Failed to load documents');
        }
    }
    
    function renderDocuments(data) {
        documentsGrid.innerHTML = '';
        
        if (!data || !data.documents || data.documents.length === 0) {
            documentsGrid.innerHTML = '<p style="text-align: center; color: #64748B; padding: 40px;">No documents found matching your criteria.</p>';
            return;
        }
        
        data.documents.forEach(doc => {
            const item = documentListTemplate.content.cloneNode(true);
            
            item.querySelector('.document-list-title').textContent = doc.title || 'Untitled';
            item.querySelector('.document-list-date').textContent = doc.published || 'Unknown date';
            // Hide source to make display more compact
            item.querySelector('.document-list-source').style.display = 'none';
            item.querySelector('.document-list-authors').textContent = formatAuthors(doc.authors) || 'No authors listed';
            item.querySelector('.document-list-preview').textContent = doc.abstract_preview || 'No preview available';
            
            // Add event listeners for buttons
            const viewBtn = item.querySelector('.btn-view-document');
            const similarBtn = item.querySelector('.btn-similar-documents');
            
            viewBtn.addEventListener('click', () => showDocumentDetail(doc.id));
            similarBtn.addEventListener('click', () => showSimilarDocuments(doc.id));
            
            documentsGrid.appendChild(item);
        });
    }
    
    function renderPagination(data) {
        
        // Handle undefined or invalid data
        if (!data || !data.page || !data.limit || !data.total) {
            paginationInfo.textContent = 'No pagination data available';
            paginationControls.innerHTML = '';
            return;
        }
        
        const startItem = ((data.page - 1) * data.limit) + 1;
        const endItem = Math.min(data.page * data.limit, data.total);
        paginationInfo.textContent = `Showing ${startItem}-${endItem} of ${data.total} documents`;
        
        paginationControls.innerHTML = '';
        
        // Previous button
        const prevBtn = document.createElement('button');
        prevBtn.textContent = 'Previous';
        prevBtn.disabled = data.page <= 1;
        prevBtn.addEventListener('click', () => loadDocuments(data.page - 1));
        paginationControls.appendChild(prevBtn);
        
        // Page numbers (show current and nearby pages)
        const startPage = Math.max(1, data.page - 2);
        const endPage = Math.min(data.total_pages, data.page + 2);
        
        for (let i = startPage; i <= endPage; i++) {
            const pageBtn = document.createElement('button');
            pageBtn.textContent = i.toString();
            pageBtn.classList.toggle('active', i === data.page);
            pageBtn.addEventListener('click', () => loadDocuments(i));
            paginationControls.appendChild(pageBtn);
        }
        
        // Next button
        const nextBtn = document.createElement('button');
        nextBtn.textContent = 'Next';
        nextBtn.disabled = data.page >= data.total_pages;
        nextBtn.addEventListener('click', () => loadDocuments(data.page + 1));
        paginationControls.appendChild(nextBtn);
    }
    
    // Perform document search with filters
    async function performExploreSearch() {
        // Show loading state in documents grid
        documentsGrid.innerHTML = '<div class="search-loading"><div class="spinner"></div><p>Loading documents...</p></div>';
        
        currentSearch = exploreSearchInput.value.trim();
        currentFilters = {
            author: authorFilter.value,
            yearStart: yearStartFilter.value,
            yearEnd: yearEndFilter.value,
            sort: sortFilter.value
        };
        
        await loadDocuments(1); // Reset to first page
    }
    
    async function showDocumentDetail(docId) {
        try {
            const response = await fetch(`/api/documents/${docId}`);
            if (!response.ok) throw new Error('Failed to load document');
            
            const doc = await response.json();
            
            documentModalBody.innerHTML = `
                <div class="document-detail">
                    <h3>${doc.title}</h3>
                    <div class="document-meta">
                        <p><strong>Authors:</strong> ${formatAuthors(doc.authors) || 'Not specified'}</p>
                        <p><strong>Published:</strong> ${doc.published}</p>
                        ${doc.link ? `<p><strong>Link:</strong> <a href="${doc.link}" target="_blank">${doc.link}</a></p>` : ''}
                    </div>
                    <div class="document-abstract">
                        <h4>Abstract</h4>
                        <p>${doc.abstract}</p>
                    </div>
                </div>
            `;
            
            documentModal.classList.add('show');
        } catch (error) {
            console.error('Error loading document detail:', error);
            showErrorMessage('Failed to load document details');
        }
    }
    
    async function showSimilarDocuments(docId) {
        try {
            const response = await fetch(`/api/documents/${docId}/similar`);
            if (!response.ok) throw new Error('Failed to load similar documents');
            
            const similarDocs = await response.json();
            
            let content = '<h3>Similar Documents</h3>';
            if (similarDocs.length === 0) {
                content += '<p>No similar documents found.</p>';
            } else {
                content += '<div class="similar-documents-list">';
                similarDocs.forEach(doc => {
                    content += `
                        <div class="similar-document-item" style="margin-bottom: 16px; padding: 16px; border: 1px solid #E2E8F0; border-radius: 6px;">
                            <h4 style="margin: 0 0 8px 0;">${doc.title}</h4>
                            <p style="margin: 0 0 8px 0; font-size: 14px; color: #64748B;">
                                ${formatAuthors(doc.authors)} • ${doc.published}
                            </p>
                            <p style="margin: 0; font-size: 14px; color: #475569;">${doc.abstract_preview}</p>
                            <button onclick="showDocumentDetail(${doc.id})" style="margin-top: 8px; padding: 4px 8px; background: #3B82F6; color: white; border: none; border-radius: 4px; cursor: pointer;">View Details</button>
                        </div>
                    `;
                });
                content += '</div>';
            }
            
            documentModalBody.innerHTML = content;
            documentModal.classList.add('show');
        } catch (error) {
            console.error('Error loading similar documents:', error);
            showErrorMessage('Failed to load similar documents');
        }
    }
    
    function closeDocumentModalHandler() {
        documentModal.classList.remove('show');
    }
    
    function showErrorMessage(message) {
        // Simple error display - could be enhanced with a proper notification system
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = 'position: fixed; top: 20px; right: 20px; background: #EF4444; color: white; padding: 12px 16px; border-radius: 6px; z-index: 1001;';
        errorDiv.textContent = message;
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            document.body.removeChild(errorDiv);
        }, 5000);
    }
    
    // Initialize exploration if returning to that view
    const savedView = localStorage.getItem('currentView');
    if (savedView === 'explore') {
        switchView('explore');
    }
    
    // Authentication Functions
    async function checkAuthStatus() {
        if (authToken) {
            try {
                const response = await fetch('/api/auth/me', {
                    headers: {
                        'Authorization': `Bearer ${authToken}`
                    }
                });
                
                if (response.ok) {
                    currentUser = await response.json();
                    updateAuthUI(true);
                } else {
                    // Token invalid, clear it
                    localStorage.removeItem('authToken');
                    authToken = null;
                    updateAuthUI(false);
                }
            } catch (error) {
                console.error('Error checking auth status:', error);
                updateAuthUI(false);
            }
        } else {
            updateAuthUI(false);
        }
    }
    
    function updateAuthUI(isAuthenticated) {
        if (isAuthenticated && currentUser) {
            loginForm.classList.add('hidden');
            authStatus.classList.remove('hidden');
            welcomeText.textContent = `Welcome, ${currentUser.username}`;
        } else {
            loginForm.classList.remove('hidden');
            authStatus.classList.add('hidden');
            currentUser = null;
        }
    }
    
    async function handleLogin() {
        const username = usernameInput.value.trim();
        const password = passwordInput.value;
        
        if (!username || !password) {
            showError('Please enter both username and password');
            return;
        }
        
        try {
            loginBtn.disabled = true;
            loginBtn.textContent = 'Logging in...';
            
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username, password })
            });
            
            if (response.ok) {
                const data = await response.json();
                authToken = data.access_token;
                currentUser = data.user;
                localStorage.setItem('authToken', authToken);
                updateAuthUI(true);
                
                // Clear form
                usernameInput.value = '';
                passwordInput.value = '';
                
                showSuccess('Login successful!');
            } else {
                const error = await response.json();
                showError(error.detail || 'Login failed');
            }
        } catch (error) {
            console.error('Login error:', error);
            showError('Login failed. Please try again.');
        } finally {
            loginBtn.disabled = false;
            loginBtn.textContent = 'Login';
        }
    }
    
    async function handleRegister() {
        const username = usernameInput.value.trim();
        const password = passwordInput.value;
        
        if (!username || !password) {
            showError('Please enter both username and password');
            return;
        }
        
        if (password.length < 6) {
            showError('Password must be at least 6 characters long');
            return;
        }
        
        try {
            registerBtn.disabled = true;
            registerBtn.textContent = 'Registering...';
            
            const response = await fetch('/api/auth/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username, password })
            });
            
            if (response.ok) {
                const data = await response.json();
                authToken = data.access_token;
                currentUser = data.user;
                localStorage.setItem('authToken', authToken);
                updateAuthUI(true);
                
                // Clear form
                usernameInput.value = '';
                passwordInput.value = '';
                
                showSuccess('Registration successful!');
            } else {
                const error = await response.json();
                showError(error.detail || 'Registration failed');
            }
        } catch (error) {
            console.error('Registration error:', error);
            showError('Registration failed. Please try again.');
        } finally {
            registerBtn.disabled = false;
            registerBtn.textContent = 'Register';
        }
    }
    
    function handleLogout() {
        authToken = null;
        currentUser = null;
        localStorage.removeItem('authToken');
        updateAuthUI(false);
        showSuccess('Logged out successfully');
    }
    
    function showSuccess(message) {
        const successDiv = document.createElement('div');
        successDiv.style.cssText = 'position: fixed; top: 20px; right: 20px; background: #10B981; color: white; padding: 12px 16px; border-radius: 6px; z-index: 1001;';
        successDiv.textContent = message;
        document.body.appendChild(successDiv);
        
        setTimeout(() => {
            if (document.body.contains(successDiv)) {
                document.body.removeChild(successDiv);
            }
        }, 3000);
    }
    
    // Make showDocumentDetail globally accessible for similar documents
    window.showDocumentDetail = showDocumentDetail;
    
});