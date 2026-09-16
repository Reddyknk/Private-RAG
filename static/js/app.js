/**
 * Private RAG Frontend Application Logic
 */

// Application state
let currentConversations = [];
let currentLogs = [];
let selectedConversationId = null;
let activeModalPayload = null;
let activeEmbedderModel = "all-minilm";
let pendingEmbedderModel = null;
let availableEmbedders = [];
let availableChatModels = [];
let selectedChatModel = "";

document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    loadStats();
    loadLogs();
    loadEmbedderModels();
    loadChatModels();
});

// ==========================================
// 1. Tab Switching
// ==========================================
function switchTab(tabId) {
    // Hide all pages
    document.querySelectorAll(".page-view").forEach(page => {
        page.classList.remove("active");
    });

    // Deactivate all tab buttons
    document.querySelectorAll(".nav-tab").forEach(btn => {
        btn.classList.remove("active");
        btn.setAttribute("aria-selected", "false");
    });

    // Activate selected page and button
    const targetPage = document.getElementById(tabId);
    if (targetPage) {
        targetPage.classList.add("active");
    }

    if (tabId === "pageIngest") {
        const btn = document.getElementById("tabBtnIngest");
        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        loadStats();
    } else if (tabId === "pageQuery") {
        const btn = document.getElementById("tabBtnQuery");
        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        document.getElementById("inputQuestion").focus();
    } else if (tabId === "pageTelemetry") {
        const btn = document.getElementById("tabBtnTelemetry");
        if (btn) {
            btn.classList.add("active");
            btn.setAttribute("aria-selected", "true");
        }
        loadTelemetry();
    } else if (tabId === "pageLogs") {
        const btn = document.getElementById("tabBtnLogs");
        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        loadLogs();
    }
}

// ==========================================
// 2. Health & Status
// ==========================================
async function checkHealth() {
    const pill = document.getElementById("backendStatusPill");
    const label = document.getElementById("backendStatusLabel");
    const dot = pill.querySelector(".status-dot");

    try {
        const res = await fetch("/api/health");
        const data = await res.json();

        if (data.status === "online") {
            const ollamaStatus = data.ollama?.available ? "Ollama Ready" : "Ollama Offline";
            const gemmaStatus = data.gemma?.api_key_configured ? "Gemma Ready" : "Missing API Key";
            
            label.textContent = `${ollamaStatus} • ${gemmaStatus}`;
            dot.classList.remove("error");
            
            if (data.gemma?.primary_model) {
                const modelBadge = document.getElementById("activeGemmaModel");
                if (modelBadge) {
                    modelBadge.textContent = data.gemma.primary_model.replace("models/", "");
                }
            }
        } else {
            label.textContent = "Service Error";
            dot.classList.add("error");
        }
    } catch (err) {
        label.textContent = "Server Offline";
        dot.classList.add("error");
    }
}

// ==========================================
// 3. Page 1: Vector DB Ingestion
// ==========================================
function toggleSourceTypeInputs() {
    const isUrl = document.getElementById("radioUrl").checked;
    const label = document.getElementById("labelInputPath");
    const input = document.getElementById("inputPath");
    const hint = document.getElementById("hintInputPath");

    if (isUrl) {
        label.textContent = "Web URL to Crawl & Ingest";
        input.placeholder = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation";
        hint.textContent = "Fetches article text, strips HTML tags, and breaks into vector chunks.";
    } else {
        label.textContent = "Local Directory Path to Ingest";
        input.placeholder = "./sample_docs or /path/to/my/documents";
        hint.textContent = "Recursively parses .txt, .md, .pdf, .json, and code files in directory.";
    }
}

function setPreset(type, value) {
    if (type === "url") {
        document.getElementById("radioUrl").checked = true;
    } else {
        document.getElementById("radioDir").checked = true;
    }
    toggleSourceTypeInputs();
    document.getElementById("inputPath").value = value;
}

async function handleIngest(event) {
    event.preventDefault();
    const isUrl = document.getElementById("radioUrl").checked;
    const sourceType = isUrl ? "url" : "directory";
    const path = document.getElementById("inputPath").value.trim();
    const chunkSize = document.getElementById("inputChunkSize").value;
    const chunkOverlap = document.getElementById("inputChunkOverlap").value;

    if (!path) return;

    // UI Progress State
    const submitBtn = document.getElementById("btnSubmitIngest");
    const spinner = document.getElementById("ingestSpinner");
    const emptyState = document.getElementById("ingestEmptyState");
    const activeCard = document.getElementById("ingestActiveCard");
    const title = document.getElementById("ingestTitle");
    const message = document.getElementById("ingestMessage");
    const badge = document.getElementById("ingestBadge");
    const elapsed = document.getElementById("ingestElapsed");

    submitBtn.disabled = true;
    spinner.style.display = "inline-block";
    emptyState.style.display = "none";
    activeCard.style.display = "block";

    badge.textContent = "Processing";
    badge.className = "status-badge";
    title.textContent = `Ingesting ${sourceType === "url" ? "Web Page" : "Directory"}...`;
    message.textContent = `Extracting text, chunking, and computing Ollama embeddings for ${path}...`;

    const startTime = Date.now();
    const timer = setInterval(() => {
        elapsed.textContent = `${((Date.now() - startTime) / 1000).toFixed(1)}s`;
    }, 100);

    try {
        const response = await fetch("/api/ingest", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                source_type: sourceType,
                path: path,
                chunk_size: parseInt(chunkSize, 10),
                chunk_overlap: parseInt(chunkOverlap, 10)
            })
        });

        clearInterval(timer);
        const data = await response.json();

        if (response.ok && data.status === "success") {
            badge.textContent = "Indexed";
            badge.classList.add("badge-success");
            title.textContent = "Vector Database Updated!";
            message.textContent = `Saved ${data.chunks_count} chunks into database/. Total chunks in DB: ${data.total_documents_in_db}.`;
            showToast(`Ingested ${data.chunks_count} chunks from ${path}!`, "success");
            loadStats();
            loadLogs();
        } else {
            badge.textContent = "Failed";
            badge.classList.add("badge-error");
            title.textContent = "Ingestion Error";
            message.textContent = data.error || data.message || "An unknown error occurred during ingestion.";
            showToast(data.error || "Ingestion failed", "error");
        }
    } catch (err) {
        clearInterval(timer);
        badge.textContent = "Failed";
        badge.classList.add("badge-error");
        title.textContent = "Network Error";
        message.textContent = err.message;
        showToast("Connection to server failed", "error");
    } finally {
        submitBtn.disabled = false;
        spinner.style.display = "none";
    }
}

async function loadStats() {
    try {
        const res = await fetch("/api/stats");
        const data = await res.json();

        if (data.vector_store) {
            document.getElementById("statTotalChunks").textContent = data.vector_store.total_chunks || 0;
            document.getElementById("statTotalSources").textContent = data.vector_store.unique_sources || 0;
            document.getElementById("statDbSize").textContent = `${data.vector_store.db_size_mb || 0} MB`;

            // Populate Sources List with documents from Document Database
            const listElem = document.getElementById("indexedSourcesList");
            const sources = data.vector_store.sources_list || data.vector_store.sources || [];
            const details = data.vector_store.source_details || [];
            
            if (sources.length === 0) {
                listElem.innerHTML = `<p class="text-muted" style="padding: 10px 0;">No documents indexed yet in Document Database.</p>`;
            } else if (details.length > 0) {
                listElem.innerHTML = details.map(d => {
                    const icon = d.is_url ? "🌐" : "📄";
                    const shortName = d.name || d.source;
                    const chunkLabel = d.chunk_count ? `${d.chunk_count} chunk${d.chunk_count > 1 ? 's' : ''}` : '';
                    return `
                    <div class="source-item" style="display: flex; align-items: center; justify-content: space-between; padding: 10px 12px; margin-bottom: 8px; background: rgba(255, 255, 255, 0.04); border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08);">
                        <div style="display: flex; align-items: center; gap: 10px; overflow: hidden; flex: 1; margin-right: 8px;">
                            <span style="font-size: 16px;">${icon}</span>
                            <div style="display: flex; flex-direction: column; overflow: hidden;">
                                <span class="source-title" style="font-weight: 600; font-size: 13px;" title="${escapeHtml(d.source)}">${escapeHtml(shortName)}</span>
                                <span style="font-size: 11.5px; color: var(--text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${escapeHtml(d.source)}">${escapeHtml(d.source)}</span>
                            </div>
                        </div>
                        <div style="display: flex; align-items: center; gap: 6px; flex-shrink: 0;">
                            ${chunkLabel ? `<span class="badge" style="font-size: 11px; background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3);">${chunkLabel}</span>` : ''}
                            <span class="badge badge-private">Docs DB</span>
                        </div>
                    </div>`;
                }).join("");
            } else {
                listElem.innerHTML = sources.map(s => `
                    <div class="source-item">
                        <span class="source-title" title="${escapeHtml(s)}">${escapeHtml(s)}</span>
                        <span class="badge badge-private">Docs DB</span>
                    </div>
                `).join("");
            }
        }

        if (data.logs_summary) {
            document.getElementById("logBadgeCount").textContent = data.logs_summary.total_logs || 0;
            document.getElementById("metricTotalCalls").textContent = data.logs_summary.total_logs || 0;
            document.getElementById("metricGemmaCalls").textContent = data.logs_summary.type_breakdown?.google_ai_studio_gemma || 0;
            
            const ollamaCount = (data.logs_summary.type_breakdown?.local_ollama_embedding || 0) + 
                                (data.logs_summary.type_breakdown?.local_ollama_embed_batch || 0);
            document.getElementById("metricOllamaCalls").textContent = ollamaCount;
            document.getElementById("metricAvgLatency").textContent = `${data.logs_summary.avg_latency_ms || 0} ms`;
        }
    } catch (err) {
        console.error("Failed to load stats:", err);
    }
}

async function confirmResetDb() {
    if (!confirm("Are you sure you want to reset the private vector database? All indexed document vectors will be erased.")) {
        return;
    }
    try {
        const res = await fetch("/api/database/reset", { method: "POST" });
        const data = await res.json();
        if (res.ok) {
            showToast("Vector database reset successfully.", "info");
            loadStats();
        } else {
            showToast(data.error || "Reset failed", "error");
        }
    } catch (err) {
        showToast("Error resetting database", "error");
    }
}

// ==========================================
// 4. Page 2: Private RAG Query (Gemma)
// ==========================================
function fillQuestion(text) {
    const input = document.getElementById("inputQuestion");
    input.value = text;
    input.focus();
}

function handleTextareaKey(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        handleQuery(event);
    }
}

async function handleQuery(event) {
    if (event) event.preventDefault();
    const input = document.getElementById("inputQuestion");
    const question = input.value.trim();
    const topK = document.getElementById("selectTopK").value;
    const submitBtn = document.getElementById("btnSubmitQuery");
    const spinner = document.getElementById("querySpinner");
    const qaFeed = document.getElementById("qaFeed");

    if (!question) return;

    // Append User Bubble
    appendMessage("user", question);
    input.value = "";

    // Append Assistant Loading Placeholder
    const modelDisplayName = getSelectedChatModelName();
    const assistantMsgElem = appendMessage("assistant", `Searching private vector database and generating answer with ${modelDisplayName}...`, true);

    submitBtn.disabled = true;
    spinner.style.display = "inline-block";

    const startTime = Date.now();

    try {
        const payload = {
            question: question,
            top_k: parseInt(topK, 10)
        };
        if (selectedChatModel) {
            payload.model = selectedChatModel;
            if (selectedChatModel === "custom_model_api") {
                const endpointInput = document.getElementById("inputCustomEndpoint");
                if (endpointInput && endpointInput.value.trim()) {
                    payload.custom_endpoint = endpointInput.value.trim();
                }
            }
        }

        const res = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        const duration = ((Date.now() - startTime) / 1000).toFixed(1);

        if (res.ok) {
            updateAssistantMessage(
                assistantMsgElem, 
                data.answer, 
                data.model ? `Model: ${data.model.replace("models/", "")} • ${duration}s • Log ID: ${data.log_id ? data.log_id.slice(0, 8) : 'N/A'}` : null,
                data.components || [],
                data.conversation_id
            );

            // Render Retrieved Context Chunks on right panel
            renderRetrievedChunks(data.sources || []);
            loadStats();
        } else {
            updateAssistantMessage(
                assistantMsgElem, 
                `⚠️ Error generating answer: ${data.error || "Server error"}`, 
                "Error"
            );
            showToast(data.error || "Query failed", "error");
        }
    } catch (err) {
        updateAssistantMessage(
            assistantMsgElem, 
            `⚠️ Network error: Could not reach backend server (${err.message}).`, 
            "Offline"
        );
    } finally {
        submitBtn.disabled = false;
        spinner.style.display = "none";
        qaFeed.scrollTop = qaFeed.scrollHeight;
    }
}

function appendMessage(role, text, isLoading = false) {
    const qaFeed = document.getElementById("qaFeed");
    const msg = document.createElement("div");
    msg.className = `chat-message ${role}`;

    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";
    bubble.innerHTML = formatMarkdown(text);
    if (isLoading) {
        bubble.classList.add("loading-pulse");
    }

    msg.appendChild(bubble);
    qaFeed.appendChild(msg);
    qaFeed.scrollTop = qaFeed.scrollHeight;
    return msg;
}

let _componentCardSeq = 0;

function updateAssistantMessage(msgElem, text, metaText, components = [], conversationId = null) {
    const bubble = msgElem.querySelector(".chat-bubble");
    bubble.classList.remove("loading-pulse");
    bubble.innerHTML = formatMarkdown(text);

    // If components trace provided, render the comprehensive pipeline trace directly in the message
    if (components && components.length > 0) {
        _componentCardSeq++;
        const boxId = `comp-box-${_componentCardSeq}`;
        
        const compBox = document.createElement("div");
        compBox.className = "message-components-box";
        compBox.id = boxId;

        const pillsHtml = components.map((c, i) => {
            const cardId = `${boxId}-card-${i}`;
            const compClass = (c.name || "comp").toLowerCase().replace(/\s+/g, '-');
            const durationTag = c.duration_ms !== undefined && c.duration_ms !== null ? `${c.duration_ms}ms` : '';
            return `
                <span class="comp-pill comp-pill-${compClass}" onclick="toggleSingleComponent('${cardId}', event)" title="Click to jump to ${escapeHtml(c.name)} Request & Response">
                    <span>${c.icon || '⚙️'}</span>
                    <span>${escapeHtml(c.name)}</span>
                    <span class="comp-time">${durationTag}</span>
                </span>
            `;
        }).join("");

        const detailsHtml = components.map((c, i) => {
            const cardId = `${boxId}-card-${i}`;
            const reqJson = JSON.stringify(c.request || {}, null, 2);
            const resJson = JSON.stringify(c.response || {}, null, 2);
            const statusClass = (c.status || "success") === "error" ? "badge-error" : "badge-success";
            const roleBadge = c.role ? `<span class="badge badge-info" style="font-size: 10.5px;">${escapeHtml(c.role)}</span>` : "";

            return `
                <div class="component-card" id="${cardId}">
                    <div class="component-card-top">
                        <div class="component-card-title">
                            <span>${c.icon || '⚙️'}</span>
                            <strong>${escapeHtml(c.name)}</strong>
                            ${roleBadge}
                        </div>
                        <div class="component-card-meta">
                            <span class="badge ${statusClass}">${(c.status || 'SUCCESS').toUpperCase()}</span>
                            <span>${c.duration_ms !== undefined && c.duration_ms !== null ? c.duration_ms + ' ms' : ''}</span>
                        </div>
                    </div>
                    ${c.description ? `<div class="component-desc">${escapeHtml(c.description)}</div>` : ''}
                    <div class="req-res-split">
                        <div class="req-pane">
                            <div class="pane-header">
                                <span>📤 Request Payload</span>
                                <button class="copy-mini-btn" onclick="copySnippet(this)">Copy</button>
                            </div>
                            <pre class="code-box-pre"><code>${escapeHtml(reqJson)}</code></pre>
                        </div>
                        <div class="res-pane">
                            <div class="pane-header">
                                <span>📥 Response Payload</span>
                                <button class="copy-mini-btn" onclick="copySnippet(this)">Copy</button>
                            </div>
                            <pre class="code-box-pre"><code>${escapeHtml(resJson)}</code></pre>
                        </div>
                    </div>
                </div>
            `;
        }).join("");

        compBox.innerHTML = `
            <div class="components-header" onclick="toggleComponentsDrawer(this)">
                <span class="components-title">
                    <span style="color: var(--accent-cyan);">⚡</span> 
                    Pipeline Trace (${components.length} Components: ${components.map(c => c.name).join(', ')})
                </span>
                <span class="toggle-hint">Click to inspect Requests & Responses ▼</span>
            </div>
            <div class="component-pills-row">
                ${pillsHtml}
            </div>
            <div class="components-details-container" style="display: none;">
                ${detailsHtml}
                ${conversationId ? `
                    <button type="button" class="btn-inspect-conv" onclick="goToAuditLogs('${escapeHtml(conversationId)}')">
                        <span>🔍 Inspect full execution timeline in Audit Logs & DB Explorer</span>
                    </button>
                ` : ''}
            </div>
        `;
        bubble.appendChild(compBox);
    }

    if (metaText) {
        let meta = msgElem.querySelector(".chat-meta");
        if (!meta) {
            meta = document.createElement("div");
            meta.className = "chat-meta";
            msgElem.appendChild(meta);
        }
        meta.textContent = metaText;
    }
}

function renderRetrievedChunks(chunks) {
    const container = document.getElementById("contextBody");
    const countBadge = document.getElementById("retrievedCountBadge");

    countBadge.textContent = `${chunks.length} Chunks`;

    if (!chunks || chunks.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">🔍</div>
                <h4>No Relevant Chunks Found</h4>
                <p>Ensure documents are indexed in Tab 1 before asking questions.</p>
            </div>
        `;
        return;
    }

    container.innerHTML = chunks.map((chunk, idx) => {
        const title = chunk.metadata?.title || chunk.metadata?.source || `Document ${idx + 1}`;
        const score = (chunk.score * 100).toFixed(1);
        const path = chunk.metadata?.source || "";

        return `
            <div class="context-chunk-card">
                <div class="chunk-header">
                    <span class="chunk-source-badge" title="${escapeHtml(path)}">#${idx + 1} ${escapeHtml(title)}</span>
                    <span class="chunk-score" title="Cosine Similarity">${score}% match</span>
                </div>
                <div class="chunk-text">${escapeHtml(chunk.content)}</div>
            </div>
        `;
    }).join("");
}

// ==========================================
// 5. Page 3: Two-Table Audit Logs & Conversation Inspector
// ==========================================
async function loadLogs() {
    await loadLogsAndConversations();
}

async function loadLogsAndConversations() {
    try {
        const [convRes, logsRes] = await Promise.all([
            fetch("/api/conversations?limit=100"),
            fetch("/api/logs?limit=250")
        ]);
        const convData = await convRes.json();
        const logsData = await logsRes.json();

        currentConversations = convData.conversations || [];
        currentLogs = logsData.logs || [];

        renderConversationsTable();
        filterEvents();
        loadStats();
    } catch (err) {
        console.error("Failed to load audit logs and conversations:", err);
    }
}

function renderConversationsTable() {
    const tbody = document.getElementById("conversationsTableBody");
    const badge = document.getElementById("convCountBadge");
    const metricConv = document.getElementById("metricTotalConversations");

    if (badge) badge.textContent = `${currentConversations.length} Conversations`;
    if (metricConv) metricConv.textContent = currentConversations.length;

    if (!tbody) return;

    if (!currentConversations || currentConversations.length === 0) {
        tbody.innerHTML = `<tr><td colspan="4" class="text-center text-muted" style="padding: 24px;">No conversations recorded yet. Ask a question in the Chat tab!</td></tr>`;
        return;
    }

    tbody.innerHTML = currentConversations.map(conv => {
        const timeFormatted = formatTimestamp(conv.timestamp);
        const isActive = selectedConversationId === conv.conversation_id;
        const activeClass = isActive ? "active-row" : "";
        const queryDisplay = conv.user_query || "-";
        const respDisplay = conv.agent_response || "-";

        return `
            <tr class="clickable-row ${activeClass}" onclick="selectConversation('${escapeHtml(conv.conversation_id)}')">
                <td class="mono-cell" title="${escapeHtml(conv.conversation_id)}">
                    <strong>${escapeHtml(conv.conversation_id)}</strong>
                </td>
                <td class="mono-cell">${escapeHtml(timeFormatted)}</td>
                <td class="query-cell" title="${escapeHtml(queryDisplay)}">${escapeHtml(truncate(queryDisplay, 65))}</td>
                <td class="response-cell" title="${escapeHtml(respDisplay)}">${escapeHtml(truncate(respDisplay, 65))}</td>
            </tr>
        `;
    }).join("");
}

function selectConversation(convId) {
    selectedConversationId = convId;
    const filterBadge = document.getElementById("selectedConvFilterBadge");
    const showAllBtn = document.getElementById("btnShowAllEvents");
    const subtitle = document.getElementById("eventsTableSubtitle");

    if (filterBadge) filterBadge.textContent = `Conversation: ${convId}`;
    if (showAllBtn) showAllBtn.style.display = "inline-flex";
    if (subtitle) subtitle.textContent = `Showing events for conversation ${convId}. Click any event row to inspect full JSON payload.`;

    renderConversationsTable();
    filterEvents();
}

function showAllEvents() {
    selectedConversationId = null;
    const filterBadge = document.getElementById("selectedConvFilterBadge");
    const showAllBtn = document.getElementById("btnShowAllEvents");
    const subtitle = document.getElementById("eventsTableSubtitle");

    if (filterBadge) filterBadge.textContent = "All Events";
    if (showAllBtn) showAllBtn.style.display = "none";
    if (subtitle) subtitle.textContent = "Click any event row to inspect full JSON payload.";

    renderConversationsTable();
    filterEvents();
}

function filterEvents() {
    const typeFilter = document.getElementById("filterCallType") ? document.getElementById("filterCallType").value.trim() : "";
    const searchFilter = document.getElementById("logSearchInput") ? document.getElementById("logSearchInput").value.toLowerCase().trim() : "";
    const tbody = document.getElementById("eventsTableBody");

    if (!tbody) return;

    let filtered = currentLogs;

    // Filter by selected conversation if chosen
    if (selectedConversationId) {
        filtered = filtered.filter(e => e.conversation_id === selectedConversationId);
    }

    // Filter by component event type
    if (typeFilter) {
        const tf = typeFilter.toLowerCase();
        filtered = filtered.filter(e => {
            const et = (e.event_type || e.type || "").toLowerCase();
            if (tf === "agent") return et.includes("agent");
            if (tf === "llm") return et.includes("llm") || et.includes("gemma");
            if (tf === "embedder") return et.includes("embed") || et.includes("ollama");
            if (tf === "tool") return et.includes("tool");
            if (tf === "external api") return et.includes("external") || et.includes("api");
            if (tf === "vector store") return et.includes("vector");
            return et.includes(tf);
        });
    }

    // Search filter
    if (searchFilter) {
        filtered = filtered.filter(e => {
            const desc = (e.short_description || "").toLowerCase();
            const inv = (e.invoker || "").toLowerCase();
            const target = (e.target || "").toLowerCase();
            const et = (e.event_type || e.type || "").toLowerCase();
            const payloadStr = JSON.stringify(e.payload || e.arguments || {}).toLowerCase();
            return desc.includes(searchFilter) || inv.includes(searchFilter) || target.includes(searchFilter) || et.includes(searchFilter) || payloadStr.includes(searchFilter);
        });
    }

    if (filtered.length === 0) {
        const msg = selectedConversationId 
            ? "No events found for this conversation matching the filter."
            : "No events match your filter criteria.";
        tbody.innerHTML = `<tr><td colspan="5" class="text-center text-muted" style="padding: 24px;">${msg}</td></tr>`;
        return;
    }

    tbody.innerHTML = filtered.map(event => {
        const timeFormatted = formatTimestamp(event.timestamp || event.time);
        const eventType = event.event_type || event.type || "Event";
        const invoker = event.invoker || "Agent";
        const target = event.target || "-";
        const shortDesc = event.short_description || "-";

        let typeBadgeClass = "badge-private";
        const etLower = eventType.toLowerCase();
        if (etLower.includes("llm") || etLower.includes("gemma")) typeBadgeClass = "badge-info";
        else if (etLower.includes("tool")) typeBadgeClass = "badge-success";
        else if (etLower.includes("query") || etLower === "agent") typeBadgeClass = "badge-purple";
        else if (etLower.includes("response")) typeBadgeClass = "badge-emerald";
        else if (etLower.includes("vector") || etLower.includes("ollama") || etLower === "embedder") typeBadgeClass = "badge-cyan";
        else if (etLower.includes("external") || etLower.includes("api")) typeBadgeClass = "badge-warning";
        else if (etLower.includes("error")) typeBadgeClass = "badge-error";

        return `
            <tr class="clickable-row" onclick="openEventModal('${escapeHtml(event.id)}')">
                <td class="mono-cell">${escapeHtml(timeFormatted)}</td>
                <td><span class="badge ${typeBadgeClass}">${escapeHtml(eventType)}</span></td>
                <td><span class="invoker-tag">${escapeHtml(invoker)}</span></td>
                <td class="target-cell" title="${escapeHtml(target)}">${escapeHtml(truncate(target, 35))}</td>
                <td class="desc-cell" title="${escapeHtml(shortDesc)}">${escapeHtml(truncate(shortDesc, 65))}</td>
            </tr>
        `;
    }).join("");
}

function openEventModal(eventId) {
    const event = currentLogs.find(l => l.id === eventId);
    if (!event) return;

    activeModalPayload = event;

    const modalTitle = document.getElementById("modalTitle");
    const typeBadge = document.getElementById("modalTypeBadge");
    const metaBar = document.getElementById("modalMetaBar");
    const descText = document.getElementById("modalEventDescription");
    const reqViewer = document.getElementById("modalRequestJson");
    const resViewer = document.getElementById("modalResponseJson");
    const jsonViewer = document.getElementById("modalArgsJson");

    if (modalTitle) modalTitle.textContent = `${event.event_type || event.type || 'Invocation'} Event Details`;
    if (typeBadge) {
        typeBadge.textContent = (event.status || "SUCCESS").toUpperCase();
        typeBadge.className = `badge ${event.status === 'error' ? 'badge-error' : 'badge-success'}`;
    }

    if (metaBar) {
        metaBar.innerHTML = `
            <span><strong>Time:</strong> ${event.timestamp || event.time}</span> | 
            <span><strong>Invoker:</strong> ${escapeHtml(event.invoker || 'Agent')}</span> | 
            <span><strong>Target:</strong> ${escapeHtml(event.target || '-')}</span> | 
            <span><strong>Duration:</strong> ${event.duration_ms !== null && event.duration_ms !== undefined ? event.duration_ms + ' ms' : '-'}</span> | 
            <span><strong>Conv:</strong> ${escapeHtml(event.conversation_id || 'system')}</span>
        `;
    }

    if (descText) descText.textContent = event.short_description || "No description provided.";

    // Extract standardized request and response payloads
    const reqData = (event.payload && event.payload.request) || event.arguments || {};
    const resData = (event.payload && event.payload.response) || event.response || {};

    if (reqViewer) reqViewer.textContent = JSON.stringify(reqData, null, 2);
    if (resViewer) resViewer.textContent = JSON.stringify(resData, null, 2);

    // Display comprehensive sanitized event payload
    const displayObj = {
        id: event.id,
        conversation_id: event.conversation_id,
        timestamp: event.timestamp || event.time,
        event_type: event.event_type || event.type,
        invoker: event.invoker,
        target: event.target,
        short_description: event.short_description,
        status: event.status,
        duration_ms: event.duration_ms,
        payload: event.payload || { request: reqData, response: resData }
    };

    if (jsonViewer) jsonViewer.textContent = JSON.stringify(displayObj, null, 2);

    document.getElementById("payloadModal").style.display = "flex";
}

function closeModal() {
    const modal = document.getElementById("payloadModal");
    if (modal) modal.style.display = "none";
    activeModalPayload = null;
}

function closeModalOnBackdrop(event) {
    if (event.target.id === "payloadModal") {
        closeModal();
    }
}

function toggleComponentsDrawer(headerElem) {
    const container = headerElem.parentElement.querySelector(".components-details-container");
    const hint = headerElem.querySelector(".toggle-hint");
    if (!container) return;
    const isHidden = container.style.display === "none";
    container.style.display = isHidden ? "flex" : "none";
    if (hint) {
        hint.textContent = isHidden ? "Click to collapse ▲" : "Click to inspect Requests & Responses ▼";
    }
}

function toggleSingleComponent(cardId, event) {
    if (event) event.stopPropagation();
    const card = document.getElementById(cardId);
    if (!card) return;
    const container = card.closest(".components-details-container");
    if (container && container.style.display === "none") {
        container.style.display = "flex";
        const header = container.parentElement.querySelector(".components-header .toggle-hint");
        if (header) header.textContent = "Click to collapse ▲";
    }
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    card.style.borderColor = "var(--accent-cyan)";
    setTimeout(() => {
        card.style.borderColor = "";
    }, 1800);
}

function copySnippet(btnElem) {
    const pre = btnElem.closest(".req-pane, .res-pane").querySelector("pre code");
    if (!pre) return;
    navigator.clipboard.writeText(pre.textContent).then(() => {
        const originalText = btnElem.textContent;
        btnElem.textContent = "Copied!";
        setTimeout(() => { btnElem.textContent = originalText; }, 1800);
    }).catch(() => showToast("Failed to copy", "error"));
}

function copyModalSubField(subfield) {
    if (!activeModalPayload) return;
    const data = (activeModalPayload.payload && activeModalPayload.payload[subfield]) || activeModalPayload[subfield] || {};
    navigator.clipboard.writeText(JSON.stringify(data, null, 2))
        .then(() => showToast(`Copied ${subfield} payload to clipboard!`, "success"))
        .catch(() => showToast("Failed to copy", "error"));
}

function goToAuditLogs(convId) {
    switchTab("pageLogs");
    setTimeout(() => {
        selectConversation(convId);
    }, 200);
}

function copyModalJson(field) {
    if (!activeModalPayload) return;
    const jsonStr = JSON.stringify(activeModalPayload, null, 2);
    navigator.clipboard.writeText(jsonStr)
        .then(() => showToast("Detailed JSON copied to clipboard!", "success"))
        .catch(() => showToast("Failed to copy JSON", "error"));
}

function formatTimestamp(isoStr) {
    if (!isoStr) return "-";
    try {
        const d = new Date(isoStr);
        return d.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) + 
               " (" + d.toISOString().slice(0, 10) + ")";
    } catch {
        return isoStr;
    }
}

function truncate(str, maxLen = 50) {
    if (!str) return "";
    return str.length > maxLen ? str.slice(0, maxLen) + "..." : str;
}

async function confirmClearLogs() {
    if (!confirm("Are you sure you want to clear database/logs.json and database/conversations.json? All logged calls and conversations will be purged.")) {
        return;
    }
    try {
        const res = await fetch("/api/logs/clear", { method: "POST" });
        if (res.ok) {
            showToast("Audit logs and conversations cleared.", "info");
            selectedConversationId = null;
            loadLogsAndConversations();
        }
    } catch (err) {
        showToast("Failed to clear logs", "error");
    }
}

// ==========================================
// Utilities
// ==========================================
function formatMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italic
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    // Code blocks
    html = html.replace(/`([^`]+)`/g, '<code style="background: rgba(0,0,0,0.3); padding: 2px 5px; border-radius: 4px; font-family: monospace;">$1</code>');
    return html;
}

function escapeHtml(str) {
    if (typeof str !== "string") str = String(str);
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function showToast(message, type = "info") {
    const toast = document.getElementById("appToast");
    const msgElem = document.getElementById("toastMessage");
    const iconElem = document.getElementById("toastIcon");

    msgElem.textContent = message;
    if (type === "success") iconElem.textContent = "✅";
    else if (type === "error") iconElem.textContent = "❌";
    else iconElem.textContent = "ℹ️";

    toast.style.display = "flex";
    setTimeout(() => {
        toast.style.display = "none";
    }, 3500);
}

// ==========================================
// 6. Embedder Model Management & Warning Modal
// ==========================================
async function loadEmbedderModels() {
    try {
        const res = await fetch("/api/embedder/models");
        const data = await res.json();
        if (data.status !== "success") return;

        activeEmbedderModel = data.current_model || "all-minilm";
        availableEmbedders = data.models || [];

        // 1. Populate Dropdown in Document Source Configuration header
        const selectElem = document.getElementById("embedderSelect");
        if (selectElem) {
            selectElem.innerHTML = availableEmbedders.map(m => {
                const isSelected = m.id === activeEmbedderModel || activeEmbedderModel.startsWith(m.id + ":");
                const statusTag = m.installed ? " (Ready)" : " (Pull Needed)";
                return `<option value="${escapeHtml(m.id)}" ${isSelected ? "selected" : ""}>${escapeHtml(m.name)} • ${m.dimensions}d${statusTag}</option>`;
            }).join("");
        }

        // 2. Populate Reference Table at Bottom of Page 1
        const tableBody = document.getElementById("embeddersTableBody");
        if (tableBody) {
            tableBody.innerHTML = availableEmbedders.map(m => {
                const isCurrent = m.id === activeEmbedderModel || activeEmbedderModel.startsWith(m.id + ":");
                let statusBadge = "";
                if (isCurrent) {
                    statusBadge = '<span class="badge badge-success">✓ Active Model</span>';
                } else if (m.installed) {
                    statusBadge = '<span class="badge badge-info">Installed locally</span>';
                } else {
                    statusBadge = `<span class="badge badge-private" title="Will pull on demand or via 'ollama pull ${escapeHtml(m.id)}'">Available to Pull</span>`;
                }

                return `
                    <tr class="${isCurrent ? 'row-active-model' : ''}">
                        <td>
                            <div class="model-col">
                                <span class="model-name">${escapeHtml(m.name)}</span>
                                <span class="model-id">${escapeHtml(m.id)}</span>
                            </div>
                        </td>
                        <td><span class="dim-badge">${m.dimensions} dims</span></td>
                        <td>${escapeHtml(m.context_window)}</td>
                        <td>${escapeHtml(m.size)}</td>
                        <td class="use-case">
                            <div>${escapeHtml(m.description)}</div>
                            <small class="text-muted" style="display:block; margin-top:2px;"><strong>Best for:</strong> ${escapeHtml(m.recommended_for)}</small>
                        </td>
                        <td>${statusBadge}</td>
                    </tr>
                `;
            }).join("");
        }
    } catch (err) {
        console.error("Failed to load embedder models:", err);
    }
}

function handleEmbedderDropdownChange(event) {
    const selectedModel = event.target.value;
    if (selectedModel === activeEmbedderModel) return;

    pendingEmbedderModel = selectedModel;

    // Open Confirmation Warning Modal
    const modal = document.getElementById("modelChangeModal");
    const promptCode = document.getElementById("requiredConfirmPhrase");
    const input = document.getElementById("confirmModelChangeInput");
    const okBtn = document.getElementById("btnConfirmModelChange");
    const hint = document.getElementById("confirmMatchHint");

    const expectedPhrase = `Change to ${pendingEmbedderModel}`;
    promptCode.textContent = expectedPhrase;

    input.value = "";
    input.classList.remove("input-matched");
    okBtn.disabled = true;
    hint.textContent = "Exact text match required to enable the OK button.";
    hint.style.color = "var(--text-muted)";

    modal.style.display = "flex";
    setTimeout(() => input.focus(), 50);
}

function validateModelChangeInput() {
    const input = document.getElementById("confirmModelChangeInput");
    const okBtn = document.getElementById("btnConfirmModelChange");
    const hint = document.getElementById("confirmMatchHint");

    if (!pendingEmbedderModel) return;

    const expectedPhrase = `Change to ${pendingEmbedderModel}`;
    const currentValue = input.value.trim();

    if (currentValue === expectedPhrase) {
        okBtn.disabled = false;
        input.classList.add("input-matched");
        hint.textContent = "✓ Phrase matches! Click OK to switch model and purge vector storage.";
        hint.style.color = "var(--accent-emerald)";
    } else {
        okBtn.disabled = true;
        input.classList.remove("input-matched");
        hint.textContent = "Exact text match required to enable the OK button.";
        hint.style.color = "var(--text-muted)";
    }
}

function cancelModelChange() {
    const modal = document.getElementById("modelChangeModal");
    modal.style.display = "none";

    // Abort and restore dropdown to current model
    const selectElem = document.getElementById("embedderSelect");
    if (selectElem) {
        selectElem.value = activeEmbedderModel;
    }

    pendingEmbedderModel = null;
    const input = document.getElementById("confirmModelChangeInput");
    if (input) input.value = "";
}

function closeModelChangeModalOnBackdrop(event) {
    if (event.target.id === "modelChangeModal") {
        cancelModelChange();
    }
}

async function executeModelChange() {
    if (!pendingEmbedderModel) return;

    const okBtn = document.getElementById("btnConfirmModelChange");
    const cancelBtn = document.getElementById("btnCancelModelChange");
    const spinner = document.getElementById("modelChangeSpinner");
    const targetModel = pendingEmbedderModel;
    const confirmationPhrase = `Change to ${targetModel}`;

    okBtn.disabled = true;
    cancelBtn.disabled = true;
    spinner.style.display = "inline-block";

    try {
        const res = await fetch("/api/embedder/change", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                new_model: targetModel,
                confirmation_phrase: confirmationPhrase
            })
        });

        const data = await res.json();
        if (res.ok && data.status === "success") {
            activeEmbedderModel = targetModel;
            pendingEmbedderModel = null;

            // Close modal
            document.getElementById("modelChangeModal").style.display = "none";

            showToast(`Switched embedder to '${targetModel}'. Vector DB was purged.`, "success");

            // Refresh UI components
            await loadEmbedderModels();
            await loadStats();
            await checkHealth();
        } else {
            showToast(data.error || "Failed to change embedder model.", "error");
            cancelModelChange();
        }
    } catch (err) {
        showToast(`Network error: ${err.message}`, "error");
        cancelModelChange();
    } finally {
        spinner.style.display = "none";
        cancelBtn.disabled = false;
    }
}

// ==========================================
// 7. Google AI Studio Chat Model Selection & Custom LLM
// ==========================================
async function loadChatModels() {
    try {
        const res = await fetch("/api/models");
        const data = await res.json();
        if (data.status !== "success") return;

        availableChatModels = data.models || [];
        if (!selectedChatModel && data.default_model) {
            selectedChatModel = data.default_model;
        }

        const selectElem = document.getElementById("chatModelSelect");
        if (selectElem) {
            selectElem.innerHTML = availableChatModels.map(m => {
                const isSelected = (m.id === selectedChatModel) || (!selectedChatModel && m.is_default);
                if (isSelected) {
                    selectedChatModel = m.id;
                }
                return `<option value="${escapeHtml(m.id)}" ${isSelected ? "selected" : ""}>${escapeHtml(m.name)}</option>`;
            }).join("");
        }

        syncCustomEndpointVisibility();
        updateChatModelBadge();
    } catch (err) {
        console.error("Failed to load models:", err);
    }
}

function handleChatModelChange(event) {
    selectedChatModel = event.target.value;
    syncCustomEndpointVisibility();
    updateChatModelBadge();
    const modelName = getSelectedChatModelName();
    showToast(`Selected model: ${modelName}`, "info");
}

function syncCustomEndpointVisibility() {
    const box = document.getElementById("customEndpointBox");
    if (!box) return;
    if (selectedChatModel === "custom_model_api") {
        box.style.display = "flex";
        const input = document.getElementById("inputCustomEndpoint");
        if (input && !input.value) {
            input.value = "http://127.0.0.1:8000/v1/chat/completions";
        }
    } else {
        box.style.display = "none";
    }
}

function getSelectedChatModelName() {
    if (selectedChatModel === "custom_model_api") return "Custom Model API";
    const found = availableChatModels.find(m => m.id === selectedChatModel);
    if (found && found.name) return found.name;
    if (selectedChatModel) return selectedChatModel.replace("models/", "");
    return "AI Model";
}

function updateChatModelBadge() {
    const badge = document.getElementById("activeModelTag");
    if (!badge) return;
    if (selectedChatModel === "custom_model_api") {
        badge.textContent = "Custom Private LLM";
        badge.title = "Direct private LLM endpoint execution";
        badge.className = "badge badge-primary";
        return;
    }
    badge.className = "badge badge-success";
    const found = availableChatModels.find(m => m.id === selectedChatModel);
    if (found) {
        badge.textContent = found.name.length > 22 ? found.name.slice(0, 20) + "..." : found.name;
        badge.title = `${found.id} - ${found.description || 'Google AI Studio text model'}`;
    }
}

// ==========================================================================
// Application Shutdown Handlers
// ==========================================================================

function openShutdownModal() {
    const modal = document.getElementById("shutdownModal");
    if (modal) {
        modal.style.display = "flex";
        document.body.style.overflow = "hidden";
    }
}

function closeShutdownModal() {
    const modal = document.getElementById("shutdownModal");
    if (modal) {
        modal.style.display = "none";
        document.body.style.overflow = "";
    }
}

function closeShutdownModalOnBackdrop(e) {
    if (e.target && e.target.id === "shutdownModal") {
        closeShutdownModal();
    }
}

async function executeAppShutdown() {
    const btn = document.getElementById("btnConfirmShutdown");
    const cancelBtn = document.getElementById("btnCancelShutdown");
    const spinner = document.getElementById("shutdownSpinner");

    if (btn) btn.disabled = true;
    if (cancelBtn) cancelBtn.disabled = true;
    if (spinner) spinner.style.display = "inline-block";

    try {
        const resp = await fetch("/api/shutdown", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "shutdown" })
        });
        const data = await resp.json().catch(() => ({}));

        const modalBody = document.querySelector("#shutdownModal .modal-body");
        if (modalBody) {
            modalBody.innerHTML = `
                <div style="text-align: center; padding: 18px 8px;">
                    <div style="font-size: 44px; margin-bottom: 12px;">🛑</div>
                    <h3 style="color: #fda4af; margin-bottom: 8px; font-size: 18px;">Application Shut Down</h3>
                    <p style="color: var(--text-secondary); font-size: 14px; line-height: 1.6;">
                        ${escapeHtml(data.message || "Agent with RAG server is shutting down. The application has been disabled.")}
                    </p>
                    <p style="color: var(--text-muted); font-size: 12.5px; margin-top: 14px;">
                        The server process is terminating and is no longer accessible. You may now close this browser tab.
                    </p>
                </div>
            `;
        }
        const modalFooter = document.querySelector("#shutdownModal .modal-footer");
        if (modalFooter) {
            modalFooter.innerHTML = `<button type="button" class="btn btn-outline" onclick="closeShutdownModal()">Close Dialog</button>`;
        }

        // Update top status indicator to offline
        const statusLabel = document.getElementById("backendStatusLabel");
        const statusDot = document.querySelector("#backendStatusPill .status-dot");
        if (statusLabel) statusLabel.textContent = "Server Offline";
        if (statusDot) {
            statusDot.className = "status-dot error";
            statusDot.title = "Server has shut down";
        }
    } catch (err) {
        console.warn("Shutdown request dispatched, server may have already stopped:", err);
        const modalBody = document.querySelector("#shutdownModal .modal-body");
        if (modalBody) {
            modalBody.innerHTML = `
                <div style="text-align: center; padding: 18px 8px;">
                    <div style="font-size: 44px; margin-bottom: 12px;">🛑</div>
                    <h3 style="color: #fda4af; margin-bottom: 8px; font-size: 18px;">Application Shut Down</h3>
                    <p style="color: var(--text-secondary); font-size: 14px; line-height: 1.6;">
                        Agent with RAG server process has terminated and is now offline.
                    </p>
                    <p style="color: var(--text-muted); font-size: 12.5px; margin-top: 14px;">
                        You may now close this browser tab.
                    </p>
                </div>
            `;
        }
    }
}

// ==========================================
// 10. Telemetry & Performance Analytics
// ==========================================
let telemetryRequestsChart = null;
let telemetryTokensChart = null;
let telemetryIsLoading = false;

async function loadTelemetry(isManualRefresh = false) {
    if (telemetryIsLoading) return;
    telemetryIsLoading = true;

    const refreshBtn = document.getElementById("btnRefreshTelemetry");
    const refreshIcon = refreshBtn ? refreshBtn.querySelector(".btn-icon") : null;
    if (refreshIcon) {
        refreshIcon.style.display = "inline-block";
        refreshIcon.style.transition = "transform 0.5s ease";
        refreshIcon.style.transform = "rotate(360deg)";
    }

    try {
        const modelSelect = document.getElementById("telemetryModelSelect");
        const intervalSelect = document.getElementById("telemetryIntervalSelect");
        const timeRangeSelect = document.getElementById("telemetryTimeRangeSelect");
        const startDateInput = document.getElementById("telemetryStartDate");
        const endDateInput = document.getElementById("telemetryEndDate");

        const selectedModel = modelSelect ? modelSelect.value : "all";
        const selectedInterval = intervalSelect ? intervalSelect.value : "15m";
        const selectedTimeRange = timeRangeSelect ? timeRangeSelect.value : "1d";

        let url = `/api/telemetry?model=${encodeURIComponent(selectedModel)}&interval=${encodeURIComponent(selectedInterval)}&time_range=${encodeURIComponent(selectedTimeRange)}`;

        if (selectedTimeRange === "custom") {
            const startDate = startDateInput ? startDateInput.value : "";
            const endDate = endDateInput ? endDateInput.value : "";
            if (startDate) url += `&start_date=${encodeURIComponent(startDate)}`;
            if (endDate) url += `&end_date=${encodeURIComponent(endDate)}`;
        }

        const res = await fetch(url);
        if (!res.ok) {
            throw new Error(`Server returned HTTP ${res.status}`);
        }
        const json = await res.json();
        if (json.status !== "success") {
            throw new Error(json.message || "Failed to load telemetry data");
        }

        const data = json.data || json;

        // 1. Update Top 5 Statistics Metric Cards
        const totals = data.summary || data.totals || {};
        const totalPromptsEl = document.getElementById("statTotalPrompts");
        const totalResponsesEl = document.getElementById("statTotalResponses");
        const totalErrorsEl = document.getElementById("statTotalErrors");
        const totalInputTokensEl = document.getElementById("statTotalInputTokens");
        const totalOutputTokensEl = document.getElementById("statTotalOutputTokens");

        if (totalPromptsEl) totalPromptsEl.textContent = Number(totals.total_prompts || 0).toLocaleString();
        if (totalResponsesEl) totalResponsesEl.textContent = Number(totals.total_responses || 0).toLocaleString();
        if (totalErrorsEl) totalErrorsEl.textContent = Number(totals.total_errors || 0).toLocaleString();
        if (totalInputTokensEl) totalInputTokensEl.textContent = Number(totals.total_input_tokens || 0).toLocaleString();
        if (totalOutputTokensEl) totalOutputTokensEl.textContent = Number(totals.total_output_tokens || 0).toLocaleString();

        // 2. Populate Model Filter Dropdown
        const models = data.models || data.available_models || [];
        if (modelSelect && Array.isArray(models)) {
            const currentVal = modelSelect.value;
            const existingOptions = Array.from(modelSelect.options).map(o => o.value);
            const newOptions = ["all", ...models];
            const hasChanged = existingOptions.length !== newOptions.length || !newOptions.every(m => existingOptions.includes(m));

            if (hasChanged) {
                modelSelect.innerHTML = `<option value="all">All Models</option>`;
                models.forEach(m => {
                    const opt = document.createElement("option");
                    opt.value = m;
                    opt.textContent = m;
                    modelSelect.appendChild(opt);
                });
                if (newOptions.includes(currentVal)) {
                    modelSelect.value = currentVal;
                } else {
                    modelSelect.value = "all";
                }
            }
        }

        // 3. Update Interval Badges on Graph Cards
        const intervalLabels = {
            "1m": "1 min interval",
            "15m": "15 min interval",
            "1h": "1 hr interval",
            "1d": "1 day interval"
        };
        const currentInterval = (data.filters && data.filters.interval) || data.interval || "15m";
        const intervalText = intervalLabels[currentInterval] || `${currentInterval} interval`;
        const leftBadge = document.getElementById("leftPlotIntervalBadge");
        const rightBadge = document.getElementById("rightPlotIntervalBadge");
        if (leftBadge) leftBadge.textContent = intervalText;
        if (rightBadge) rightBadge.textContent = intervalText;

        // 4. Render Dual Graphs (Chart.js)
        renderTelemetryCharts(data.time_series || []);

        if (isManualRefresh) {
            showToast("Telemetry metrics refreshed successfully", "success");
        }
    } catch (err) {
        console.error("Error loading telemetry:", err);
        showToast("Failed to fetch telemetry data: " + err.message, "error");
    } finally {
        telemetryIsLoading = false;
        if (refreshIcon) {
            setTimeout(() => {
                refreshIcon.style.transform = "none";
            }, 500);
        }
    }
}

function renderTelemetryCharts(series) {
    if (typeof Chart === "undefined") {
        console.warn("Chart.js is not loaded, skipping chart rendering");
        return;
    }

    let labels = [];
    let promptsData = [];
    let responsesData = [];
    let errorsData = [];
    let inputTokensData = [];
    let outputTokensData = [];

    if (Array.isArray(series)) {
        labels = series.map(item => item.timestamp || item.label || "");
        promptsData = series.map(item => item.prompts || 0);
        responsesData = series.map(item => item.responses || 0);
        errorsData = series.map(item => item.errors || 0);
        inputTokensData = series.map(item => item.input_tokens || 0);
        outputTokensData = series.map(item => item.output_tokens || 0);
    } else if (series && typeof series === "object") {
        labels = series.labels || [];
        promptsData = series.prompts || [];
        responsesData = series.responses || [];
        errorsData = series.errors || [];
        inputTokensData = series.input_tokens || [];
        outputTokensData = series.output_tokens || [];
    }

    const commonScales = {
        x: {
            grid: {
                color: "rgba(255, 255, 255, 0.05)",
                drawBorder: false
            },
            ticks: {
                color: "#94a3b8",
                font: { size: 10.5, family: "'Inter', sans-serif" },
                maxRotation: 45,
                minRotation: 0,
                autoSkip: true,
                maxTicksLimit: 12
            }
        },
        y: {
            beginAtZero: true,
            grid: {
                color: "rgba(255, 255, 255, 0.05)",
                drawBorder: false
            },
            ticks: {
                color: "#94a3b8",
                font: { size: 10.5, family: "'Inter', sans-serif" },
                precision: 0
            }
        }
    };

    const commonPlugins = {
        legend: {
            position: "top",
            labels: {
                color: "#cbd5e1",
                font: { size: 12, family: "'Inter', sans-serif" },
                boxWidth: 14,
                boxHeight: 14,
                usePointStyle: true,
                pointStyle: "circle",
                padding: 16
            }
        },
        tooltip: {
            backgroundColor: "rgba(15, 21, 35, 0.95)",
            titleColor: "#f8fafc",
            bodyColor: "#cbd5e1",
            borderColor: "rgba(56, 189, 248, 0.3)",
            borderWidth: 1,
            padding: 10,
            cornerRadius: 8,
            titleFont: { size: 12, weight: "bold" },
            bodyFont: { size: 12 }
        }
    };

    // 1. Left Plot: Requests Activity (Prompts, Responses, Errors)
    const requestsCanvas = document.getElementById("telemetryRequestsChart");
    if (requestsCanvas) {
        if (telemetryRequestsChart) {
            telemetryRequestsChart.destroy();
        }
        const ctxReq = requestsCanvas.getContext("2d");
        telemetryRequestsChart = new Chart(ctxReq, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Prompts",
                        data: promptsData,
                        borderColor: "#6366f1",
                        backgroundColor: "rgba(99, 102, 241, 0.15)",
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 2,
                        pointHoverRadius: 5,
                        fill: false
                    },
                    {
                        label: "Responses",
                        data: responsesData,
                        borderColor: "#10b981",
                        backgroundColor: "rgba(16, 185, 129, 0.15)",
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 2,
                        pointHoverRadius: 5,
                        fill: false
                    },
                    {
                        label: "Errors",
                        data: errorsData,
                        borderColor: "#f43f5e",
                        backgroundColor: "rgba(244, 63, 94, 0.15)",
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 2,
                        pointHoverRadius: 5,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: "index",
                    intersect: false
                },
                scales: commonScales,
                plugins: commonPlugins
            }
        });
    }

    // 2. Right Plot: Token Throughput (Input & Output Tokens)
    const tokensCanvas = document.getElementById("telemetryTokensChart");
    if (tokensCanvas) {
        if (telemetryTokensChart) {
            telemetryTokensChart.destroy();
        }
        const ctxTok = tokensCanvas.getContext("2d");
        telemetryTokensChart = new Chart(ctxTok, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Input Tokens",
                        data: inputTokensData,
                        borderColor: "#06b6d4",
                        backgroundColor: "rgba(6, 182, 212, 0.15)",
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 2,
                        pointHoverRadius: 5,
                        fill: false
                    },
                    {
                        label: "Output Tokens",
                        data: outputTokensData,
                        borderColor: "#a855f7",
                        backgroundColor: "rgba(168, 85, 247, 0.15)",
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 2,
                        pointHoverRadius: 5,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: "index",
                    intersect: false
                },
                scales: commonScales,
                plugins: commonPlugins
            }
        });
    }
}

function handleTelemetryModelChange() {
    loadTelemetry();
}

function handleTelemetryIntervalChange() {
    loadTelemetry();
}

function handleTelemetryTimeRangeChange() {
    const rangeSelect = document.getElementById("telemetryTimeRangeSelect");
    const customBar = document.getElementById("telemetryCustomDates");
    if (!rangeSelect) return;

    if (rangeSelect.value === "custom") {
        if (customBar) customBar.style.display = "flex";
        const startInput = document.getElementById("telemetryStartDate");
        const endInput = document.getElementById("telemetryEndDate");
        if (startInput && !startInput.value) {
            const d = new Date();
            d.setDate(d.getDate() - 7);
            startInput.value = d.toISOString().split("T")[0];
        }
        if (endInput && !endInput.value) {
            endInput.value = new Date().toISOString().split("T")[0];
        }
        loadTelemetry();
    } else {
        if (customBar) customBar.style.display = "none";
        loadTelemetry();
    }
}

function applyCustomDates() {
    const startInput = document.getElementById("telemetryStartDate");
    const endInput = document.getElementById("telemetryEndDate");
    if (!startInput || !endInput || !startInput.value || !endInput.value) {
        showToast("Please select both starting and ending dates.", "warning");
        return;
    }
    if (startInput.value > endInput.value) {
        showToast("Starting date cannot be later than ending date.", "warning");
        return;
    }
    loadTelemetry();
}

function refreshTelemetry() {
    loadTelemetry(true);
}
