/**
 * Private RAG Frontend Application Logic
 */

// Application state
let currentLogs = [];
let activeModalPayload = null;

document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    loadStats();
    loadLogs();
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

            // Populate Sources List
            const listElem = document.getElementById("indexedSourcesList");
            const sources = data.vector_store.sources_list || [];
            if (sources.length === 0) {
                listElem.innerHTML = `<p class="text-muted" style="padding: 10px 0;">No documents indexed yet.</p>`;
            } else {
                listElem.innerHTML = sources.map(s => `
                    <div class="source-item">
                        <span class="source-title" title="${escapeHtml(s)}">${escapeHtml(s)}</span>
                        <span class="badge badge-private">Stored in DB</span>
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
    const assistantMsgElem = appendMessage("assistant", "Searching private vector database and generating answer with Gemma...", true);

    submitBtn.disabled = true;
    spinner.style.display = "inline-block";

    const startTime = Date.now();

    try {
        const res = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: question, top_k: parseInt(topK, 10) })
        });

        const data = await res.json();
        const duration = ((Date.now() - startTime) / 1000).toFixed(1);

        if (res.ok) {
            updateAssistantMessage(
                assistantMsgElem, 
                data.answer, 
                data.model ? `Model: ${data.model.replace("models/", "")} • ${duration}s • Log ID: ${data.log_id ? data.log_id.slice(0, 8) : 'N/A'}` : null
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

function updateAssistantMessage(msgElem, text, metaText) {
    const bubble = msgElem.querySelector(".chat-bubble");
    bubble.classList.remove("loading-pulse");
    bubble.innerHTML = formatMarkdown(text);

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
// 5. Page 3: Audit Logs & Payload Inspector
// ==========================================
async function loadLogs() {
    try {
        const res = await fetch("/api/logs?limit=100");
        const data = await res.json();
        currentLogs = data.logs || [];
        filterLogs();
        loadStats();
    } catch (err) {
        console.error("Failed to load audit logs:", err);
    }
}

function filterLogs() {
    const typeFilter = document.getElementById("filterCallType").value;
    const searchFilter = document.getElementById("logSearchInput").value.toLowerCase().trim();
    const tbody = document.getElementById("logsTableBody");

    let filtered = currentLogs;
    if (typeFilter) {
        filtered = filtered.filter(l => l.type === typeFilter);
    }
    if (searchFilter) {
        filtered = filtered.filter(l => {
            const argStr = JSON.stringify(l.arguments || {}).toLowerCase();
            const respStr = JSON.stringify(l.response || {}).toLowerCase();
            const typeStr = (l.type || "").toLowerCase();
            return argStr.includes(searchFilter) || respStr.includes(searchFilter) || typeStr.includes(searchFilter);
        });
    }

    if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted" style="padding: 24px;">No log entries match your filter.</td></tr>`;
        return;
    }

    tbody.innerHTML = filtered.map(log => {
        const timeFormatted = new Date(log.time).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) + 
                              " (" + new Date(log.time).toISOString().slice(0, 10) + ")";
        
        let badgeClass = "badge-private";
        let typeDisplay = log.type;
        if (log.type === "google_ai_studio_gemma") {
            badgeClass = "badge-info";
            typeDisplay = "Google AI (Gemma)";
        } else if (log.type.startsWith("local_ollama")) {
            badgeClass = "badge-private";
            typeDisplay = log.type.includes("batch") ? "Ollama Batch Embed" : "Ollama Single Embed";
        } else if (log.type === "external_url_fetch") {
            badgeClass = "badge-success";
            typeDisplay = "Web URL Fetch";
        }

        const statusBadge = log.status === "success" 
            ? `<span class="badge badge-success">200 OK</span>` 
            : `<span class="badge badge-error">Error</span>`;

        const duration = log.duration_ms ? `${log.duration_ms} ms` : "-";

        // Generate concise summary of arguments
        let argSummary = "";
        if (log.arguments) {
            if (log.arguments.question) argSummary = `Q: "${log.arguments.question.slice(0, 40)}..."`;
            else if (log.arguments.url) argSummary = `URL: ${log.arguments.url.slice(0, 40)}`;
            else if (log.arguments.texts_count) argSummary = `Batch of ${log.arguments.texts_count} chunks`;
            else if (log.arguments.text_sample) argSummary = `Text: "${log.arguments.text_sample.slice(0, 35)}..."`;
            else argSummary = JSON.stringify(log.arguments).slice(0, 40);
        }

        return `
            <tr>
                <td class="mono-cell">${escapeHtml(timeFormatted)}</td>
                <td><span class="badge ${badgeClass}">${escapeHtml(typeDisplay)}</span></td>
                <td>${statusBadge}</td>
                <td class="mono-cell">${duration}</td>
                <td title="${escapeHtml(JSON.stringify(log.arguments))}">${escapeHtml(argSummary)}</td>
                <td>
                    <button class="btn btn-xs btn-outline" onclick="openPayloadModal('${log.id}')">Inspect</button>
                </td>
            </tr>
        `;
    }).join("");
}

function openPayloadModal(logId) {
    const log = currentLogs.find(l => l.id === logId);
    if (!log) return;

    activeModalPayload = log;

    document.getElementById("modalTitle").textContent = `Call Audit: ${log.type}`;
    document.getElementById("modalTypeBadge").textContent = log.status.toUpperCase();
    document.getElementById("modalTypeBadge").className = `badge ${log.status === 'success' ? 'badge-success' : 'badge-error'}`;

    document.getElementById("modalMetaBar").innerHTML = `
        <span><strong>Time:</strong> ${log.time}</span> | 
        <span><strong>Duration:</strong> ${log.duration_ms || 0} ms</span> | 
        <span><strong>ID:</strong> ${log.id}</span>
    `;

    document.getElementById("modalArgsJson").textContent = JSON.stringify(log.arguments, null, 2);
    document.getElementById("modalResponseJson").textContent = JSON.stringify(log.response, null, 2);

    document.getElementById("payloadModal").style.display = "flex";
}

function closeModal() {
    document.getElementById("payloadModal").style.display = "none";
    activeModalPayload = null;
}

function closeModalOnBackdrop(event) {
    if (event.target.id === "payloadModal") {
        closeModal();
    }
}

function copyModalJson(field) {
    if (!activeModalPayload) return;
    const data = field === "args" ? activeModalPayload.arguments : activeModalPayload.response;
    navigator.clipboard.writeText(JSON.stringify(data, null, 2))
        .then(() => showToast("Copied to clipboard!", "success"))
        .catch(() => showToast("Failed to copy", "error"));
}

async function confirmClearLogs() {
    if (!confirm("Are you sure you want to clear database/logs.json? All logged calls will be purged.")) {
        return;
    }
    try {
        const res = await fetch("/api/logs/clear", { method: "POST" });
        if (res.ok) {
            showToast("Audit logs cleared successfully.", "info");
            loadLogs();
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
