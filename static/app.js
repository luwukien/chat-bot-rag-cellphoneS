// ================================================
// CellphoneS AI — Frontend Logic (Minimalist + Debug)
// ================================================

const $ = (sel) => document.querySelector(sel);
const chatBody   = $('#chatBody');
const userInput   = $('#userInput');
const chatForm    = $('#chatForm');
const sendBtn     = $('#sendBtn');
const typingBar   = $('#typingIndicator');
const clearBtn    = $('#clearChatBtn');
const debugToggle = $('#debugToggleBtn');
const debugPanel  = $('#debugPanel');
const debugBody   = $('#debugBody');
const debugClose  = $('#debugCloseBtn');

let debugEnabled = false;

// ---- Utilities ----

function time() {
    return new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
}

function esc(text) {
    const el = document.createElement('span');
    el.textContent = text;
    return el.innerHTML;
}

function formatMarkdown(text) {
    if (!text) return '';
    let s = esc(text);
    s = s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/__(.*?)__/g, '<strong>$1</strong>');
    s = s.replace(/\*(.*?)\*/g, '<em>$1</em>');
    s = s.replace(/^\s*[\-\*]\s+(.*)$/gm, '• $1');
    s = s.replace(/\n/g, '<br>');
    return s;
}

function scrollDown() {
    chatBody.scrollTop = chatBody.scrollHeight;
}

// ---- Textarea auto-expand ----

userInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
});

userInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.requestSubmit();
    }
});

// ---- Debug Panel ----

debugToggle.addEventListener('click', () => {
    debugEnabled = !debugEnabled;
    debugPanel.classList.toggle('hidden', !debugEnabled);
    debugToggle.style.color = debugEnabled ? 'var(--accent)' : '';
});

debugClose.addEventListener('click', () => {
    debugEnabled = false;
    debugPanel.classList.add('hidden');
    debugToggle.style.color = '';
});

function addDebugEntry(query, data, elapsedMs) {
    if (!debugEnabled) return;

    // Remove empty state
    const empty = debugBody.querySelector('.debug-empty');
    if (empty) empty.remove();

    const entry = document.createElement('div');
    entry.className = 'debug-entry fade-in';

    const sources = (data.sources || []).map((s, i) =>
        `  [${i}] ${s.type} | score=${s.score} | ${s.product_name || '—'}\n      ${s.text.substring(0, 100)}...`
    ).join('\n');

    entry.innerHTML = `
        <div class="debug-entry-header">
            <span class="label">CHAT</span>
            <span class="timestamp">${time()} · ${elapsedMs}ms</span>
        </div>
        <div class="debug-section">
            <div class="debug-section-title">Request</div>
            <div class="debug-kv"><span class="key">user_msg:</span><span class="val">"${esc(query)}"</span></div>
        </div>
        <div class="debug-section">
            <div class="debug-section-title">Pipeline</div>
            <div class="debug-kv"><span class="key">cleaned:</span><span class="val">"${esc(data.cleaned_query || query)}"</span></div>
            <div class="debug-kv"><span class="key">sub_queries:</span><span class="val">[${(data.sub_queries || []).map(q => '"' + esc(q) + '"').join(', ')}]</span></div>
            <div class="debug-kv"><span class="key">sources_n:</span><span class="val--num">${(data.sources || []).length}</span></div>
            <div class="debug-kv"><span class="key">latency:</span><span class="val--num">${elapsedMs}ms</span></div>
        </div>
        <div class="debug-section">
            <div class="debug-section-title">Sources (top ${(data.sources || []).length})</div>
            <pre>${sources || '(none)'}</pre>
        </div>
        <div class="debug-section">
            <div class="debug-section-title">Answer (raw, first 300 chars)</div>
            <pre>${esc((data.answer || '').substring(0, 300))}${(data.answer || '').length > 300 ? '…' : ''}</pre>
        </div>
    `;

    debugBody.prepend(entry);  // newest on top
}

// ---- Clear Chat ----

clearBtn.addEventListener('click', () => {
    if (!confirm('Xóa tất cả tin nhắn?')) return;
    chatBody.innerHTML = `
        <div class="msg msg--bot fade-in">
            <div class="msg-bubble">Đã xóa. Bạn muốn hỏi gì tiếp?</div>
        </div>
    `;
});

// ---- Suggestion Chips ----

function sendSuggestion(text) {
    userInput.value = text;
    chatForm.requestSubmit();
}
// expose globally for onclick
window.sendSuggestion = sendSuggestion;

// ---- Append Messages ----

function appendUser(text) {
    const div = document.createElement('div');
    div.className = 'msg msg--user fade-in';
    div.innerHTML = `
        <div class="msg-bubble">${esc(text)}</div>
        <span class="msg-time">${time()}</span>
    `;
    chatBody.appendChild(div);
    scrollDown();
}

function appendBot(answer, sources = []) {
    const div = document.createElement('div');
    div.className = 'msg msg--bot fade-in';

    let sourcesHtml = '';
    if (sources.length > 0) {
        const cards = sources.map(s => {
            const badge = `badge-${s.type.toLowerCase()}`;
            const name = s.product_name ? `<strong>${esc(s.product_name)}</strong> · ` : '';
            return `
                <div class="source-card">
                    <span class="source-badge ${badge}">${esc(s.type)}</span>
                    ${name}<span style="color:var(--text-secondary)">score ${s.score}</span>
                    <div class="source-text">${esc(s.text)}</div>
                </div>
            `;
        }).join('');

        sourcesHtml = `
            <div class="sources-section">
                <button class="sources-toggle" onclick="toggleSrc(this)">
                    <span>📄 ${sources.length} nguồn tham khảo</span>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="6 9 12 15 18 9"/></svg>
                </button>
                <div class="sources-list hidden">${cards}</div>
            </div>
        `;
    }

    div.innerHTML = `
        <div class="msg-bubble">${formatMarkdown(answer)}</div>
        ${sourcesHtml}
        <span class="msg-time">${time()}</span>
    `;
    chatBody.appendChild(div);
    scrollDown();
}

function appendError(text) {
    const div = document.createElement('div');
    div.className = 'msg msg--bot fade-in';
    div.innerHTML = `<div class="msg-error">⚠️ ${esc(text)}</div>`;
    chatBody.appendChild(div);
    scrollDown();
}

// ---- Toggle Sources ----

function toggleSrc(btn) {
    const list = btn.nextElementSibling;
    const isHidden = list.classList.contains('hidden');
    list.classList.toggle('hidden');
    btn.classList.toggle('open', isHidden);
}
window.toggleSrc = toggleSrc;

// ---- Typing Indicator ----

function showTyping() { typingBar.classList.remove('hidden'); scrollDown(); }
function hideTyping() { typingBar.classList.add('hidden'); }

// ---- Main Submit ----

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const msg = userInput.value.trim();
    if (!msg) return;

    appendUser(msg);
    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;
    showTyping();

    const t0 = performance.now();

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg })
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();
        const elapsed = Math.round(performance.now() - t0);

        hideTyping();
        appendBot(data.answer, data.sources);
        addDebugEntry(msg, data, elapsed);

    } catch (err) {
        console.error('API Error:', err);
        hideTyping();
        appendError('Không kết nối được tới server. Kiểm tra lại API và thử lại.');
    } finally {
        sendBtn.disabled = false;
        userInput.focus();
    }
});
