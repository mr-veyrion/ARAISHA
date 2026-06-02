// =========================================
// ARAISHA Professional UI - JavaScript
// =========================================

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

// =========================================
// Toast Notifications
// =========================================
function toast(msg, type = 'info') {
  const t = $("#toast");
  $("#toastMsg").textContent = msg;
  t.classList.remove('hidden');
  t.classList.add('show');
  setTimeout(() => {
    t.classList.remove('show');
    setTimeout(() => t.classList.add('hidden'), 400);
  }, 3000);
}

// =========================================
// API Helper
// =========================================
async function api(path, opts = {}) {
  try {
    const res = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
      ...opts
    });
    const ct = res.headers.get('content-type') || '';
    if (!res.ok) {
      const err = ct.includes('application/json') ? await res.json() : { error: await res.text() };
      throw new Error(err.error || res.statusText);
    }
    return ct.includes('application/json') ? await res.json() : await res.text();
  } catch (e) {
    toast(e.message || String(e), 'error');
    throw e;
  }
}

// =========================================
// Theme Management
// =========================================
function initTheme() {
  const theme = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-theme', theme);
  updateThemeIcon(theme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  updateThemeIcon(next);
}

function updateThemeIcon(theme) {
  const icon = $("#themeIcon");
  if (!icon) return;

  if (theme === 'light') {
    // Moon icon for light mode (to switch to dark)
    icon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>';
  } else {
    // Sun icon for dark mode (to switch to light)
    icon.innerHTML = '<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>';
  }
}

// =========================================
// Collapsible Panels
// =========================================
function wirePanels() {
  const panels = $$(".panel.collapsible");
  panels.forEach(panel => {
    const header = panel.querySelector('.panel-header.clickable');
    if (header) {
      header.addEventListener('click', () => {
        // Accordion: close others
        panels.forEach(p => { if (p !== panel) p.classList.remove('open'); });
        panel.classList.toggle('open');
      });
    }
  });
}

// =========================================
// Sidebar Collapse
// =========================================
function wireCollapse() {
  const btn = $("#collapseBtn");
  if (btn) {
    btn.addEventListener('click', () => {
      $("#sidebar").classList.toggle('collapsed');
      $(".app").classList.toggle('sidebar-collapsed');
    });
  }
}

// =========================================
// Model Selection
// =========================================
function updateModelFields() {
  const backend = $("#backendSelect").value;
  const apiUrlGroup = $("#apiUrlGroup");
  const apiKeyGroup = $("#apiKeyGroup");
  const modelPath = $("#modelPath");

  if (backend === 'api') {
    // Remote API (OpenAI, NVIDIA, etc.)
    apiUrlGroup.style.display = 'block';
    apiKeyGroup.style.display = 'block';
    modelPath.placeholder = 'Model name (e.g., gpt-4, meta/llama-3.1-8b)';
    $("#apiUrl").placeholder = 'https://api.openai.com/v1';
  } else if (backend === 'local_api') {
    // Local API (LM Studio, Ollama - no key needed)
    apiUrlGroup.style.display = 'block';
    apiKeyGroup.style.display = 'none';
    modelPath.placeholder = 'Model name (e.g., llama-3.1-8b)';
    $("#apiUrl").placeholder = 'http://localhost:1234/v1';
  } else {
    // Local models (GGUF, Transformers)
    apiUrlGroup.style.display = 'none';
    apiKeyGroup.style.display = 'none';
    modelPath.placeholder = 'Local model path';
  }
}

async function applyModel() {
  const backend = $("#backendSelect").value;
  const llm_model = $("#modelPath").value.trim() || null;
  const api_url = $("#apiUrl").value.trim() || null;
  const api_key = $("#apiKey").value.trim() || null;

  try {
    await api('/api/model/select', {
      method: 'POST',
      body: JSON.stringify({ backend, llm_model, api_url, api_key })
    });
    toast('Model applied successfully!', 'success');
  } catch (e) {
    console.error('Failed to apply model:', e);
  }
}

// =========================================
// Emotions Chart
// =========================================
function drawEmotionsChart(canvas, data, legendContainer) {
  if (!canvas || !legendContainer) return;

  const ctx = canvas.getContext('2d');
  const entries = Object.entries(data || {})
    .filter(([_, value]) => value > 0)
    .sort(([_, a], [__, b]) => b - a)
    .slice(0, 8);

  if (entries.length === 0) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    legendContainer.innerHTML = '<div style="text-align: center; color: var(--text-muted); font-size: 12px; padding: 20px;">No emotions detected yet</div>';
    return;
  }

  const labels = entries.map(([label]) => label);
  const values = entries.map(([_, value]) => value);
  const total = values.reduce((a, b) => a + b, 0) || 1;

  // Modern color palette
  const colors = [
    '#6366f1', '#8b5cf6', '#a855f7', '#ec4899',
    '#f43f5e', '#f97316', '#eab308', '#22c55e'
  ];

  // Canvas setup for high DPI
  const dpr = window.devicePixelRatio || 1;
  const size = 160;
  canvas.width = size * dpr;
  canvas.height = size * dpr;
  canvas.style.width = size + 'px';
  canvas.style.height = size + 'px';
  ctx.scale(dpr, dpr);

  const cx = size / 2;
  const cy = size / 2;
  const r = (size / 2) - 10;

  let start = -Math.PI / 2;
  ctx.clearRect(0, 0, size, size);

  // Draw pie slices
  values.forEach((value, i) => {
    const angle = (value / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, start, start + angle);
    ctx.fillStyle = colors[i % colors.length];
    ctx.fill();
    start += angle;
  });

  // Center circle for donut effect
  ctx.beginPath();
  ctx.arc(cx, cy, r * 0.55, 0, Math.PI * 2);
  ctx.fillStyle = '#12121a';
  ctx.fill();

  // Update legend
  legendContainer.innerHTML = entries.map(([label, value], i) => {
    const percent = ((value / total) * 100).toFixed(0);
    const color = colors[i % colors.length];
    return `
      <div class="emotion-item">
        <div class="emotion-color" style="background-color: ${color}"></div>
        <div class="emotion-label">${label}</div>
        <div class="emotion-percent">${percent}%</div>
      </div>
    `;
  }).join('');
}

async function refreshEmotions() {
  try {
    const data = await api('/api/emotions');
    drawEmotionsChart($("#emotionsChart"), data.percentages || {}, $("#emotionsLegend"));
  } catch (e) {
    console.error('Failed to refresh emotions:', e);
  }
}

async function refreshStyles() {
  try {
    const data = await api('/api/styles');
    let text = data.text || 'No style patterns detected yet.';
    // Clean up display
    const lines = text.split('\n').filter(line => !line.startsWith('Emotional Palette'));
    $("#styleText").textContent = lines.join('\n') || 'No style patterns detected yet.';
  } catch (e) {
    console.error('Failed to refresh styles:', e);
  }
}

// =========================================
// Chat Management
// =========================================
async function refreshChats() {
  try {
    const data = await api('/api/chats');
    const ul = $("#chatsList");
    ul.innerHTML = '';

    (data.sessions || []).forEach(s => {
      const li = document.createElement('li');
      li.className = 'chat-item';

      const name = document.createElement('span');
      name.className = 'name';
      name.textContent = s.id;
      if (s.id === data.active) {
        name.style.color = 'var(--accent-secondary)';
        name.style.fontWeight = '600';
      }

      const actions = document.createElement('div');
      actions.className = 'actions';

      const btnReuse = document.createElement('button');
      btnReuse.className = 'btn';
      btnReuse.innerHTML = '↻';
      btnReuse.title = 'Switch to this chat';

      const btnRen = document.createElement('button');
      btnRen.className = 'btn';
      btnRen.innerHTML = '✎';
      btnRen.title = 'Rename chat';

      const btnDel = document.createElement('button');
      btnDel.className = 'btn';
      btnDel.innerHTML = '✕';
      btnDel.title = 'Delete chat';

      btnReuse.onclick = async () => {
        try {
          await api(`/api/chats/${encodeURIComponent(s.id)}/reuse`, { method: 'POST' });
          await refreshChats();
          clearMessages();
          toast('Switched to chat: ' + s.id);
        } catch (e) {
          console.error('Failed to switch chat:', e);
        }
      };

      btnRen.onclick = async () => {
        const nid = prompt('New chat name:', s.id);
        if (!nid || nid === s.id) return;
        try {
          await api(`/api/chats/${encodeURIComponent(s.id)}/rename`, {
            method: 'POST',
            body: JSON.stringify({ new_id: nid })
          });
          await refreshChats();
          toast('Chat renamed to: ' + nid);
        } catch (e) {
          console.error('Failed to rename chat:', e);
        }
      };

      btnDel.onclick = async () => {
        if (!confirm(`Delete chat "${s.id}"?`)) return;
        try {
          await api(`/api/chats/${encodeURIComponent(s.id)}`, { method: 'DELETE' });
          await refreshChats();
          toast('Chat deleted');
        } catch (e) {
          console.error('Failed to delete chat:', e);
        }
      };

      actions.append(btnReuse, btnRen, btnDel);
      li.append(name, actions);
      ul.append(li);
    });
  } catch (e) {
    console.error('Failed to refresh chats:', e);
  }
}

// =========================================
// Messages
// =========================================
function addMessage(role, text) {
  // Remove welcome card if present
  const welcome = $(".welcome-card");
  if (welcome) welcome.remove();

  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  wrap.textContent = text;
  $("#messages").append(wrap);
  wrap.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function clearMessages() {
  const messages = $("#messages");
  messages.innerHTML = `
    <div class="welcome-card">
      <div class="welcome-icon">🧠</div>
      <h2>Welcome to ARAISHA</h2>
      <p>Your intelligent memory companion. Store memories, explore relationships, and let AI understand your context.</p>
      <div class="welcome-tips">
        <div class="tip">
          <span class="tip-icon">💾</span>
          <span><strong>@store</strong> — Save a new memory</span>
        </div>
        <div class="tip">
          <span class="tip-icon">🔍</span>
          <span><strong>@remember</strong> — Search memories</span>
        </div>
        <div class="tip">
          <span class="tip-icon">🗺️</span>
          <span><strong>@viz</strong> — Visualize knowledge graph</span>
        </div>
      </div>
    </div>
  `;
}

// =========================================
// Trigger Commands
// =========================================
const TRIGGERS = [
  { name: '@store', desc: 'Store a new memory' },
  { name: '@remember', desc: 'Search and recall memories' },
  { name: '@update', desc: 'Update a memory by ID' },
  { name: '@updateq', desc: 'Update memories by query' },
  { name: '@delete', desc: 'Delete memories' },
  { name: '@query', desc: 'Run DSL query on graph' },
  { name: '@path', desc: 'Find path between entities' },
  { name: '@viz', desc: 'Visualize knowledge graph' },
  { name: '@merge', desc: 'Merge entities or relationships' },
  { name: '@rebuild', desc: 'Rebuild vector index' }
];

function showTriggerSuggestions(filterText = '') {
  const suggestions = $("#triggerSuggestions");
  const filtered = TRIGGERS.filter(t =>
    filterText === '' || t.name.toLowerCase().includes(filterText.toLowerCase())
  );

  if (filtered.length === 0) {
    suggestions.classList.remove('show');
    return;
  }

  suggestions.innerHTML = filtered.map(trigger => `
    <div class="trigger-item" data-trigger="${trigger.name}">
      <div class="trigger-name">${trigger.name}</div>
      <div class="trigger-desc">${trigger.desc}</div>
    </div>
  `).join('');

  suggestions.classList.add('show');

  // Click handlers
  suggestions.querySelectorAll('.trigger-item').forEach(item => {
    item.addEventListener('click', () => {
      const trigger = item.dataset.trigger;
      const input = $("#input");
      input.value = trigger + ' ';
      input.focus();
      hideTriggerSuggestions();
    });
  });
}

function hideTriggerSuggestions() {
  $("#triggerSuggestions").classList.remove('show');
}

// =========================================
// Send Message
// =========================================
async function sendMessage() {
  const ta = $("#input");
  const msg = ta.value.trim();
  if (!msg) return;

  ta.value = '';
  autoResizeTextArea();
  addMessage('user', msg);

  // Check for trigger commands
  const triggers = ['@store', '@remember', '@update', '@updateq', '@delete', '@query', '@path', '@viz', '@merge', '@rebuild', '@eval'];
  const isTrigger = triggers.some(t => msg.toLowerCase().startsWith(t.toLowerCase()));

  if (isTrigger) {
    // Handle trigger commands
    try {
      if (msg.toLowerCase().startsWith('@remember')) {
        // Streaming response for @remember
        const res = await fetch('/api/trigger', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: msg })
        });

        if (!res.ok) {
          const err = await res.json();
          addMessage('assistant', `❌ Error: ${err.error || res.statusText}`);
          return;
        }

        const reader = res.body.getReader();
        let assistant = '';
        const node = document.createElement('div');
        node.className = 'msg assistant';
        node.textContent = '';
        // Remove welcome card if present
        const welcome = $(".welcome-card");
        if (welcome) welcome.remove();
        $("#messages").append(node);

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = new TextDecoder().decode(value);
          assistant += chunk;
          node.textContent = assistant;
        }
        node.scrollIntoView({ behavior: 'smooth', block: 'end' });
      } else {
        // JSON response for other triggers
        const result = await api('/api/trigger', {
          method: 'POST',
          body: JSON.stringify({ message: msg })
        });

        let displayText = formatTriggerResult(result);
        addMessage('assistant', displayText);
      }

      // Refresh data after state-modifying triggers
      if (msg.toLowerCase().startsWith('@store') || msg.toLowerCase().startsWith('@remember')) {
        setTimeout(() => {
          refreshEmotions();
          refreshStyles();
          refreshChats();
        }, 500);
      }
    } catch (e) {
      addMessage('assistant', `❌ Error: ${e.message}`);
    }
  } else {
    // Normal chat - streaming response
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, stream: true })
      });

      if (!res.ok) {
        toast((await res.json()).error || res.statusText, 'error');
        return;
      }

      const reader = res.body.getReader();
      let assistant = '';
      const node = document.createElement('div');
      node.className = 'msg assistant';
      node.textContent = '';
      // Remove welcome card if present
      const welcome = $(".welcome-card");
      if (welcome) welcome.remove();
      $("#messages").append(node);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = new TextDecoder().decode(value);
        assistant += chunk;
        node.textContent = assistant;
      }
      node.scrollIntoView({ behavior: 'smooth', block: 'end' });
    } catch (e) {
      addMessage('assistant', `❌ Error: ${e.message}`);
    }
  }
}

function formatTriggerResult(result) {
  if (result.stored) {
    // Friendly confirmation message for @store
    let text = `✅ I have received your memory.`;

    // Show relationships (the main storage info)
    if (result.relationships && result.relationships.length > 0) {
      text += `\n\n📝 Storage Info:\n${result.relationships.join('\n')}`;
    }

    // Show merge info if any
    if (result.merges && (result.merges.entities > 0 || result.merges.relationships > 0)) {
      text += `\n\n🔄 Merged: ${result.merges.entities} entities, ${result.merges.relationships} relationships`;
    }

    return text;
  } else if (result.query_all || result.query) {
    const data = result.query_all || result.query;
    return `📊 Query Results:\n${JSON.stringify(data, null, 2)}`;
  } else if (result.path_all || result.path_nodes) {
    const data = result.path_all || { nodes: result.path_nodes, edges: result.path_edges };
    return `🛤️ Path Results:\n${JSON.stringify(data, null, 2)}`;
  } else if (result.viz_all || result.viz) {
    let text = `📈 Visualization: ${result.viz_all || result.viz}`;
    if (result.url) {
      text += `\n\n🔗 Open: ${result.url}`;
      window.open(result.url, '_blank');
    }
    return text;
  } else if (result.entity_merged || result.relationship_merged) {
    return `🔗 ${result.entity_merged || result.relationship_merged}\n${JSON.stringify(result.stats, null, 2)}`;
  } else if (result.rebuild) {
    return `🔧 ${result.rebuild}`;
  }
  return JSON.stringify(result, null, 2);
}

// =========================================
// Input Handling
// =========================================
function autoResizeTextArea() {
  const ta = $("#input");
  ta.style.height = 'auto';
  ta.style.height = Math.min(200, ta.scrollHeight) + 'px';
}

function wireInput() {
  const input = $("#input");

  // Send button
  $("#send").addEventListener('click', sendMessage);

  // Enter to send
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      hideTriggerSuggestions();
      sendMessage();
    }
    if (e.key === 'Escape') {
      hideTriggerSuggestions();
    }
  });

  // Auto-resize and trigger suggestions
  input.addEventListener('input', () => {
    autoResizeTextArea();

    const text = input.value;
    const cursorPos = input.selectionStart;
    const beforeCursor = text.substring(0, cursorPos);
    const atMatch = beforeCursor.match(/@(\w*)$/);

    if (atMatch) {
      showTriggerSuggestions(atMatch[1]);
    } else {
      hideTriggerSuggestions();
    }
  });

  // Hide suggestions on blur
  input.addEventListener('blur', () => {
    setTimeout(() => hideTriggerSuggestions(), 200);
  });
}

// =========================================
// Wire Model Selection
// =========================================
function wireModel() {
  const backendSelect = $("#backendSelect");
  const applyBtn = $("#applyModel");

  if (backendSelect) {
    backendSelect.addEventListener('change', updateModelFields);
    updateModelFields();
  }

  if (applyBtn) {
    applyBtn.addEventListener('click', applyModel);
  }
}

// =========================================
// Wire New Chat Button
// =========================================
function wireNewChat() {
  const btn = $("#newChatBtn");
  if (btn) {
    btn.addEventListener('click', async () => {
      try {
        await api('/api/chats', { method: 'POST', body: JSON.stringify({}) });
        await refreshChats();
        clearMessages();
        toast('New chat created');
      } catch (e) {
        console.error('Failed to create new chat:', e);
      }
    });
  }
}

// =========================================
// Wire Theme Button
// =========================================
function wireTheme() {
  const btn = $("#themeBtn");
  if (btn) {
    btn.addEventListener('click', toggleTheme);
  }
}

// =========================================
// Initialize
// =========================================
async function init() {
  initTheme();
  wireTheme();
  wirePanels();
  wireCollapse();
  wireInput();
  wireModel();
  wireNewChat();

  // Initial data load
  await Promise.all([
    refreshEmotions(),
    refreshStyles(),
    refreshChats()
  ]);

  // Auto-refresh intervals
  setInterval(refreshEmotions, 5000);
  setInterval(refreshStyles, 8000);
}

// Start app when DOM is ready
window.addEventListener('DOMContentLoaded', init);
