const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("status");
const sourcesEl = document.getElementById("sources");

let ws;
let currentAssistantBubble = null;

function addBubble(role, text) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function setStatus(t) {
  statusEl.textContent = t;
}

function renderSources(sources) {
  sourcesEl.innerHTML = "";
  if (!sources || sources.length === 0) {
    sourcesEl.textContent = "(no sources)";
    return;
  }
  for (const s of sources) {
    const card = document.createElement("div");
    card.className = "source";
    card.innerHTML = `
      <div class="source-title">${s.id}</div>
      <div class="source-preview"></div>
      <pre class="source-meta"></pre>
    `;
    card.querySelector(".source-preview").textContent = s.text_preview || "";
    card.querySelector(".source-meta").textContent = JSON.stringify(
      s.properties || {},
      null,
      2
    );
    sourcesEl.appendChild(card);
  }
}

function connect() {
  // IMPORTANT: Use wss:// when the page is served over https://
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/chat`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => setStatus("Connected");
  ws.onclose = () => setStatus("Disconnected");
  ws.onerror = () => setStatus("Error");

  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);

    if (msg.type === "assistant_start") {
      currentAssistantBubble = addBubble("assistant", "");
    } else if (msg.type === "assistant_delta") {
      if (!currentAssistantBubble) currentAssistantBubble = addBubble("assistant", "");
      currentAssistantBubble.textContent += msg.delta;
      chatEl.scrollTop = chatEl.scrollHeight;
    } else if (msg.type === "assistant_done") {
      if (!currentAssistantBubble) currentAssistantBubble = addBubble("assistant", msg.text || "");
      currentAssistantBubble.textContent = msg.text || currentAssistantBubble.textContent;
      currentAssistantBubble = null;
      renderSources(msg.sources || []);
    } else if (msg.type === "error") {
      addBubble("assistant", `Error: ${msg.message}`);
    }
  };
}

function send() {
  const text = (inputEl.value || "").trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  addBubble("user", text);
  ws.send(JSON.stringify({ type: "user_message", text }));
  inputEl.value = "";
}

sendBtn.addEventListener("click", send);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

connect();
