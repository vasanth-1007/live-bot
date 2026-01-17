/**
 * Gemini Live RAG - Frontend
 * Handles text chat, voice recording, and audio playback
 */

class GeminiLiveChat {
    constructor() {
        // DOM Elements
        this.chatEl = document.getElementById("chat");
        this.inputEl = document.getElementById("input");
        this.sendBtn = document.getElementById("send");
        this.statusEl = document.getElementById("status");
        this.sourcesEl = document.getElementById("sources");
        this.micBtn = document.getElementById("mic-btn");
        this.recordingStatus = document.getElementById("recording-status");
        this.transcriptPreview = document.getElementById("transcript-preview");
        this.liveTranscript = document.getElementById("live-transcript");

        // WebSocket
        this.ws = null;

        // Chat state
        this.currentAssistantBubble = null;

        // Audio Recording
        this.isRecording = false;
        this.audioContext = null;
        this.mediaStream = null;
        this.processor = null;
        this.source = null;

        // Audio Playback
        this.playbackContext = null;
        this.audioQueue = [];
        this.isPlaying = false;
        this.sampleRate = 24000; // Default output sample rate

        // Initialize
        this.init();
    }

    init() {
        this.connect();
        this.setupEventListeners();
    }

    // =========================================================================
    // WebSocket Connection
    // =========================================================================

    connect() {
        this.setStatus("connecting", "Connecting…");

        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws/chat`;

        console.log("Connecting to:", wsUrl);
        this.ws = new WebSocket(wsUrl);
        this.ws.binaryType = "arraybuffer";

        this.ws.onopen = () => {
            console.log("WebSocket connected");
            this.setStatus("connected", "✓ Connected");
        };

        this.ws.onclose = (evt) => {
            console.warn("WebSocket closed:", evt);
            this.setStatus("disconnected", "✗ Disconnected");
            // Reconnect after 3 seconds
            setTimeout(() => this.connect(), 3000);
        };

        this.ws.onerror = (evt) => {
            console.error("WebSocket error:", evt);
            this.setStatus("disconnected", "Connection Error");
        };

        this.ws.onmessage = (evt) => this.handleMessage(evt);
    }

    setStatus(type, text) {
        this.statusEl.textContent = text;
        this.statusEl.className = `status ${type}`;
    }

    // =========================================================================
    // Message Handling
    // =========================================================================

    handleMessage(evt) {
        // Binary data = audio from assistant
        if (evt.data instanceof ArrayBuffer) {
            console.log("Received audio chunk:", evt.data.byteLength, "bytes");
            this.queueAudio(evt.data);
            return;
        }

        // Text data = JSON
        try {
            const msg = JSON.parse(evt.data);
            console.log("Received:", msg.type, msg);

            switch (msg.type) {
                case "assistant_audio_format":
                    // Parse sample rate from mime_type like "audio/pcm;rate=24000"
                    const match = msg.mime_type?.match(/rate=(\d+)/);
                    if (match) {
                        this.sampleRate = parseInt(match[1], 10);
                        console.log("Audio sample rate:", this.sampleRate);
                    }
                    break;

                case "assistant_start":
                    this.onAssistantStart();
                    break;

                case "assistant_delta":
                    this.onAssistantDelta(msg.delta);
                    break;

                case "assistant_done":
                    this.onAssistantDone(msg.text, msg.sources);
                    break;

                case "user_transcript":
                    this.onUserTranscript(msg.text);
                    break;

                case "error":
                    this.addBubble("error", `Error: ${msg.message}`);
                    break;

                default:
                    console.log("Unknown message type:", msg.type);
            }
        } catch (e) {
            console.error("Failed to parse message:", e);
        }
    }

    onAssistantStart() {
        // Remove welcome message if present
        const welcome = this.chatEl.querySelector(".welcome-message");
        if (welcome) welcome.remove();

        // Create new assistant bubble
        this.currentAssistantBubble = this.addBubble("assistant", "");
        this.currentAssistantBubble.classList.add("playing");
    }

    onAssistantDelta(delta) {
        if (!this.currentAssistantBubble) {
            this.currentAssistantBubble = this.addBubble("assistant", "");
        }
        this.currentAssistantBubble.textContent += delta;
        this.scrollToBottom();
    }

    onAssistantDone(text, sources) {
        if (this.currentAssistantBubble) {
            this.currentAssistantBubble.classList.remove("playing");
            if (text) {
                this.currentAssistantBubble.textContent = text;
            }
        }
        this.currentAssistantBubble = null;
        this.renderSources(sources || []);
        this.scrollToBottom();
    }

    onUserTranscript(text) {
        // Show what the user said (from voice input)
        if (text && text.trim()) {
            // Remove welcome message if present
            const welcome = this.chatEl.querySelector(".welcome-message");
            if (welcome) welcome.remove();

            this.addBubble("user", text);
            this.liveTranscript.textContent = text;
        }
    }

    addBubble(role, text) {
        const div = document.createElement("div");
        div.className = `bubble ${role}`;
        div.textContent = text;
        this.chatEl.appendChild(div);
        this.scrollToBottom();
        return div;
    }

    scrollToBottom() {
        this.chatEl.scrollTop = this.chatEl.scrollHeight;
    }

    renderSources(sources) {
        this.sourcesEl.innerHTML = "";
        if (!sources || sources.length === 0) {
            this.sourcesEl.textContent = "(No sources retrieved)";
            return;
        }

        for (const s of sources) {
            const card = document.createElement("div");
            card.className = "source";
            card.innerHTML = `
                <div class="source-title">${s.id || "Unknown"}</div>
                <div class="source-preview"></div>
                <pre class="source-meta"></pre>
            `;
            card.querySelector(".source-preview").textContent = s.text_preview || "";
            card.querySelector(".source-meta").textContent = 
                JSON.stringify(s.properties || {}, null, 2);
            this.sourcesEl.appendChild(card);
        }
    }

    // =========================================================================
    // Event Listeners
    // =========================================================================

    setupEventListeners() {
        // Text send button
        this.sendBtn.addEventListener("click", () => this.sendText());

        // Enter key to send
        this.inputEl.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                this.sendText();
            }
        });

        // Microphone button - Push to talk
        this.micBtn.addEventListener("mousedown", (e) => {
            e.preventDefault();
            this.startRecording();
        });

        this.micBtn.addEventListener("mouseup", () => {
            this.stopRecording();
        });

        this.micBtn.addEventListener("mouseleave", () => {
            if (this.isRecording) {
                this.stopRecording();
            }
        });

        // Touch support for mobile
        this.micBtn.addEventListener("touchstart", (e) => {
            e.preventDefault();
            this.startRecording();
        });

        this.micBtn.addEventListener("touchend", (e) => {
            e.preventDefault();
            this.stopRecording();
        });

        this.micBtn.addEventListener("touchcancel", () => {
            this.stopRecording();
        });
    }

    sendText() {
        const text = (this.inputEl.value || "").trim();
        if (!text || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;

        // Remove welcome message if present
        const welcome = this.chatEl.querySelector(".welcome-message");
        if (welcome) welcome.remove();

        this.addBubble("user", text);
        this.ws.send(JSON.stringify({ type: "user_message", text }));
        this.inputEl.value = "";

        // Ensure playback context is ready
        this.ensurePlaybackContext();
    }

    // =========================================================================
    // Audio Recording (User voice input)
    // =========================================================================

    async startRecording() {
        if (this.isRecording) return;

        try {
            // Ensure playback context is ready (requires user gesture)
            this.ensurePlaybackContext();

            // Get microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
            });

            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000,
            });

            this.source = this.audioContext.createMediaStreamSource(this.mediaStream);

            // Create ScriptProcessor for capturing audio
            // Note: ScriptProcessorNode is deprecated but widely supported
            // For production, consider using AudioWorkletNode
            this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);

            this.processor.onaudioprocess = (e) => {
                if (!this.isRecording) return;

                const float32Data = e.inputBuffer.getChannelData(0);
                const pcm16Data = this.float32ToPCM16(float32Data);

                // Send audio data to server
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(pcm16Data.buffer);
                }
            };

            this.source.connect(this.processor);
            this.processor.connect(this.audioContext.destination);

            // Tell server we're starting audio
            this.ws.send(JSON.stringify({
                type: "audio_start",
                sample_rate_hz: 16000,
            }));

            this.isRecording = true;
            this.micBtn.classList.add("recording");
            this.micBtn.querySelector(".mic-text").textContent = "Recording...";
            this.recordingStatus.classList.remove("hidden");
            this.transcriptPreview.classList.remove("hidden");
            this.liveTranscript.textContent = "";

            console.log("Recording started");
        } catch (err) {
            console.error("Failed to start recording:", err);
            alert("Could not access microphone. Please check permissions.");
        }
    }

    stopRecording() {
        if (!this.isRecording) return;

        this.isRecording = false;
        this.micBtn.classList.remove("recording");
        this.micBtn.querySelector(".mic-text").textContent = "Hold to Speak";
        this.recordingStatus.classList.add("hidden");

        // Tell server recording ended
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: "audio_end" }));
        }

        // Clean up audio resources
        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }
        if (this.source) {
            this.source.disconnect();
            this.source = null;
        }
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach((track) => track.stop());
            this.mediaStream = null;
        }

        console.log("Recording stopped");
    }

    float32ToPCM16(float32Array) {
        const pcm16 = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            let s = Math.max(-1, Math.min(1, float32Array[i]));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        return pcm16;
    }

    // =========================================================================
    // Audio Playback (Assistant voice output)
    // =========================================================================

    ensurePlaybackContext() {
        if (!this.playbackContext || this.playbackContext.state === "closed") {
            this.playbackContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate,
            });
        }
        if (this.playbackContext.state === "suspended") {
            this.playbackContext.resume();
        }
    }

    queueAudio(arrayBuffer) {
        this.audioQueue.push(arrayBuffer);
        if (!this.isPlaying) {
            this.playNextChunk();
        }
    }

    async playNextChunk() {
        if (this.audioQueue.length === 0) {
            this.isPlaying = false;
            return;
        }

        this.isPlaying = true;
        this.ensurePlaybackContext();

        const arrayBuffer = this.audioQueue.shift();

        try {
            // Convert PCM16 (Int16) to Float32
            const pcm16 = new Int16Array(arrayBuffer);
            const float32 = new Float32Array(pcm16.length);
            for (let i = 0; i < pcm16.length; i++) {
                float32[i] = pcm16[i] / 32768.0;
            }

            // Create audio buffer
            const audioBuffer = this.playbackContext.createBuffer(
                1, // mono
                float32.length,
                this.sampleRate
            );
            audioBuffer.getChannelData(0).set(float32);

            // Play
            const sourceNode = this.playbackContext.createBufferSource();
            sourceNode.buffer = audioBuffer;
            sourceNode.connect(this.playbackContext.destination);
            sourceNode.onended = () => this.playNextChunk();
            sourceNode.start();
        } catch (e) {
            console.error("Playback error:", e);
            // Try next chunk
            this.playNextChunk();
        }
    }
}

// ============================================================================
// Initialize on page load
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
    window.chat = new GeminiLiveChat();
});