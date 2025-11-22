// static/js/call_tts.js
document.addEventListener("DOMContentLoaded", () => {
  const speakBtn = document.getElementById("speak-btn");
  const backBtn = document.getElementById("back-btn");
  const autoListenCheckbox = document.getElementById("auto-listen");
  const statusEl = document.getElementById("status");
  const hiddenTranscript = document.getElementById("hidden-transcript");

  const filterTypeEl = document.getElementById("filter-type");
  const freqEl = document.getElementById("freq");
  const qEl = document.getElementById("q");
  const gainEl = document.getElementById("gain");

  const canvas = document.getElementById("wave-canvas");
  const ctx = canvas.getContext("2d");

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  backBtn && backBtn.addEventListener("click", () => { window.location.href = "/"; });

  let audioCtx = null;
  let inputAnalyser = null;
  let outputAnalyser = null;
  let inputStreamSource = null;
  let inputAnimationId = null;
  let outputAnimationId = null;

  function setStatus(t){ if(statusEl) statusEl.textContent = t; }

  function createFilterNode(ctx){
    const type = filterTypeEl ? filterTypeEl.value : "none";
    if(type === "none") return null;
    const f = ctx.createBiquadFilter();
    f.type = type;
    f.frequency.value = parseFloat(freqEl.value);
    f.Q.value = parseFloat(qEl.value);
    f.gain.value = parseFloat(gainEl.value);
    return f;
  }

  function drawWaveforms() {
    if (!canvas) return;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);

    const half = Math.floor(w / 2);
    const gap = 6;
    const leftW = half - gap;
    const rightW = w - half - gap;

    ctx.fillStyle = "#061019";
    ctx.fillRect(0, 0, leftW, h);
    ctx.fillRect(half + gap, 0, rightW, h);

    ctx.fillStyle = "#8fb8df";
    ctx.font = "12px sans-serif";
    ctx.fillText("INPUT", 8, 16);
    ctx.fillText("OUTPUT", half + gap + 8, 16);

    if (inputAnalyser) {
      const bufferLength = inputAnalyser.fftSize;
      const data = new Uint8Array(bufferLength);
      inputAnalyser.getByteTimeDomainData(data);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#66ff99";
      ctx.beginPath();
      const sliceWidth = leftW / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = data[i] / 128.0;
        const y = (v * h) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.stroke();
    } else {
      ctx.strokeStyle = "rgba(102,255,153,0.15)";
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(leftW, h / 2);
      ctx.stroke();
    }

    if (outputAnalyser) {
      const bufferLength = outputAnalyser.fftSize;
      const data = new Uint8Array(bufferLength);
      outputAnalyser.getByteTimeDomainData(data);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#99ccff";
      ctx.beginPath();
      const sliceWidth = rightW / bufferLength;
      let x = half + gap;
      for (let i = 0; i < bufferLength; i++) {
        const v = data[i] / 128.0;
        const y = (v * h) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.stroke();
    } else {
      ctx.strokeStyle = "rgba(153,204,255,0.12)";
      ctx.beginPath();
      ctx.moveTo(half + gap, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();
    }
  }

  async function startInputVisualizer() {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (inputStreamSource) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      inputStreamSource = audioCtx.createMediaStreamSource(stream);
      inputAnalyser = audioCtx.createAnalyser();
      inputAnalyser.fftSize = 2048;
      inputStreamSource.connect(inputAnalyser);

      const tick = () => {
        drawWaveforms();
        inputAnimationId = requestAnimationFrame(tick);
      };
      if (!inputAnimationId) tick();
    } catch (err) {
      console.warn("Microphone access denied or not available:", err);
    }
  }

  function stopInputVisualizer() {
    if (inputAnimationId) {
      cancelAnimationFrame(inputAnimationId);
      inputAnimationId = null;
    }
    if (inputStreamSource && inputAnalyser) {
      try { inputStreamSource.disconnect(); } catch (e) {}
      inputStreamSource = null;
    }
    inputAnalyser = null;
    drawWaveforms();
  }

  async function playArrayBufferAudio(arrayBuffer) {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await audioCtx.decodeAudioData(arrayBuffer);
    const src = audioCtx.createBufferSource();
    src.buffer = decoded;

    const filterNode = createFilterNode(audioCtx);
    outputAnalyser = audioCtx.createAnalyser();
    outputAnalyser.fftSize = 2048;

    if (filterNode) {
      src.connect(filterNode);
      filterNode.connect(outputAnalyser);
    } else {
      src.connect(outputAnalyser);
    }
    outputAnalyser.connect(audioCtx.destination);

    const tickOut = () => {
      drawWaveforms();
      outputAnimationId = requestAnimationFrame(tickOut);
    };
    if (!outputAnimationId) tickOut();

    return new Promise(resolve => {
      src.onended = () => {
        if (outputAnimationId) {
          cancelAnimationFrame(outputAnimationId);
          outputAnimationId = null;
        }
        try { outputAnalyser.disconnect(); } catch (e) {}
        outputAnalyser = null;
        drawWaveforms();
        resolve();
      };
      src.start(0);
    });
  }

  function base64ToArrayBuffer(b64) {
    const binaryStr = atob(b64);
    const len = binaryStr.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binaryStr.charCodeAt(i);
    return bytes.buffer;
  }

  async function requestTTSAndPlay(text) {
    setStatus("Generating voice...");
    try {
      const res = await fetch("/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      const j = await res.json();
      if (j.error) {
        console.error("TTS error:", j);
        setStatus("TTS server error");
        return;
      }
      const arrayBuffer = base64ToArrayBuffer(j.audio_base64);
      setStatus("Playing reply...");
      await playArrayBufferAudio(arrayBuffer);
      setStatus("Ready.");
    } catch (e) {
      console.error("Request/Play error:", e);
      setStatus("Error");
    }
  }

  // Start chat on load: call /start_chat and play intro
  async function startChatOnLoad() {
    try {
      setStatus("Connecting...");
      const resp = await fetch('/start_chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({}) });
      const j = await resp.json();
      if (j.reply) {
        setStatus(j.reply);
        await requestTTSAndPlay(j.reply);
        if (autoListenCheckbox && autoListenCheckbox.checked) {
          try { recognition.start(); } catch (e) { console.warn(e); }
        }
      } else {
        setStatus("Ready. Tap to speak.");
      }
    } catch (e) {
      console.error("start_chat failed", e);
      setStatus("Could not start chat.");
    }
  }

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    speakBtn.disabled = true;
    speakBtn.textContent = "Voice not supported";
    setStatus("Speech recognition not supported");
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    setStatus("Listening...");
    startInputVisualizer();
  };

  recognition.onresult = async (ev) => {
    const userText = ev.results[0][0].transcript;
    hiddenTranscript.textContent = userText;
    setStatus("Processing...");
    stopInputVisualizer();

    try {
      const p = await fetch("/process_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText })
      });
      const pj = await p.json();
      const reply = pj.reply || "Sorry, I couldn't process that.";
      await requestTTSAndPlay(reply);

      if (pj.ended) {
        try { await fetch('/end_chat', { method: 'POST' }); } catch (e) {}
        window.location.href = '/rating';
        return;
      }

      if (autoListenCheckbox && autoListenCheckbox.checked) {
        try { recognition.start(); } catch (e) { console.warn(e); }
      } else {
        setStatus("Ready.");
      }
    } catch (err) {
      console.error("process_text error:", err);
      setStatus("Server error.");
    }
  };

  recognition.onend = () => {
    if (!(autoListenCheckbox && autoListenCheckbox.checked)) setStatus("Ready. Tap to speak.");
    stopInputVisualizer();
  };

  recognition.onerror = (e) => {
    console.error("Recognition error:", e);
    setStatus("Recognition error.");
    setTimeout(()=> setStatus("Ready. Tap to speak."), 1000);
    stopInputVisualizer();
  };

  speakBtn.addEventListener("click", () => {
    setStatus("Listening...");
    try { recognition.start(); } catch (e) { console.warn("recognition.start error:", e); }
  });

  drawWaveforms();
  setStatus("Ready. Tap to speak.");
  startChatOnLoad();
});
