document.addEventListener("DOMContentLoaded", () => {
  const chatBox = document.getElementById("chat-box");
  const sendBtn = document.getElementById("send-btn");
  const voiceBtn = document.getElementById("voice-btn");
  const userInput = document.getElementById("user-input");
  const inputModes = document.getElementsByName("input-mode");
  const backBtn = document.getElementById("back-btn");

  backBtn && backBtn.addEventListener("click", () => { window.location.href = "/"; });

  let currentMode = "text";
  inputModes.forEach(r => r.addEventListener("change", e => {
    currentMode = e.target.value;
    if (currentMode === "text") {
      userInput.style.display = "block";
      sendBtn.style.display = "inline-block";
      voiceBtn.style.display = "none";
    } else {
      userInput.style.display = "none";
      sendBtn.style.display = "none";
      voiceBtn.style.display = "inline-block";
    }
  }));

  // init visibility
  if (currentMode === "voice") {
    userInput.style.display = "none";
    sendBtn.style.display = "none";
    voiceBtn.style.display = "inline-block";
  }

  function addMessage(text, cls) {
    const d = document.createElement("div");
    d.className = `message ${cls}`;
    d.textContent = text;
    chatBox.appendChild(d);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  async function sendToBackend(text) {
    try {
      console.log("Sending to /process_text:", text);
      const res = await fetch("/process_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
      });
      const data = await res.json();
      addMessage(data.reply, "bot-message");
    } catch (err) {
      console.error("Error sending to backend:", err);
      addMessage("⚠️ Server error", "bot-message");
    }
  }

  sendBtn.addEventListener("click", () => {
    const txt = userInput.value.trim();
    if (!txt) return;
    addMessage(txt, "user-message");
    userInput.value = "";
    sendToBackend(txt);
  });

  // Voice support
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    voiceBtn.addEventListener("click", () => recognition.start());

    recognition.onresult = (ev) => {
      const txt = ev.results[0][0].transcript;
      addMessage(txt, "user-message");
      sendToBackend(txt);
    };

    recognition.onerror = (e) => {
      console.error("Recognition error:", e);
    };
  } else {
    voiceBtn.disabled = true;
    voiceBtn.textContent = "Voice not supported";
  }
});
