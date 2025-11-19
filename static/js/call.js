document.addEventListener("DOMContentLoaded", () => {
  const speakBtn = document.getElementById("speak-btn");
  const chatBox = document.getElementById("call-chat");
  const backBtn = document.getElementById("back-btn");
  const autoListen = document.getElementById("auto-listen");

  backBtn && backBtn.addEventListener("click", () => { window.location.href = "/"; });

  function addMessage(text, cls) {
    const d = document.createElement("div");
    d.className = `message ${cls}`;
    d.textContent = text;
    chatBox.appendChild(d);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  // TTS helper
  function speak(text, onend) {
    if (!("speechSynthesis" in window)) {
      addMessage("TTS not supported in this browser.", "bot-message");
      if (onend) onend();
      return;
    }
    const u = new SpeechSynthesisUtterance(text);
    u.onend = () => { if (onend) onend(); };
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  }

  // Voice recognition
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    speakBtn.disabled = true;
    speakBtn.textContent = "Voice not supported";
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.interimResults = false;

  // When user clicks speak button, start recognition
  speakBtn.addEventListener("click", () => {
    // UI feedback
    speakBtn.textContent = "Listening...";
    recognition.start();
  });

  recognition.onresult = (ev) => {
    const text = ev.results[0][0].transcript;
    addMessage(text, "user-message");

    // Send to backend
    fetch("/process_text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
    })
      .then(r => r.json())
      .then(data => {
        addMessage(data.reply, "bot-message");
        // speak the reply, and optionally auto-listen afterwards
        speak(data.reply, () => {
          if (autoListen && autoListen.checked) {
            // auto re-start listening
            recognition.start();
          } else {
            speakBtn.textContent = "🎙️ Tap to Speak";
          }
        });
      })
      .catch(err => {
        console.error(err);
        addMessage("⚠️ Server error", "bot-message");
        speakBtn.textContent = "🎙️ Tap to Speak";
      });
  };

  recognition.onend = () => {
    // reset button if not auto-listening
    if (!(autoListen && autoListen.checked)) speakBtn.textContent = "🎙️ Tap to Speak";
  };

  recognition.onerror = (e) => {
    console.error("Recognition error:", e);
    speakBtn.textContent = "🎙️ Tap to Speak";
  };
});
