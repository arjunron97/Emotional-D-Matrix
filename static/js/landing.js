document.addEventListener("DOMContentLoaded", () => {
  const chatBtn = document.getElementById("chat-mode-btn");
  const callBtn = document.getElementById("call-mode-btn");

  function gatherUserInfo() {
    const name = document.getElementById("name").value.trim();
    const age = document.getElementById("age").value.trim();
    const gender = document.getElementById("gender").value;
    const email = document.getElementById("email").value.trim();
    const location = document.getElementById("location").value.trim();

    if (!name || !age || !gender) {
      alert("Please fill Name, Age, and Gender.");
      return null;
    }

    return { name, age, gender, email, location };
  }

  async function saveAndGo(mode) {
    const info = gatherUserInfo();
    if (!info) return;
    try {
      const res = await fetch("/save_user_info", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(info)
      });
      const data = await res.json();
      if (data.status !== "ok") {
        alert("Failed to save info: " + (data.message || "unknown"));
        return;
      }
      // redirect
      window.location.href = mode === "chat" ? "/chat" : "/call";
    } catch (err) {
      console.error("Error saving user info", err);
      alert("Could not save user info. Check server.");
    }
  }

  chatBtn.addEventListener("click", () => saveAndGo("chat"));
  callBtn.addEventListener("click", () => saveAndGo("call"));
});
