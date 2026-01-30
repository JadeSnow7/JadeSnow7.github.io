(() => {
    const container = document.getElementById("chatbot-container");
    if (!container) return;

    // HTML Structure
    const html = `
    <button class="chatbot-toggle" aria-label="Open Chat" aria-expanded="false">
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z" fill="currentColor"/>
      </svg>
    </button>
    <div class="chatbot-window" role="dialog" aria-modal="false" aria-label="Chat Window" hidden>
      <header class="chatbot-header">
        <h3>Chat with Bot</h3>
        <button class="chatbot-close" aria-label="Close Chat">×</button>
      </header>
      <div class="chatbot-messages" role="log" aria-live="polite">
        <div class="message bot">Hello! I'm a demo bot. How can I help you?</div>
      </div>
      <div class="chatbot-input-area">
        <input type="text" class="chatbot-input" placeholder="Type a message..." aria-label="Message Input">
        <button class="chatbot-send" aria-label="Send Message" disabled>
          <svg viewBox="0 0 24 24">
            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="currentColor"/>
          </svg>
        </button>
      </div>
    </div>
  `;

    container.innerHTML = html;

    // Elements
    const toggleBtn = container.querySelector(".chatbot-toggle");
    const windowEl = container.querySelector(".chatbot-window");
    const closeBtn = container.querySelector(".chatbot-close");
    const messagesEl = container.querySelector(".chatbot-messages");
    const inputEl = container.querySelector(".chatbot-input");
    const sendBtn = container.querySelector(".chatbot-send");

    // State
    let isOpen = false;
    const STATE_KEY = "chatbot_open";

    // Functions
    const toggleChat = (forceState) => {
        isOpen = forceState !== undefined ? forceState : !isOpen;

        if (isOpen) {
            windowEl.classList.add("is-open");
            windowEl.removeAttribute("hidden");
            toggleBtn.setAttribute("aria-expanded", "true");
            // Focus management
            setTimeout(() => inputEl.focus(), 50); // Small delay for transition
        } else {
            windowEl.classList.remove("is-open");
            toggleBtn.setAttribute("aria-expanded", "false");
            // Wait for transition to finish before hiding (optional, or just hidden for accessibility)
            // For a11y, we need to ensure it's hidden from screen readers when closed
            // But we want transition. We'll leave it in DOM but visually hidden via CSS transform
            // And add 'hidden' attribute after timeout? 
            // Actually CSS handles opacity/pointer-events. 'hidden' attribute hides it completely.
            // Let's rely on CSS mostly, but for screen readers:
            setTimeout(() => {
                if (!isOpen) windowEl.setAttribute("hidden", "");
            }, 300);
            toggleBtn.focus();
        }

        try {
            localStorage.setItem(STATE_KEY, isOpen);
        } catch (e) { }
    };

    const addMessage = (text, type = "bot") => {
        const msg = document.createElement("div");
        msg.className = `message ${type}`;
        msg.textContent = text;
        messagesEl.appendChild(msg);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    };

    // --- MOCK BACKEND CALL ---
    // API SECURITY WARNING:
    // When implementing the real backend, DO NOT expose your API Keys here.
    // Instead, call your own backend/proxy (e.g. /api/chat) which handles the keys.
    const sendMessage = async (text) => {
        addMessage(text, "user");
        inputEl.value = "";
        sendBtn.disabled = true;

        // Simulate network delay
        setTimeout(() => {
            // Simple mock response logic
            let reply = "I'm just a demo bot currently. Use me to test the UI!";
            if (text.toLowerCase().includes("hello")) reply = "Hi there! Welcome to Hu Aodong's portfolio.";
            if (text.toLowerCase().includes("project")) reply = "You can check out the Projects section for details on StoryToVideo and SwiftSweep.";
            if (text.toLowerCase().includes("contact")) reply = "Feel free to email at 1079249368@qq.com";

            addMessage(reply, "bot");
            sendBtn.disabled = false;
            inputEl.focus();
        }, 800);
    };

    // Event Listeners
    toggleBtn.addEventListener("click", () => toggleChat());
    closeBtn.addEventListener("click", () => toggleChat(false));

    inputEl.addEventListener("input", () => {
        sendBtn.disabled = !inputEl.value.trim();
    });

    inputEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey && inputEl.value.trim()) {
            e.preventDefault();
            sendMessage(inputEl.value.trim());
        }
    });

    sendBtn.addEventListener("click", () => {
        if (inputEl.value.trim()) sendMessage(inputEl.value.trim());
    });

    // Global Close on Escape
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && isOpen) {
            toggleChat(false);
        }
    });

    // Initialization
    try {
        const savedState = localStorage.getItem(STATE_KEY);
        if (savedState === "true") {
            toggleChat(true);
        }
    } catch (e) { }

})();
