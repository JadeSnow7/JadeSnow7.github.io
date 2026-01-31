(() => {
    const container = document.getElementById("chatbot-container");
    if (!container) return;

    const STATE_KEY = "chatbot_open";
    const MODE_KEY = "chatbot_mode";

    const FALLBACK = {
        ui: {
            title: "Resume Assistant",
            subtitlePro: "Projects, stack, and resume Q&A",
            subtitleFun: "Interests mode (optional)",
            greetingPro: "Hi! I'm Hu Aodong's resume assistant.",
            greetingFun: "Hi! You're in interests mode.",
            inputPlaceholder: "Type a question",
            openLabel: "Open chat",
            closeLabel: "Close chat",
            clearLabel: "Clear",
            modePro: "Pro",
            modeFun: "Fun",
            switchToFun: "Switch to Fun",
            switchToPro: "Switch to Pro",
            funDisclaimer: "Astrology/tarot is for fun only.",
            proGate: "Switch to Fun mode to talk about hobbies.",
            unknown: "Try a quick action below.",
            modeChangedPro: "Switched to Pro mode.",
            modeChangedFun: "Switched to Fun mode.",
            error: "Sorry, failed to send.",
        },
        labels: {
            projectsTitle: "Project Overview",
            projectsIntro: "Here are the main projects:",
            projectDetail: "Project Detail",
            stackTitle: "Tech Stack",
            resumeTitle: "Resume Highlights",
            contactTitle: "Contact",
            essaysTitle: "Essays",
            hobbiesTitle: "Interests",
            cultureTitle: "Traditional Culture",
            historyTitle: "Worldbuilding",
            tarotTitle: "Astrology & Tarot",
            languages: "Languages",
            systems: "Systems & Tools",
            ai: "AI / Multimodal",
            env: "Dev Environment",
            email: "Email",
            phone: "Phone",
            github: "GitHub",
            resumePage: "Resume Page",
            resumePdf: "Resume PDF",
            essaysLink: "Go to Essays",
        },
        actions: {
            projects: "Projects",
            stack: "Tech Stack",
            resume: "Resume",
            essays: "Essays",
            contact: "Contact",
            hobbies: "Hobbies",
            culture: "Traditional Culture",
            history: "Worldbuilding",
            tarot: "Astrology & Tarot",
            modeFun: "Switch to Fun",
            modePro: "Switch to Pro",
        },
        profile: { summary: "", highlights: [] },
        projects: [],
        stack: { languages: [], systems: [], ai: [], env: [] },
        essays: { intro: "", link: "essays/index.html" },
        contact: { email: "", phone: "", github: "", resumePage: "resume.html", resumePdf: "assets/resume.pdf" },
        hobbies: { summary: "", culture: [], history: [], tarot: [] },
    };

    const getLang = () => {
        const raw = (window.__SITE_LANG__ || document.documentElement.lang || "").toLowerCase();
        return raw.startsWith("en") ? "en" : "zh";
    };

    const getStrings = (lang) => {
        const kb = window.__CHATBOT_KB__ || {};
        return kb[lang] || kb.zh || FALLBACK;
    };

    const getConfig = () => window.__CHATBOT_CONFIG__ || {};

	    const config = getConfig();
	    let apiEndpoint = (container.dataset.endpoint || config.apiEndpoint || "").trim();
	    if (!apiEndpoint) {
	        const host = window.location.hostname;
	        if (host === "localhost" || host === "127.0.0.1" || window.location.protocol === "file:") {
	            apiEndpoint = "http://127.0.0.1:8000/api/chat";
	        } else if (!host.endsWith("github.io")) {
	            // When self-hosting (e.g. huaodong.online), use same-origin backend to avoid CORS/mixed-content issues.
	            apiEndpoint = "/api/chat";
	        }
	    }

    const html = `
    <button class="chatbot-toggle" aria-label="Open Chat" aria-expanded="false">
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z" fill="currentColor"/>
      </svg>
    </button>
    <div class="chatbot-window" role="dialog" aria-modal="false" aria-label="Chat Window" hidden>
      <header class="chatbot-header">
        <div class="chatbot-header-top">
          <div class="chatbot-title">
            <h3 data-role="title"></h3>
            <p class="chatbot-subtitle" data-role="subtitle"></p>
          </div>
          <button class="chatbot-close" aria-label="Close Chat">×</button>
        </div>
        <div class="chatbot-header-bottom">
          <div class="chatbot-mode-switch" role="group" aria-label="Chat mode">
            <button class="chatbot-mode-btn" type="button" data-mode="pro"></button>
            <button class="chatbot-mode-btn" type="button" data-mode="fun"></button>
          </div>
          <button class="chatbot-clear" type="button"></button>
        </div>
      </header>
      <div class="chatbot-messages" role="log" aria-live="polite"></div>
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

    const toggleBtn = container.querySelector(".chatbot-toggle");
    const windowEl = container.querySelector(".chatbot-window");
    const titleEl = container.querySelector("[data-role=\"title\"]");
    const subtitleEl = container.querySelector("[data-role=\"subtitle\"]");
    const closeBtn = container.querySelector(".chatbot-close");
    const clearBtn = container.querySelector(".chatbot-clear");
    const modeButtons = Array.from(container.querySelectorAll(".chatbot-mode-btn"));
    const messagesEl = container.querySelector(".chatbot-messages");
    const inputEl = container.querySelector(".chatbot-input");
    const sendBtn = container.querySelector(".chatbot-send");

    let isOpen = false;
    let mode = "pro";

    const FUN_INTENTS = new Set(["hobbies", "culture", "history", "tarot"]);

    const ACTIONS_BY_MODE = {
        pro: ["projects", "stack", "resume", "essays", "contact", "hobbies"],
        fun: ["culture", "history", "tarot", "projects", "stack", "contact"],
    };

    const ACTION_INTENTS = {
        projects: "projects",
        stack: "stack",
        resume: "resume",
        essays: "essays",
        contact: "contact",
        hobbies: "hobbies",
        culture: "culture",
        history: "history",
        tarot: "tarot",
    };

    const KEYWORDS = {
        projectStory: ["storytovideo", "story to video", "视频生成", "视频", "分镜"],
        projectSwift: ["swiftsweep", "swift sweep", "mac", "系统优化", "清理"],
        projectGraduation: ["教学平台", "毕业设计", "graduation", "teaching platform"],
        projects: ["项目", "project", "projects", "作品", "开源"],
        stack: ["技术栈", "stack", "skills", "能力", "技术", "语言", "工具"],
        resume: ["简历", "resume", "cv", "经历", "教育", "背景"],
        contact: ["联系", "contact", "邮箱", "email", "电话", "phone"],
        essays: ["随笔", "essays", "写作", "记录", "文章"],
        hobbies: ["兴趣", "爱好", "hobby", "hobbies"],
        culture: ["传统文化", "国学", "文化", "classics", "culture"],
        history: ["架空历史", "worldbuilding", "alternate history", "设定"],
        tarot: ["星盘", "塔罗", "占卜", "astrology", "tarot"],
    };

    const includesAny = (text, keywords) => keywords.some((kw) => text.includes(kw));

    const getIntent = (text, override) => {
        if (override) return override;
        const normalized = text.toLowerCase();
        if (includesAny(normalized, KEYWORDS.projectStory)) return "project_storytovideo";
        if (includesAny(normalized, KEYWORDS.projectSwift)) return "project_swiftsweep";
        if (includesAny(normalized, KEYWORDS.projectGraduation)) return "project_graduation";
        if (includesAny(normalized, KEYWORDS.projects)) return "projects";
        if (includesAny(normalized, KEYWORDS.stack)) return "stack";
        if (includesAny(normalized, KEYWORDS.resume)) return "resume";
        if (includesAny(normalized, KEYWORDS.contact)) return "contact";
        if (includesAny(normalized, KEYWORDS.essays)) return "essays";
        if (includesAny(normalized, KEYWORDS.tarot)) return "tarot";
        if (includesAny(normalized, KEYWORDS.culture)) return "culture";
        if (includesAny(normalized, KEYWORDS.history)) return "history";
        if (includesAny(normalized, KEYWORDS.hobbies)) return "hobbies";
        return "unknown";
    };

    const setMode = (nextMode, options = {}) => {
        if (mode === nextMode) return;
        mode = nextMode;
        updateModeButtons();
        updateStrings();
        try {
            localStorage.setItem(MODE_KEY, mode);
        } catch (e) {
            /* ignore */
        }
        if (options.announce) {
            const strings = getStrings(getLang());
            addMessage("system", nextMode === "fun" ? strings.ui.modeChangedFun : strings.ui.modeChangedPro);
        }
        renderQuickActions();
    };

    const updateModeButtons = () => {
        modeButtons.forEach((btn) => {
            const isActive = btn.dataset.mode === mode;
            btn.classList.toggle("is-active", isActive);
            btn.setAttribute("aria-pressed", isActive ? "true" : "false");
        });
    };

    const updateStrings = () => {
        const strings = getStrings(getLang());
        const ui = strings.ui || FALLBACK.ui;
        if (titleEl) titleEl.textContent = ui.title || FALLBACK.ui.title;
        if (subtitleEl) subtitleEl.textContent = mode === "fun" ? ui.subtitleFun : ui.subtitlePro;
        if (toggleBtn) toggleBtn.setAttribute("aria-label", ui.openLabel || FALLBACK.ui.openLabel);
        if (closeBtn) closeBtn.setAttribute("aria-label", ui.closeLabel || FALLBACK.ui.closeLabel);
        if (clearBtn) clearBtn.textContent = ui.clearLabel || FALLBACK.ui.clearLabel;
        modeButtons.forEach((btn) => {
            if (btn.dataset.mode === "pro") btn.textContent = ui.modePro || FALLBACK.ui.modePro;
            if (btn.dataset.mode === "fun") btn.textContent = ui.modeFun || FALLBACK.ui.modeFun;
        });
        if (inputEl) inputEl.placeholder = ui.inputPlaceholder || FALLBACK.ui.inputPlaceholder;
        updateActionLabels();
    };

    const updateActionLabels = () => {
        const strings = getStrings(getLang());
        const actions = strings.actions || FALLBACK.actions;
        messagesEl.querySelectorAll(".chatbot-chip").forEach((btn) => {
            const key = btn.dataset.action;
            if (key && actions[key]) btn.textContent = actions[key];
        });
    };

    const toggleChat = (forceState) => {
        isOpen = forceState !== undefined ? forceState : !isOpen;

        if (isOpen) {
            windowEl.classList.add("is-open");
            windowEl.removeAttribute("hidden");
            toggleBtn.setAttribute("aria-expanded", "true");
            setTimeout(() => inputEl.focus(), 50);
        } else {
            windowEl.classList.remove("is-open");
            toggleBtn.setAttribute("aria-expanded", "false");
            setTimeout(() => {
                if (!isOpen) windowEl.setAttribute("hidden", "");
            }, 300);
            toggleBtn.focus();
        }

        try {
            localStorage.setItem(STATE_KEY, isOpen);
        } catch (e) {
            /* ignore */
        }
    };

    const createMessage = (type, content, options = {}) => {
        const msg = document.createElement("div");
        msg.className = `message ${type}`.trim();
        if (options.extraClass) msg.classList.add(options.extraClass);
        if (typeof content === "string") {
            msg.textContent = content;
        } else if (content instanceof Node) {
            msg.appendChild(content);
        }
        return msg;
    };

    const addMessage = (type, content, options) => {
        const msg = createMessage(type, content, options);
        messagesEl.appendChild(msg);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return msg;
    };

    const buildContent = ({ title, body, bullets, footer, links }) => {
        const wrapper = document.createElement("div");
        if (title) {
            const heading = document.createElement("p");
            heading.className = "message-title";
            heading.textContent = title;
            wrapper.appendChild(heading);
        }
        if (body) {
            const bodyLines = Array.isArray(body) ? body : [body];
            bodyLines.forEach((line) => {
                if (!line) return;
                const p = document.createElement("p");
                p.textContent = line;
                wrapper.appendChild(p);
            });
        }
        if (bullets && bullets.length) {
            const ul = document.createElement("ul");
            bullets.forEach((item) => {
                const li = document.createElement("li");
                li.textContent = item;
                ul.appendChild(li);
            });
            wrapper.appendChild(ul);
        }
        if (footer) {
            const p = document.createElement("p");
            p.className = "message-meta";
            p.textContent = footer;
            wrapper.appendChild(p);
        }
        if (links && links.length) {
            const linkRow = document.createElement("div");
            linkRow.className = "message-links";
            links.forEach((link) => {
                const a = document.createElement("a");
                a.textContent = link.label;
                a.href = link.href;
                if (/^https?:\/\//.test(link.href)) {
                    a.target = "_blank";
                    a.rel = "noopener";
                }
                linkRow.appendChild(a);
            });
            wrapper.appendChild(linkRow);
        }
        return wrapper;
    };

    const renderQuickActions = () => {
        messagesEl.querySelectorAll(".message.panel").forEach((node) => node.remove());
        const strings = getStrings(getLang());
        const actions = strings.actions || FALLBACK.actions;
        const panel = document.createElement("div");
        panel.className = "chatbot-actions-panel";
        const keys = ACTIONS_BY_MODE[mode] || ACTIONS_BY_MODE.pro;
        keys.forEach((key) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "chatbot-chip";
            btn.dataset.action = key;
            btn.textContent = actions[key] || key;
            panel.appendChild(btn);
        });
        addMessage("panel", panel);
    };

    const renderWelcome = () => {
        const strings = getStrings(getLang());
        const ui = strings.ui || FALLBACK.ui;
        messagesEl.innerHTML = "";
        addMessage("bot", mode === "fun" ? ui.greetingFun : ui.greetingPro);
        renderQuickActions();
    };

    const buildStackContent = (strings) => {
        const { labels, stack } = strings;
        const wrapper = document.createElement("div");
        const rows = [
            { label: labels.languages, items: stack.languages },
            { label: labels.systems, items: stack.systems },
            { label: labels.ai, items: stack.ai },
            { label: labels.env, items: stack.env },
        ];
        rows.forEach((row) => {
            if (!row.items || !row.items.length) return;
            const p = document.createElement("p");
            const strong = document.createElement("strong");
            strong.textContent = `${row.label}: `;
            p.appendChild(strong);
            p.appendChild(document.createTextNode(row.items.join(" / ")));
            wrapper.appendChild(p);
        });
        return wrapper;
    };

    const buildStackMessage = (strings) => {
        const wrapper = document.createElement("div");
        const title = document.createElement("p");
        title.className = "message-title";
        title.textContent = strings.labels.stackTitle;
        wrapper.appendChild(title);
        wrapper.appendChild(buildStackContent(strings));
        return wrapper;
    };

    const buildContactContent = (strings) => {
        const { labels, contact } = strings;
        const links = [];
        if (contact.email) links.push({ label: `${labels.email}: ${contact.email}`, href: `mailto:${contact.email}` });
        if (contact.phone) links.push({ label: `${labels.phone}: ${contact.phone}`, href: `tel:${contact.phone}` });
        if (contact.github) links.push({ label: labels.github || "GitHub", href: contact.github });
        if (contact.resumePage) links.push({ label: labels.resumePage, href: contact.resumePage });
        if (contact.resumePdf) links.push({ label: labels.resumePdf, href: contact.resumePdf });
        return buildContent({ title: labels.contactTitle, body: strings.profile?.summary || "", links });
    };

    const buildProjectsReply = (strings) => {
        const { labels, projects } = strings;
        const bullets = projects.map((project) => `${project.name}：${project.summary}`);
        const links = projects.map((project) => ({
            label: `${project.name} GitHub`,
            href: project.link,
        }));
        const content = buildContent({
            title: labels.projectsTitle,
            body: labels.projectsIntro,
            bullets,
            links,
        });
        return [{ type: "bot", content }];
    };

    const buildProjectDetail = (strings, id) => {
        const { labels, projects } = strings;
        const project = projects.find((item) => item.id === id);
        if (!project) return buildProjectsReply(strings);
        const footer = `${labels.stackTitle}: ${project.stack.join(" / ")}`;
        const content = buildContent({
            title: `${labels.projectDetail}: ${project.name}`,
            body: project.summary,
            bullets: project.highlights,
            footer,
            links: project.link ? [{ label: `${project.name} GitHub`, href: project.link }] : [],
        });
        return [{ type: "bot", content }];
    };

    const buildResumeReply = (strings) => {
        const { labels, profile } = strings;
        const content = buildContent({ title: labels.resumeTitle, body: profile.summary, bullets: profile.highlights });
        return [{ type: "bot", content }];
    };

    const buildEssaysReply = (strings) => {
        const { labels, essays } = strings;
        const content = buildContent({
            title: labels.essaysTitle,
            body: essays.intro,
            links: [{ label: labels.essaysLink, href: essays.link }],
        });
        return [{ type: "bot", content }];
    };

    const buildHobbiesReply = (strings) => {
        const { labels, hobbies } = strings;
        const bullets = [
            `${labels.cultureTitle}: ${hobbies.culture.slice(0, 2).join(" / ")}`,
            `${labels.historyTitle}: ${hobbies.history.slice(0, 2).join(" / ")}`,
            `${labels.tarotTitle}: ${hobbies.tarot.join(" / ")}`,
        ];
        const content = buildContent({ title: labels.hobbiesTitle, body: hobbies.summary, bullets });
        return [{ type: "bot", content }, { type: "system", content: strings.ui.funDisclaimer }];
    };

    const buildHobbyDetail = (strings, kind) => {
        const { labels, hobbies } = strings;
        const map = {
            culture: { title: labels.cultureTitle, bullets: hobbies.culture },
            history: { title: labels.historyTitle, bullets: hobbies.history },
            tarot: { title: labels.tarotTitle, bullets: hobbies.tarot },
        };
        const info = map[kind];
        if (!info) return buildHobbiesReply(strings);
        const replies = [{ type: "bot", content: buildContent({ title: info.title, bullets: info.bullets }) }];
        if (kind === "tarot") replies.push({ type: "system", content: strings.ui.funDisclaimer });
        return replies;
    };

    const buildGateReply = (strings) => {
        const wrapper = document.createElement("div");
        const p = document.createElement("p");
        p.textContent = strings.ui.proGate;
        wrapper.appendChild(p);
        const button = document.createElement("button");
        button.type = "button";
        button.className = "chatbot-cta";
        button.dataset.action = "mode-fun";
        button.textContent = strings.ui.switchToFun;
        wrapper.appendChild(button);
        return [{ type: "bot", content: wrapper }];
    };

    const buildUnknownReply = (strings) => [{ type: "bot", content: strings.ui.unknown }];

    const getMockReply = (text, intentOverride) => {
        const strings = getStrings(getLang());
        const intent = getIntent(text, intentOverride);

        if (FUN_INTENTS.has(intent) && mode === "pro") {
            return buildGateReply(strings);
        }

        switch (intent) {
            case "projects":
                return buildProjectsReply(strings);
            case "project_storytovideo":
                return buildProjectDetail(strings, "storytovideo");
            case "project_swiftsweep":
                return buildProjectDetail(strings, "swiftsweep");
            case "project_graduation":
                return buildProjectDetail(strings, "graduation");
            case "stack":
                return [{ type: "bot", content: buildStackMessage(strings) }];
            case "resume":
                return buildResumeReply(strings);
            case "contact":
                return [{ type: "bot", content: buildContactContent(strings) }];
            case "essays":
                return buildEssaysReply(strings);
            case "hobbies":
                return buildHobbiesReply(strings);
            case "culture":
                return buildHobbyDetail(strings, "culture");
            case "history":
                return buildHobbyDetail(strings, "history");
            case "tarot":
                return buildHobbyDetail(strings, "tarot");
            default:
                return buildUnknownReply(strings);
        }
    };

    const renderReplies = (replies) => {
        replies.forEach((reply) => {
            const type = reply.type || "bot";
            addMessage(type, reply.content);
        });
        renderQuickActions();
    };

    const fetchBackend = async (text) => {
        const payload = {
            message: text,
            mode,
            lang: getLang(),
        };
        const response = await fetch(apiEndpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
    };

    const renderBackendResponse = (data) => {
        if (!data) return;
        if (Array.isArray(data.messages)) {
            data.messages.forEach((item) => {
                const text = typeof item.text === "string" ? item.text : "";
                const type = item.type || "bot";
                addMessage(type, text);
            });
            return;
        }
        if (typeof data.reply === "string") {
            addMessage("bot", data.reply);
        }
    };

    const sendMessage = async (text, options = {}) => {
        addMessage("user", text);
        inputEl.value = "";
        sendBtn.disabled = true;

        const typing = addMessage("system", "…", { extraClass: "typing" });

        try {
            if (apiEndpoint) {
                const data = await fetchBackend(text);
                typing.remove();
                renderBackendResponse(data);
            } else {
                const replies = getMockReply(text, options.intentOverride);
                typing.remove();
                renderReplies(replies);
            }
        } catch (e) {
            typing.remove();
            const strings = getStrings(getLang());
            addMessage("system", strings.ui.error || FALLBACK.ui.error);
        } finally {
            sendBtn.disabled = false;
            inputEl.focus();
        }
    };

    const handleAction = (action) => {
        if (!action) return;
        const strings = getStrings(getLang());
        if (action === "mode-fun") {
            setMode("fun", { announce: true });
            return;
        }
        if (action === "mode-pro") {
            setMode("pro", { announce: true });
            return;
        }
        if (action === "hobbies" && mode === "pro") {
            setMode("fun", { announce: true });
        }
        const label = (strings.actions && strings.actions[action]) || action;
        sendMessage(label, { intentOverride: ACTION_INTENTS[action] });
    };

    toggleBtn.addEventListener("click", () => toggleChat());
    closeBtn.addEventListener("click", () => toggleChat(false));

    modeButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
            const nextMode = btn.dataset.mode;
            if (nextMode) setMode(nextMode, { announce: true });
        });
    });

    clearBtn.addEventListener("click", () => {
        renderWelcome();
    });

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

    messagesEl.addEventListener("click", (event) => {
        const btn = event.target.closest("[data-action]");
        if (!btn) return;
        handleAction(btn.dataset.action);
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && isOpen) {
            toggleChat(false);
        }
    });

    window.addEventListener("site:lang-change", () => {
        updateStrings();
    });

    const initialize = () => {
        try {
            const saved = localStorage.getItem(MODE_KEY);
            if (saved === "fun" || saved === "pro") mode = saved;
        } catch (e) {
            /* ignore */
        }

        updateModeButtons();
        updateStrings();
        renderWelcome();

        try {
            const savedState = localStorage.getItem(STATE_KEY);
            if (savedState === "true") {
                toggleChat(true);
            }
        } catch (e) {
            /* ignore */
        }
    };

    initialize();
})();
