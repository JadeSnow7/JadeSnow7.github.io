(() => {
  const root = document.documentElement;
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const prefersCoarsePointer = window.matchMedia("(pointer: coarse)").matches;

  const getLang = () => {
    const raw = (window.__SITE_LANG__ || root.lang || "").toLowerCase();
    return raw.startsWith("en") ? "en" : "zh";
  };

  const ABILITY_LABELS = {
    zh: {
      arch: "架构设计",
      performance: "性能优化",
      "cross-platform": "跨平台交付",
      "ai-product": "AI 工程化",
      fullstack: "全栈交付",
      platform: "平台化/工具化",
    },
    en: {
      arch: "Architecture",
      performance: "Performance",
      "cross-platform": "Cross-platform",
      "ai-product": "AI Productization",
      fullstack: "Full-stack Delivery",
      platform: "Platform & Tooling",
    },
  };

  const PROJECT_DETAILS = {
    storytovideo: {
      eyebrow: {
        zh: "字节剪映训练营 · 小组长（客户端/多模态） · 2025.11 - 至今",
        en: "ByteDance CapCut Camp · Team Lead (Client/Multimodal) · 2025.11 - Present",
      },
      title: { zh: "StoryToVideo", en: "StoryToVideo" },
      subtitle: {
        zh: "跨平台 AI 视频生成系统（文本 → 分镜 → 图像/配音/视频）",
        en: "Cross-platform AI video generation pipeline (text → storyboard → image/voice/video)",
      },
      abilities: ["arch", "performance", "cross-platform", "ai-product"],
      highlights: {
        zh: [
          "任务网关 + 优先级队列，支持暂停/恢复/取消/重试，提升任务鲁棒性",
          "MVVM 分层设计（ViewModel/Network/Data 解耦），可测试性与复用性更强",
          "8GB 显存下稳定生成 30s 720p，渲染帧率提升到 60fps",
          "Qt/QML 跨平台 UI，支持实时编辑、预览与导出",
        ],
        en: [
          "Task gateway + priority queue enabling pause/resume/cancel/retry for robustness",
          "MVVM layers (ViewModel/Network/Data) to improve testability and reuse",
          "Stable 30s 720p generation on 8GB VRAM with render FPS up to 60",
          "Qt/QML cross-platform UI with real-time edit, preview, and export",
        ],
      },
      metrics: {
        zh: [
          { label: "生成能力", value: "30s 720p" },
          { label: "缓存复用", value: "+40%" },
          { label: "渲染帧率", value: "60fps" },
        ],
        en: [
          { label: "Generation", value: "30s 720p" },
          { label: "Cache reuse", value: "+40%" },
          { label: "Render FPS", value: "60fps" },
        ],
      },
      media: [
        {
          src: "assets/project-storytovideo-ui.svg",
          alt: { zh: "客户端界面截图（占位）", en: "Client UI screenshot (placeholder)" },
          caption: { zh: "客户端界面（占位）", en: "Client UI (placeholder)" },
        },
        {
          src: "assets/project-storytovideo-arch.svg",
          alt: { zh: "任务流水线与模型架构图（占位）", en: "Task pipeline & model architecture (placeholder)" },
          caption: { zh: "任务流水线与模型架构（占位）", en: "Task pipeline & model architecture (placeholder)" },
        },
      ],
      links: [{ label: { zh: "GitHub", en: "GitHub" }, href: "https://github.com/JadeSnow7/StoryToVideo" }],
    },
    swiftsweep: {
      eyebrow: { zh: "个人开源 · 2025.12 - 至今", en: "Open Source (Solo) · 2025.12 - Present" },
      title: { zh: "SwiftSweep", en: "SwiftSweep" },
      subtitle: { zh: "原生 macOS 系统优化工具（GUI + CLI）", en: "Native macOS optimization tool (GUI + CLI)" },
      abilities: ["arch", "performance", "platform"],
      highlights: {
        zh: [
          "统一清理链路：dry-run 预览、删除计划、审计日志",
          "XPC Helper 权限隔离 + allowlist + symlink escape 防护",
          "插件系统与相似媒体检测（pHash + LSH）",
          "SwiftUI + SwiftPM 模块化，CLI/GUI 共享核心逻辑",
        ],
        en: [
          "Unified cleanup pipeline: dry-run preview, deletion plan, audit logs",
          "Privilege isolation via XPC helper + allowlist + symlink escape protection",
          "Plugin system and similar-media detection (pHash + LSH)",
          "SwiftUI + SwiftPM modules with shared core for CLI/GUI",
        ],
      },
      metrics: {
        zh: [
          { label: "功能模块", value: "14+" },
          { label: "Insights 规则", value: "9 类" },
          { label: "发布方式", value: "Developer ID" },
        ],
        en: [
          { label: "Modules", value: "14+" },
          { label: "Insights", value: "9 categories" },
          { label: "Release", value: "Developer ID" },
        ],
      },
      media: [
        {
          src: "assets/project-swiftsweep-ui.svg",
          alt: { zh: "SwiftSweep 界面截图（占位）", en: "SwiftSweep UI screenshot (placeholder)" },
          caption: { zh: "SwiftUI 视觉界面（占位）", en: "SwiftUI interface (placeholder)" },
        },
        {
          src: "assets/project-swiftsweep-arch.svg",
          alt: { zh: "系统架构图（占位）", en: "System architecture diagram (placeholder)" },
          caption: { zh: "任务链路与插件架构（占位）", en: "Task pipeline & plugin architecture (placeholder)" },
        },
      ],
      links: [{ label: { zh: "GitHub", en: "GitHub" }, href: "https://github.com/JadeSnow7/SwiftSweep" }],
    },
    graduation: {
      eyebrow: { zh: "本科毕业设计 · 2025.11 - 至今", en: "Undergraduate Thesis · 2025.11 - Present" },
      title: { zh: "AI 智能教学平台", en: "AI-powered Teaching Platform" },
      subtitle: {
        zh: "多端教学平台：桌面端 / H5 / iOS / Android",
        en: "Multi-client teaching platform: desktop / H5 / iOS / Android",
      },
      abilities: ["fullstack", "ai-product", "cross-platform", "platform"],
      highlights: {
        zh: [
          "全栈独立交付：React 前端 + Go 后端 + Python AI 服务",
          "Qwen3 + GraphRAG 构建知识图谱与个性化答疑/批改",
          "Docker Compose 一键部署，JWT + RBAC 权限体系",
          "学生画像与长期学情轨迹，支撑课程试点",
        ],
        en: [
          "Full-stack delivery: React FE + Go BE + Python AI services",
          "Qwen3 + GraphRAG knowledge graph for personalized Q&A and grading",
          "One-command Docker Compose deployment with JWT + RBAC",
          "Student profiles and longitudinal learning trajectories for pilot courses",
        ],
      },
      metrics: {
        zh: [
          { label: "并发支持", value: "100+" },
          { label: "答疑准确率", value: "85%+" },
          { label: "多端覆盖", value: "4 类客户端" },
        ],
        en: [
          { label: "Concurrency", value: "100+" },
          { label: "Q&A accuracy", value: "85%+" },
          { label: "Clients", value: "4 platforms" },
        ],
      },
      media: [
        {
          src: "assets/project-graduation-ui.svg",
          alt: { zh: "教学平台界面截图（占位）", en: "Teaching platform UI (placeholder)" },
          caption: { zh: "教学平台界面（占位）", en: "Teaching platform UI (placeholder)" },
        },
        {
          src: "assets/project-graduation-arch.svg",
          alt: { zh: "系统架构图（占位）", en: "System architecture (placeholder)" },
          caption: { zh: "微服务 + AI 能力架构（占位）", en: "Microservices + AI architecture (placeholder)" },
        },
      ],
      links: [{ label: { zh: "GitHub", en: "GitHub" }, href: "https://github.com/JadeSnow7/graduationDesign" }],
    },
  };

  const renderAbilityTags = (container, tags, lang) => {
    const dict = ABILITY_LABELS[lang] || ABILITY_LABELS.zh;
    container.innerHTML = "";
    tags.forEach((tag) => {
      const label = dict[tag];
      if (!label) return;
      const chip = document.createElement("span");
      chip.className = "ability-tag";
      chip.textContent = label;
      container.appendChild(chip);
    });
  };

  const renderCardAbilities = (lang) => {
    document.querySelectorAll("[data-ability-tags]").forEach((container) => {
      const card = container.closest("[data-project-card]");
      if (!card) return;
      const tags = (card.dataset.tags || "")
        .split(",")
        .map((tag) => tag.trim())
        .filter(Boolean);
      renderAbilityTags(container, tags, lang);
    });
  };

  const setupFilter = () => {
    const buttons = Array.from(document.querySelectorAll("[data-filter]"));
    const cards = Array.from(document.querySelectorAll("[data-project-card]"));
    if (!buttons.length || !cards.length) return;

    const applyFilter = (filter) => {
      buttons.forEach((button) => {
        const isActive = button.dataset.filter === filter;
        button.classList.toggle("is-active", isActive);
        button.setAttribute("aria-pressed", isActive ? "true" : "false");
      });

      cards.forEach((card) => {
        const tags = (card.dataset.tags || "")
          .split(",")
          .map((tag) => tag.trim())
          .filter(Boolean);
        const match = filter === "all" || tags.includes(filter);
        card.style.display = match ? "" : "none";
      });
    };

    buttons.forEach((button) => {
      button.addEventListener("click", () => applyFilter(button.dataset.filter || "all"));
    });

    applyFilter("all");
  };

  const setupScrollProgress = () => {
    const bar = document.querySelector(".scroll-progress__bar");
    if (!bar) return;
    let ticking = false;

    const update = () => {
      const scrollTop = window.scrollY || document.documentElement.scrollTop;
      const height = document.documentElement.scrollHeight - window.innerHeight;
      const ratio = height > 0 ? scrollTop / height : 0;
      bar.style.transform = `scaleX(${Math.min(Math.max(ratio, 0), 1)})`;
      ticking = false;
    };

    const onScroll = () => {
      if (!ticking) {
        window.requestAnimationFrame(update);
        ticking = true;
      }
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    update();
  };

  const setupScrollSpy = () => {
    const navLinks = Array.from(document.querySelectorAll('.nav a[href^="#"]'));
    if (!navLinks.length) return;

    const linkMap = new Map(
      navLinks.map((link) => [link.getAttribute("href")?.slice(1), link])
    );

    const sections = Array.from(document.querySelectorAll("main section[id]")).filter((section) =>
      linkMap.has(section.id)
    );

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          linkMap.forEach((link, id) => {
            link.classList.toggle("is-active", id === entry.target.id);
          });
        });
      },
      { rootMargin: "-40% 0px -50% 0px", threshold: 0.2 }
    );

    sections.forEach((section) => observer.observe(section));
  };

  const setupReveal = () => {
    const revealElements = Array.from(document.querySelectorAll("[data-reveal]"));
    if (!revealElements.length) return;
    if (prefersReducedMotion || !("IntersectionObserver" in window)) {
      revealElements.forEach((el) => el.classList.add("is-revealed"));
      return;
    }
    const observer = new IntersectionObserver(
      (entries, obs) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-revealed");
            obs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15 }
    );
    revealElements.forEach((el) => observer.observe(el));
  };

  const setupSpotlight = () => {
    const spotlight = document.querySelector(".spotlight");
    if (!spotlight || prefersReducedMotion || prefersCoarsePointer) return;
    let pending = false;
    let lastEvent = null;

    const update = () => {
      if (!lastEvent) return;
      const x = lastEvent.clientX / window.innerWidth;
      const y = lastEvent.clientY / window.innerHeight;
      root.style.setProperty("--spot-x", `${Math.round(x * 100)}%`);
      root.style.setProperty("--spot-y", `${Math.round(y * 100)}%`);
      root.style.setProperty("--parallax-x", `${(x - 0.5) * 18}px`);
      root.style.setProperty("--parallax-y", `${(y - 0.5) * 18}px`);
      pending = false;
    };

    const onMove = (event) => {
      lastEvent = event;
      if (!pending) {
        pending = true;
        window.requestAnimationFrame(update);
      }
    };

    window.addEventListener("pointermove", onMove, { passive: true });
    window.addEventListener("pointerleave", () => {
      root.style.setProperty("--spot-x", "50%");
      root.style.setProperty("--spot-y", "30%");
      root.style.setProperty("--parallax-x", "0px");
      root.style.setProperty("--parallax-y", "0px");
    });
  };

  const setupCardTilt = () => {
    if (prefersReducedMotion || prefersCoarsePointer) return;
    const cards = Array.from(document.querySelectorAll("[data-project-card]"));
    cards.forEach((card) => {
      let ticking = false;
      card.addEventListener("pointermove", (event) => {
        const rect = card.getBoundingClientRect();
        const x = (event.clientX - rect.left) / rect.width - 0.5;
        const y = (event.clientY - rect.top) / rect.height - 0.5;
        if (!ticking) {
          ticking = true;
          window.requestAnimationFrame(() => {
            card.style.setProperty("--tilt-x", `${(-y * 6).toFixed(2)}deg`);
            card.style.setProperty("--tilt-y", `${(x * 6).toFixed(2)}deg`);
            ticking = false;
          });
        }
      });
      card.addEventListener("pointerleave", () => {
        card.style.setProperty("--tilt-x", "0deg");
        card.style.setProperty("--tilt-y", "0deg");
      });
    });
  };

  const setupDialog = () => {
    const dialog = document.querySelector("[data-project-dialog]");
    if (!dialog) return;

    const titleEl = dialog.querySelector("[data-dialog-title]");
    const subtitleEl = dialog.querySelector("[data-dialog-subtitle]");
    const eyebrowEl = dialog.querySelector("[data-dialog-eyebrow]");
    const tagsEl = dialog.querySelector("[data-dialog-tags]");
    const highlightsEl = dialog.querySelector("[data-dialog-highlights]");
    const metricsEl = dialog.querySelector("[data-dialog-metrics]");
    const mediaEl = dialog.querySelector("[data-dialog-media]");
    const linksEl = dialog.querySelector("[data-dialog-links]");
    const closeButton = dialog.querySelector("[data-dialog-close]");
    let lastActive = null;

    const renderDialog = (projectId, lang) => {
      const details = PROJECT_DETAILS[projectId];
      if (!details) return;
      dialog.dataset.projectId = projectId;

      if (eyebrowEl) eyebrowEl.textContent = details.eyebrow?.[lang] || "";
      if (titleEl) titleEl.textContent = details.title?.[lang] || "";
      if (subtitleEl) subtitleEl.textContent = details.subtitle?.[lang] || "";

      if (tagsEl) renderAbilityTags(tagsEl, details.abilities || [], lang);

      if (highlightsEl) {
        highlightsEl.innerHTML = "";
        (details.highlights?.[lang] || []).forEach((text) => {
          const li = document.createElement("li");
          li.textContent = text;
          highlightsEl.appendChild(li);
        });
      }

      if (metricsEl) {
        metricsEl.innerHTML = "";
        (details.metrics?.[lang] || []).forEach((metric) => {
          const card = document.createElement("div");
          card.className = "metric-card";
          const label = document.createElement("span");
          label.textContent = metric.label;
          const value = document.createElement("strong");
          value.textContent = metric.value;
          card.append(label, value);
          metricsEl.appendChild(card);
        });
      }

      if (mediaEl) {
        mediaEl.innerHTML = "";
        (details.media || []).forEach((item) => {
          const figure = document.createElement("figure");
          const img = document.createElement("img");
          img.src = item.src;
          img.alt = item.alt?.[lang] || "";
          img.loading = "lazy";
          const caption = document.createElement("figcaption");
          caption.textContent = item.caption?.[lang] || "";
          figure.append(img, caption);
          mediaEl.appendChild(figure);
        });
      }

      if (linksEl) {
        linksEl.innerHTML = "";
        (details.links || []).forEach((link) => {
          const a = document.createElement("a");
          a.className = "dialog-link";
          a.href = link.href;
          a.target = "_blank";
          a.rel = "noopener";
          a.textContent = link.label?.[lang] || link.href;
          linksEl.appendChild(a);
        });
      }

      if (closeButton) {
        closeButton.setAttribute("aria-label", lang === "en" ? "Close dialog" : "关闭弹窗");
      }
    };

    const cleanupDialog = () => {
      document.body.classList.remove("is-dialog-open");
      if (lastActive && typeof lastActive.focus === "function") {
        lastActive.focus();
      }
    };

    const openDialog = (projectId) => {
      const lang = getLang();
      renderDialog(projectId, lang);
      lastActive = document.activeElement;
      document.body.classList.add("is-dialog-open");
      if (typeof dialog.showModal === "function") {
        dialog.showModal();
      } else {
        dialog.setAttribute("open", "");
      }
    };

    const closeDialog = () => {
      if (typeof dialog.close === "function") {
        dialog.close();
      } else {
        dialog.removeAttribute("open");
        cleanupDialog();
      }
    };

    if (closeButton) closeButton.addEventListener("click", closeDialog);
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) closeDialog();
    });

    dialog.addEventListener("close", cleanupDialog);

    document.querySelectorAll("[data-project-card]").forEach((card) => {
      const projectId = card.dataset.projectId;
      if (!projectId) return;
      card.addEventListener("click", (event) => {
        if (event.target.closest("a")) return;
        openDialog(projectId);
      });
      card.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openDialog(projectId);
        }
      });
    });

    window.addEventListener("site:lang-change", (event) => {
      const lang = event.detail?.lang || getLang();
      if (dialog.open || dialog.hasAttribute("open")) {
        renderDialog(dialog.dataset.projectId, lang);
      }
    });
  };

  const setupCopy = () => {
    const toast = document.querySelector("[data-toast]");
    const buttons = Array.from(document.querySelectorAll("[data-copy]"));
    if (!toast || !buttons.length) return;

    let toastTimer = null;
    const messages = {
      zh: { phone: "已复制手机号", email: "已复制邮箱", default: "已复制" },
      en: { phone: "Phone copied", email: "Email copied", default: "Copied" },
    };

    const showToast = (message) => {
      toast.textContent = message;
      toast.classList.add("is-visible");
      if (toastTimer) window.clearTimeout(toastTimer);
      toastTimer = window.setTimeout(() => {
        toast.classList.remove("is-visible");
      }, 2000);
    };

    const copyText = async (value) => {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(value);
        return true;
      }
      const input = document.createElement("input");
      input.value = value;
      document.body.appendChild(input);
      input.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(input);
      return ok;
    };

    buttons.forEach((button) => {
      button.addEventListener("click", async () => {
        const value = button.dataset.copyValue || "";
        if (!value) return;
        const lang = getLang();
        const type = button.dataset.copyType || "default";
        try {
          await copyText(value);
          showToast((messages[lang] && messages[lang][type]) || messages[lang].default);
        } catch {
          showToast((messages[lang] && messages[lang].default) || "Copied");
        }
      });
    });
  };

  const init = () => {
    const lang = getLang();
    renderCardAbilities(lang);
    setupFilter();
    setupScrollProgress();
    setupScrollSpy();
    setupReveal();
    setupSpotlight();
    setupCardTilt();
    setupDialog();
    setupCopy();
  };

  window.addEventListener("site:lang-change", (event) => {
    const lang = event.detail?.lang || getLang();
    renderCardAbilities(lang);
  });

  init();
})();
