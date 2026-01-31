import asyncio
import json
import math
import os
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class _HTMLToText(HTMLParser):
    """Minimal HTML -> text extractor (no external deps)."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return

        if tag in {"p", "br", "li", "h1", "h2", "h3", "h4", "section", "div"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth = max(self._skip_depth - 1, 0)
            return

        if tag in {"p", "li", "h1", "h2", "h3", "h4"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        raw = " ".join(self._chunks)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _extract_text_from_html(html: str) -> str:
    parser = _HTMLToText()
    parser.feed(html)
    return parser.text()


def _read_text_file(path: Path) -> str:
    data = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in {".html", ".htm"}:
        return _extract_text_from_html(data)
    return data


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, 0)
    return chunks


def _tokenize(text: str) -> List[str]:
    # Mixed tokenizer:
    # - ASCII words/numbers as tokens
    # - CJK characters as tokens for Chinese queries without requiring jieba
    text = text.lower()
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)


class KnowledgeBase:
    def __init__(self, chunks: List[Dict[str, str]]):
        self.chunks = chunks
        self._idf: Dict[str, float] = {}
        self._chunk_tf: List[Dict[str, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        df: Dict[str, int] = {}
        docs_tokens: List[List[str]] = []

        for chunk in self.chunks:
            tokens = _tokenize(chunk.get("text", ""))
            docs_tokens.append(tokens)
            for tok in set(tokens):
                df[tok] = df.get(tok, 0) + 1

        n_docs = max(len(docs_tokens), 1)
        self._idf = {tok: math.log((n_docs + 1) / (count + 1)) + 1.0 for tok, count in df.items()}

        self._chunk_tf = []
        for tokens in docs_tokens:
            tf: Dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self._chunk_tf.append(tf)

    def search(self, query: str, k: int = 6) -> List[Dict[str, str]]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        q_tf: Dict[str, int] = {}
        for tok in q_tokens:
            q_tf[tok] = q_tf.get(tok, 0) + 1

        scored: List[Tuple[float, int]] = []
        for i, tf in enumerate(self._chunk_tf):
            score = 0.0
            for tok, q_count in q_tf.items():
                if tok not in tf:
                    continue
                score += (tf[tok] * self._idf.get(tok, 0.0)) * (1.0 + 0.1 * q_count)
            if score > 0:
                scored.append((score, i))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [self.chunks[i] for _, i in scored[:k]]


def _fetch_url_sync(url: str, timeout_s: float) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "JadeSnow7-chatbot/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            charset = getattr(resp.headers, "get_content_charset", lambda: None)() or "utf-8"
            return resp.read().decode(charset, errors="ignore")
    except Exception:
        return None


async def _fetch_url(url: str, timeout_s: float) -> Optional[str]:
    return await asyncio.to_thread(_fetch_url_sync, url, timeout_s)


async def build_kb() -> KnowledgeBase:
    kb_root = os.getenv("KB_LOCAL_ROOT", "").strip()
    repo_root = Path(kb_root).expanduser().resolve() if kb_root else Path(__file__).resolve().parents[1]
    chunks: List[Dict[str, str]] = []

    # 1) Local sources (preferred when running in the repo).
    local_candidates = [
        repo_root / "index.html",
        repo_root / "resume.html",
        repo_root / "scripts" / "site.js",
        repo_root / "scripts" / "chatbot-kb.js",
    ]

    for path in local_candidates:
        if not path.exists():
            continue
        text = _read_text_file(path)
        for idx, piece in enumerate(_chunk_text(text)):
            chunks.append(
                {
                    "source": str(path.relative_to(repo_root)),
                    "text": piece,
                    "id": f"{path.name}:{idx}",
                }
            )

    # Index essays/notes HTML for richer Q&A (best-effort).
    for folder in ["essays", "notes"]:
        base = repo_root / folder
        if not base.exists():
            continue
        for path in base.rglob("*.html"):
            try:
                text = _read_text_file(path)
            except Exception:
                continue
            for idx, piece in enumerate(_chunk_text(text)):
                chunks.append(
                    {
                        "source": str(path.relative_to(repo_root)),
                        "text": piece,
                        "id": f"{path.name}:{idx}",
                    }
                )

    # 2) Remote sources (optional / for server deployment without the repo).
    # Enabled by KB_SOURCE=remote or KB_SOURCE=auto when local site isn't present.
    kb_source = os.getenv("KB_SOURCE", "auto").lower()
    want_remote = kb_source == "remote" or (kb_source == "auto" and not (repo_root / "index.html").exists())
    include_github = os.getenv("KB_INCLUDE_GITHUB", "1").lower() not in {"0", "false", "no"}
    github_urls = [
        "https://raw.githubusercontent.com/JadeSnow7/StoryToVideo/main/README.md",
        "https://raw.githubusercontent.com/JadeSnow7/SwiftSweep/main/README.md",
        "https://raw.githubusercontent.com/JadeSnow7/graduationDesign/main/README.md",
    ]

    fetch_urls: List[str] = []
    if want_remote:
        site_base = os.getenv("SITE_URL_BASE", "https://jadesnow7.github.io").rstrip("/")
        fetch_urls.extend(
            [
                f"{site_base}/index.html",
                f"{site_base}/resume.html",
                f"{site_base}/scripts/site.js",
            ]
        )
    if include_github:
        fetch_urls.extend(github_urls)

    if fetch_urls:
        timeout_s = float(os.getenv("KB_REMOTE_TIMEOUT", "10"))
        for url in fetch_urls:
            body = await _fetch_url(url, timeout_s)
            if not body:
                continue
            text = _extract_text_from_html(body) if url.endswith(".html") else body
            for idx, piece in enumerate(_chunk_text(text)):
                chunks.append({"source": url, "text": piece, "id": f"{url}:{idx}"})

    if not chunks:
        chunks.append(
            {
                "source": "kb:fallback",
                "text": "Portfolio knowledge base is empty. Provide safe, minimal answers.",
                "id": "fallback:0",
            }
        )

    return KnowledgeBase(chunks)


class ChatRequest(BaseModel):
    message: str
    mode: Optional[str] = "pro"
    lang: Optional[str] = "zh"


class ChatResponse(BaseModel):
    reply: str


def _lang_is_en(lang: Optional[str]) -> bool:
    return (lang or "").lower().startswith("en")


def _build_system_prompt(lang: Optional[str], mode: str) -> str:
    if _lang_is_en(lang):
        base = (
            "You are a resume assistant for Hu Aodong. "
            "Answer questions about his projects, tech stack, essays, and contact info using the provided context. "
            "If the context is insufficient, say you are not sure and suggest checking the resume page or GitHub links. "
            "Keep answers concise and professional."
        )
        if mode == "fun":
            base += (
                " Interests mode: you may chat about hobbies (traditional culture, worldbuilding, astrology/tarot). "
                "For astrology/tarot, clearly state it's for fun only and not for real-life decisions."
            )
        else:
            base += (
                " Pro mode: prioritize interview-relevant info (projects, stack, impact). "
                "If asked about hobbies, suggest switching to Fun mode."
            )
        return base

    base = (
        "你是胡傲东的简历助手。请基于提供的上下文回答有关他的项目、技术栈、随笔与联系方式的问题。"
        "如果上下文不足，请明确说明不确定，并建议查看在线简历或 GitHub。"
        "回答要简洁、专业、可验证。"
    )
    if mode == "fun":
        base += "兴趣模式：可以聊传统文化、架空历史、星盘/塔罗；星盘/塔罗必须强调仅供娱乐，不提供现实决策建议。"
    else:
        base += "专业模式：优先回答应聘相关内容（项目、技术栈、产出/指标）；若用户聊兴趣，请建议切换到兴趣模式。"
    return base


def _build_user_prompt(message: str, mode: str, lang: Optional[str], contexts: List[Dict[str, str]]) -> str:
    header = (
        f"User message: {message}\nMode: {mode}\nLanguage: {lang}\n\n"
        if _lang_is_en(lang)
        else f"用户问题：{message}\n模式：{mode}\n语言：{lang}\n\n"
    )

    ctx_lines: List[str] = []
    for i, chunk in enumerate(contexts, start=1):
        src = chunk.get("source", "")
        text = chunk.get("text", "")
        ctx_lines.append(f"[{i}] Source: {src}\n{text}")

    ctx_block = "\n\n".join(ctx_lines) if ctx_lines else "(no context)"
    if _lang_is_en(lang):
        return header + "Context:\n" + ctx_block + "\n\nPlease answer based on the context."
    return header + "上下文：\n" + ctx_block + "\n\n请基于上下文作答。"


import ollama

async def _ollama_chat(messages: List[Dict[str, str]]) -> str:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = os.getenv("OLLAMA_MODEL", "gemini-3-flash-preview:cloud")
    timeout_s = float(os.getenv("OLLAMA_TIMEOUT", "60"))

    # Configure ollama client host if needed (the library uses OLLAMA_HOST env var by default)
    # But strictly speaking, the library might look for different env vars or we set it globally.
    # The 'ollama' library respects OLLAMA_HOST environment variable.
    
    # We will wrap the synchronous library call in a thread 
    # because the official library is currently synchronous (as per user snippet).
    # Note: user snippet used 'from ollama import chat', we used 'import ollama' top level.
    
    def _call() -> str:
        try:
            # client = ollama.Client(host=host) # If we needed explicit host
            # But relying on env var is standard for the lib.
            response = ollama.chat(
                model=model,
                messages=messages,
                options={"temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))},
            )
            # The library returns an object where message.content is the string
            return response['message']['content']
        except Exception as e:
            raise RuntimeError(f"ollama lib error: {e}") from e

    return await asyncio.to_thread(_call)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_KB: Optional[KnowledgeBase] = None
_KB_LOCK = asyncio.Lock()


async def _get_kb() -> KnowledgeBase:
    global _KB
    if _KB is not None:
        return _KB
    async with _KB_LOCK:
        if _KB is None:
            _KB = await build_kb()
    return _KB


@app.on_event("startup")
async def _startup() -> None:
    # Warm the KB at startup so first request is fast.
    try:
        await _get_kb()
    except Exception:
        pass


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    mode = (request.mode or "pro").lower()
    if mode not in {"pro", "fun"}:
        mode = "pro"
    lang = request.lang or "zh"

    kb = await _get_kb()
    contexts = kb.search(request.message, k=int(os.getenv("KB_TOP_K", "6")))

    system_prompt = _build_system_prompt(lang, mode)
    user_prompt = _build_user_prompt(request.message, mode, lang, contexts)

    try:
        reply = await _ollama_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as e:
        if _lang_is_en(lang):
            hint = (
                "Ollama request failed. Make sure Ollama is running and the model is pulled. "
                "Example: `ollama pull gemini-3-flash-preview:cloud`"
            )
        else:
            hint = "Ollama 调用失败：请确认已启动 Ollama 且已拉取模型，例如 `ollama pull gemini-3-flash-preview:cloud`。"
        reply = f"{hint}\n({type(e).__name__}: {e})"

    return ChatResponse(reply=reply)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "kbReady": _KB is not None}


@app.get("/api/health")
async def api_health() -> Dict[str, Any]:
    return await health()
