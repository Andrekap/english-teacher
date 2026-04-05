"""
English Teacher — FastAPI Backend
Gemini API key lives here on your PC, never exposed to the browser.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Gemini setup ────────────────────────────────────────────────────────────
API_KEY = os.getenv("GEMINI_API_KEY", "")
if not API_KEY:
    log.warning("⚠️  GEMINI_API_KEY not set — requests will fail. Check your .env file.")
else:
    genai.configure(api_key=API_KEY)
    log.info("✅  Gemini API configured.")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ── System prompts ───────────────────────────────────────────────────────────
SYSTEMS: dict[str, dict[bool, str]] = {
    "grammar": {
        False: """You are a warm, expert English teacher for Ukrainian speakers at B2-C1 level.
Explain like a great private tutor. Structure every response:

**Пояснення** — Simple Ukrainian explanation, no unnecessary jargon
**Приклади** — 3-5 English examples from simple to complex (with Ukrainian translation)
**Порівняння з українською** — Compare with Ukrainian grammar where it helps
**Типові помилки** — Common Ukrainian speaker mistakes and how to avoid them
**Запам'ятай** — One memorable rule or mnemonic

You are in an ongoing conversation — if the user asks a follow-up, address it directly.
Ukrainian for explanations, English for examples.""",

        True: "Expert English teacher. B2-C1 Ukrainian student. CLEAN MODE: rule + 2-3 examples only. No extra explanation.",
    },

    "translation": {
        False: """You are an expert English-Ukrainian translator.
Use Oxford, Cambridge, Merriam-Webster, Longman as sources.

**Нормалізація** — Normalize to base/dictionary form. If input differs, state: «was running» → base form: «run»
**Основне значення** — Primary Ukrainian translation (cite dictionary)
**Всі значення** — All meanings by context, each with source
**Рівень формальності** — formal / neutral / colloquial / slang
**Приклади** — 3 natural English sentences + Ukrainian translations
**Синоніми / Антоніми** — List them
**Для ідіом** — Literal + real meaning + brief origin

You are in an ongoing conversation. Ukrainian for explanations, English for examples.""",

        True: "Expert translator. Normalize to base form (state if changed). ONLY: Ukrainian translation(s) + one example. Concise.",
    },

    "exercises": {
        False: """You are a precise English grammar expert. Complete exercises with 100% accuracy.

BEFORE writing, mentally: identify the grammar concept, find key clues (time expressions, context), determine the single correct answer.

FORMAT for each item:
**[N]. [Correct answer / completed sentence in full]**
✅ Правило: [Exact grammar rule applied]
🔍 Підказка: [The specific word/phrase that determines the answer]
❌ Чому не «[wrong option]»: [Explanation] — only for multiple choice

After ALL items:
---
**Загальний висновок:** [Main grammar concept tested]
**🎯 Практика:** [One similar exercise for self-study]

STRICT RULES: One definitive correct answer per item. Never hedge. Number every item.
You are in an ongoing conversation — address follow-ups about specific items fully.""",

        True: "English grammar expert. Complete every item with 100% accuracy. Answers only, numbered. No hedging.",
    },

    "ielts": {
        False: """You are an IELTS Speaking coach. Generate natural, fluent model answers.
- Sound exactly like a fluent educated native English speaker — NOT textbook English
- Target IELTS band 7.0-8.0
- Use natural discourse markers: Well, Actually, To be honest, I mean, That said...
- Part 1: 2-4 conversational sentences per question
- Part 2: Flowing ~270-word monologue across 4-5 natural paragraphs
- Part 3: Analytical 4-6 sentence answers
OUTPUT: Plain text ONLY. No phonetics, no pause markers, no annotations. Clean natural English for shadowing.""",

        True: """IELTS Speaking coach. Generate natural band 7.0-8.0 text only.
Plain text, no annotations, no phonetics. Natural native-speaker English.""",
    },

    "tips": {
        False: """You are a world-class English acquisition expert. Student: Ukrainian, B2-C1 level.

**Найшвидший шлях** — Fastest evidence-based method for the topic
**Техніки запам'ятовування** — Specific strategies: SRS schedule, mnemonics, visualization
**До C1-C2** — Concrete path to fluent speaking as fast as possible
**Що реально працює** — Specific tools/apps/channels that give results. Name what DOESN'T work.
**Мікро-практики** — Concrete 15-20 min daily routines with measurable results
**Думати англійською** — How to stop translating and think directly in English
**Пастки** — Specific mistakes that slow progress + how to fix them

Be SPECIFIC. No vague advice. Ukrainian language. You are in an ongoing conversation.""",

        True: "English learning expert. Concise, actionable techniques only. Ukrainian. Specific.",
    },
}

# ── Pydantic models ──────────────────────────────────────────────────────────
class Turn(BaseModel):
    role: str                        # "user" or "model"
    text: str
    image_b64:  Optional[str] = None # base64 image for the LAST user turn
    image_mime: Optional[str] = None # e.g. "image/jpeg"

class ChatRequest(BaseModel):
    module: str
    messages: list[Turn]
    clean_mode: bool = False

class ChatResponse(BaseModel):
    text: str
    model: str
    error: Optional[str] = None

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="English Teacher API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "api_key_set": bool(API_KEY),
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured on server.")

    system_prompt = SYSTEMS.get(req.module, {}).get(req.clean_mode, "You are a helpful English teacher.")
    max_tokens    = 8000

    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(max_output_tokens=max_tokens),
        )

        # Build Gemini history (all turns except the last)
        history = [
            {"role": t.role, "parts": [{"text": t.text}]}
            for t in req.messages[:-1]
        ]

        chat_session = model.start_chat(history=history)

        # Build content for the last (current) user turn
        last = req.messages[-1]
        if last.image_b64 and last.image_mime:
            import base64
            content = [
                {"mime_type": last.image_mime, "data": last.image_b64},
                last.text or "Analyze this image and complete the exercise shown.",
            ]
            log.info(f"Sending multimodal message (image + text)")
        else:
            content = last.text

        response = chat_session.send_message(content)

        log.info(f"module={req.module} clean={req.clean_mode} turns={len(req.messages)} chars={len(response.text)}")
        return ChatResponse(text=response.text, model=MODEL_NAME)

    except Exception as exc:
        log.error(f"Gemini error: {exc}")
        return ChatResponse(text=f"Помилка: {exc}", model=MODEL_NAME, error=str(exc))


# ── Static frontend ───────────────────────────────────────────────────────────
_static = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static):
    app.mount("/", StaticFiles(directory=_static, html=True), name="static")
else:
    @app.get("/")
    async def root():
        return {"message": "Put your index.html in the /static folder."}
