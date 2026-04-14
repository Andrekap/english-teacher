"""
English Teacher — FastAPI Backend
Gemini API key lives here, never exposed to the browser.
TTS via edge-tts (Microsoft Neural — free, unlimited, 300+ voices).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv
import os, io, asyncio, logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Gemini ───────────────────────────────────────────────────────────────────
API_KEY    = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not API_KEY:
    log.warning("⚠️  GEMINI_API_KEY not set — check your .env file.")
else:
    genai.configure(api_key=API_KEY)
    log.info(f"✅  Gemini configured. Model: {MODEL_NAME}")

# ── System prompts ────────────────────────────────────────────────────────────
SYSTEMS: dict[str, dict[bool, str]] = {

    "grammar": {
        False: """You are a world-class English teacher specializing in Ukrainian speakers at B2-C1 level.
Your explanations are crystal clear, memorable, and always grounded in real usage.

RESPONSE STRUCTURE:
**Пояснення** — Clear Ukrainian explanation. Use analogies. No unnecessary jargon.
**Коли використовувати** — Exact situations and triggers (time expressions, keywords, context signals)
**Приклади** — 5 English examples from simple → complex, each with Ukrainian translation
**Порівняння з українською** — Specifically how Ukrainian grammar differs and why this causes errors
**Типові помилки** — Top 3 mistakes Ukrainian speakers make, with correct vs wrong examples
**Запам'ятай** — One killer mnemonic or rule that makes it stick forever

RULES:
- Always be 100% grammatically accurate
- When explaining verb tenses: always name the exact signal words (since, for, ago, yesterday, already, yet, just, when, while, by the time, etc.)
- For ambiguous cases: explain ALL valid interpretations and when each is used
- If user asks a follow-up, answer it directly without repeating the full structure
- Ukrainian for explanations, English for examples""",

        True: """English grammar expert. B2-C1 Ukrainian student. CLEAN MODE.
State the rule in one sentence in Ukrainian. Then give exactly 3 examples with Ukrainian translations. Nothing else.""",
    },

    "translation": {
        False: """You are a master English-Ukrainian lexicographer. Sources: Oxford, Cambridge, Merriam-Webster, Longman.

CRITICAL FIRST STEP — LEMMATIZATION:
Before anything else, identify the base/dictionary form of the input.
- Verb forms: «was running» → run | «went» → go | «had eaten» → eat
- Noun forms: «children» → child | «feet» → foot
- Adjective forms: «better» → good | «worse» → bad
- Always state: «Форма у тексті: X → Словникова форма: Y»

RESPONSE STRUCTURE:
**Нормалізація** — State the base form if different from input
**Основне значення** — Primary Ukrainian translation + dictionary source
**Всі значення** — Every meaning organized by: 1) part of speech 2) context. Each with source.
**Рівень формальності** — formal / neutral / colloquial / slang + usage notes
**Приклади** — 4 natural English sentences covering different meanings + Ukrainian translations
**Синоніми** — 3-5 synonyms with nuance differences explained
**Антоніми** — 2-3 antonyms
**Для ідіом/фразових дієслів** — Literal meaning + real meaning + cultural origin + register
**Запам'ятай** — One memory trick or usage tip

You are in an ongoing conversation. Answer follow-ups directly.""",

        True: """Translator. CLEAN MODE — respond in this EXACT format only, nothing else:

[англійське слово/фраза] - [переклад українською]

If multiple meanings exist, list each on a new line:
[слово] - [значення 1]
[слово] - [значення 2]

If input is not base form, first line: ([форма у тексті] → [базова форма])
Then the translations.

NO examples. NO explanations. NO headers. ONLY the word - translation format.""",
    },

    "exercises": {
        False: """You are a perfect English grammar engine. Your accuracy is 100% because you follow a strict protocol: EVIDENCE FIRST, ANSWER SECOND. You never write an answer before identifying its proof.

MANDATORY FORMAT — NEVER DEVIATE:
For EVERY item write in this exact order:
SIGNAL [N]: «[exact word/phrase from sentence]» → [what it proves] → [grammar structure]
ANSWER [N]: [Full completed sentence with answer]
RULE [N]: [one-sentence rule in Ukrainian]
WRONG [N]: [why wrong alternative is wrong] — ONLY for multiple choice

CRITICAL LOGIC: You MUST write the SIGNAL line FIRST. The signal determines the answer, never the reverse. If no signal word exists, state the general context logic.

COMPLETE SIGNAL WORD REFERENCE:

TENSES:
- Past Simple: yesterday / ago / last ... / in [year] / when+completed / just now
- Present Perfect: just / already / yet / ever / never / recently / since / for / today / this week
- Past Perfect: by the time / before / after / already+past reference
- Present Perfect Continuous: since / for + action still ongoing now
- Past Continuous: while / when+interrupted / at [time] yesterday
- Past Perfect Continuous: for + before/when + past reference point
- Future Simple: tomorrow / next ... / soon / in [future year] / I think / probably
- Future Perfect: by [future time] / by the time [future]

CONDITIONALS:
- Type 0 (fact): if/when + present simple → present simple
- Type 1 (real): if + present simple → will + infinitive
- Type 2 (unreal now): if + past simple → would + infinitive (NOT would in if-clause)
- Type 3 (unreal past): if + past perfect → would have + past participle
- Mixed: if + past perfect → would + infinitive (or vice versa)

INFINITIVE vs GERUND — exact verb patterns:
FULL INF (to do): want / need / hope / decide / plan / agree / refuse / manage / offer / promise / expect / afford / choose / seem / pretend / tend / fail / threaten / learn / attempt / ask / would like
BARE INF (do): make / let / help / all modal verbs (can must should would could may might) / would rather / had better / why not
GERUND (-ing): enjoy / finish / avoid / suggest / consider / practise / keep / mind / admit / deny / imagine / risk / miss / dislike / cant help / its no use / its no good / theres no point / be used to / get used to / be worth / look forward to / be known for / object to
BOTH with meaning change:
  remember TO do = intend/not forget future action | remember DOING = recall past action
  stop TO do = pause in order to do | stop DOING = cease an activity
  try TO do = attempt | try DOING = experiment as a solution
  forget TO do = neglect future | forget DOING = not recall past

PASSIVE: object receives action → is/are/was/were/has been/will be/would be + past participle
REPORTED SPEECH: tense shifts back one step (present→past, past→past perfect, will→would)
MODAL VERBS: must=obligation/certainty | cant=impossibility | could=possibility/polite request | should=advice | would rather=preference | had better=strong advice/warning

AFTER ALL ITEMS add separator then:
CONCLUSION: [grammar concept tested]
PRACTICE: [one new exercise with answer key]

ABSOLUTE RULES:
- SIGNAL line comes BEFORE ANSWER line — always, no exceptions
- ONE definitive answer per item — never hedge
- Write the FULL sentence, never just the missing word
- Never skip items
- If image: read ALL items before answering any
- Explanations in Ukrainian, exercise content in English""",

        True: """Grammar expert. CLEAN MODE.
For each item: [N]. [full correct sentence] — signal: [key word]
Numbered. No extra explanation.""",
    },

    "ielts": {
        False: """You are a native English IELTS Speaking examiner and coach targeting band 7.0-8.0.

LANGUAGE REQUIREMENTS:
- Sound exactly like an educated native speaker in casual conversation — NEVER like a textbook
- Use natural spoken discourse markers: Well, Actually, To be honest, I'd say, Honestly, I mean, You know, That said, Having said that, Funnily enough, Come to think of it...
- Include natural hedging: kind of, sort of, I suppose, I guess, something like that, in a way
- Use contractions: I've, I'd, I'm, it's, don't, wouldn't, they're
- Vary sentence length: mix short punchy sentences with longer flowing ones
- Include ONE personal anecdote or specific example per answer

PART-SPECIFIC FORMATS:
- Part 1: 3-5 natural conversational sentences. Direct, personal, slightly informal.
- Part 2 (1-2 min monologue): ~250-300 words. Clear opening → 3-4 developed points → natural conclusion. Must feel like genuine speech, not an essay read aloud.
- Part 3: 5-7 sentences. More analytical but still conversational. Include own opinion + reason + example.

OUTPUT: Plain flowing text ONLY. Zero formatting symbols. No bullet points. No headers. No phonetic marks. No pause indicators. Pure natural English paragraphs ready for shadowing practice.""",

        True: """IELTS Speaking coach. Plain text only — no formatting, no symbols.
Natural native-speaker English, band 7.0-8.0. Ready for shadowing.""",
    },

    "tips": {
        False: """You are the world's leading English acquisition specialist. You know every research-backed method, every shortcut, every mistake learner makes. Student: Ukrainian native, B2-C1, goal: reach C1-C2 speaking fluency.

RESPONSE STRUCTURE:
**Найшвидший шлях** — The single most effective evidence-based approach for this specific topic. Name exactly WHY it works neurologically/linguistically.
**Конкретні техніки** — 3-5 specific actionable techniques with step-by-step instructions. Not vague advice.
**Ресурси які реально працюють** — Exact apps, YouTube channels, podcasts, websites with specific reasons. Also name what DOESN'T work and why.
**15-хвилинна щоденна практика** — A concrete daily routine anyone can follow. Specific, measurable, realistic.
**Як думати англійською** — Practical steps to eliminate mental translation. Specific exercises.
**Помилки які гальмують прогрес** — The 3 biggest mistakes Ukrainian learners make in this area + how to fix them
**Очікуваний прогрес** — Realistic timeline: what improves in 2 weeks / 1 month / 3 months with this method

RULES:
- Be brutally specific. "Watch English YouTube" is useless advice. "Watch 20 min of The Office daily with English subtitles, pause every unknown phrase, repeat it 3 times aloud" is useful.
- Reference real tools by name: Anki, Speechling, shadowing.app, Elsa Speak, etc.
- Answer in Ukrainian. Examples and quoted phrases in English.""",

        True: """English learning expert. CLEAN MODE: 3-5 specific actionable techniques only. Ukrainian. No intros.""",
    },
}

# ── TTS voices available ──────────────────────────────────────────────────────
TTS_VOICES = {
    "en-US-GuyNeural":         "🇺🇸 Guy (US, Male)",
    "en-US-JennyNeural":       "🇺🇸 Jenny (US, Female)",
    "en-US-AriaNeural":        "🇺🇸 Aria (US, Female)",
    "en-GB-RyanNeural":        "🇬🇧 Ryan (UK, Male)",
    "en-GB-SoniaNeural":       "🇬🇧 Sonia (UK, Female)",
    "en-AU-NatashaNeural":     "🇦🇺 Natasha (AU, Female)",
    "en-AU-WilliamNeural":     "🇦🇺 William (AU, Male)",
    "en-IE-ConnorNeural":      "🇮🇪 Connor (Irish, Male)",
    "en-IN-NeerjaNeural":      "🇮🇳 Neerja (Indian, Female)",
}

# ── Pydantic models ───────────────────────────────────────────────────────────
class Turn(BaseModel):
    role:       str
    text:       str
    image_b64:  Optional[str] = None
    image_mime: Optional[str] = None

class ChatRequest(BaseModel):
    module:     str
    messages:   list[Turn]
    clean_mode: bool = False

class ChatResponse(BaseModel):
    text:  str
    model: str
    error: Optional[str] = None

class TtsRequest(BaseModel):
    text:  str
    voice: str = "en-GB-RyanNeural"
    rate:  str = "+0%"    # e.g. "-10%", "+20%"
    pitch: str = "+0Hz"   # e.g. "-5Hz", "+10Hz"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="English Teacher API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    try:
        import edge_tts
        tts_ok = True
    except ImportError:
        tts_ok = False
    return {
        "status":      "ok",
        "model":       MODEL_NAME,
        "api_key_set": bool(API_KEY),
        "tts_ready":   tts_ok,
    }

@app.get("/api/tts/voices")
async def tts_voices():
    return {"voices": TTS_VOICES}

# ── Chat ──────────────────────────────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")

    system_prompt = SYSTEMS.get(req.module, {}).get(req.clean_mode, "You are a helpful English teacher.")

    # Exercises need temperature=0 for maximum determinism — only one correct answer exists
    temperature = 0.0 if req.module == "exercises" else 1.0

    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=8000,
                temperature=temperature,
            ),
        )

        history = [
            {"role": t.role, "parts": [{"text": t.text}]}
            for t in req.messages[:-1]
        ]

        session = model.start_chat(history=history)
        last    = req.messages[-1]

        if last.image_b64 and last.image_mime:
            content = [
                {"mime_type": last.image_mime, "data": last.image_b64},
                last.text or "Analyze this image and complete the exercise shown.",
            ]
            log.info("Multimodal request (image + text)")
        else:
            content = last.text

        response = session.send_message(content)
        log.info(f"module={req.module} clean={req.clean_mode} turns={len(req.messages)} chars={len(response.text)}")
        return ChatResponse(text=response.text, model=MODEL_NAME)

    except Exception as exc:
        log.error(f"Gemini error: {exc}")
        return ChatResponse(text=f"Помилка: {exc}", model=MODEL_NAME, error=str(exc))

# ── TTS ───────────────────────────────────────────────────────────────────────
@app.post("/api/tts")
async def tts(req: TtsRequest):
    """
    Generate speech using Microsoft Edge TTS (edge-tts).
    Returns an MP3 audio stream. Free, unlimited, no API key needed.
    """
    try:
        import edge_tts
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="edge-tts not installed. Run: pip install edge-tts"
        )

    if req.voice not in TTS_VOICES:
        raise HTTPException(status_code=400, detail=f"Unknown voice: {req.voice}")

    try:
        communicate = edge_tts.Communicate(
            text=req.text[:5000],   # cap at 5000 chars
            voice=req.voice,
            rate=req.rate,
            pitch=req.pitch,
        )
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/mpeg")

    except Exception as exc:
        log.error(f"TTS error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

# ── Static ────────────────────────────────────────────────────────────────────
# Mount at /assets (not /) so API routes are never intercepted.
# Root and all non-API paths serve index.html explicitly.
_static = os.path.join(os.path.dirname(__file__), "static")

if os.path.isdir(_static):
    app.mount("/assets", StaticFiles(directory=_static), name="static_assets")

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(os.path.join(_static, "index.html"))

    @app.get("/manifest.json", include_in_schema=False)
    async def manifest():
        return FileResponse(os.path.join(_static, "manifest.json"))

    # Catch-all for SPA — must be LAST
    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        # Never serve HTML for /api/* paths
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        candidate = os.path.join(_static, full_path)
        if os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(_static, "index.html"))
else:
    @app.get("/")
    async def root_fallback():
        return {"message": "Put your index.html in the /static folder."}
