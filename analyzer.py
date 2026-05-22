"""
Module for AI-based conversation analysis.

Uses llama-cpp-python with a local GGUF model (Qwen2.5-7B-Instruct Q4_K_M)
to analyze call transcripts and return a structured JSON result.
The model is fully loaded into the GPU (CUDA) via n_gpu_layers=-1.

VRAM Consumption:
    Qwen2.5-7B Q4_K_M  ~4.5 GB  (remains in VRAM for the entire duration)
    With Whisper        ~1.6 GB
    ──────────────────────────────
    Total               ~6.1 GB  (well within the 11 GB limit)

On the first run, the model is automatically downloaded via huggingface_hub.
"""

import json
import logging
import os
from dataclasses import dataclass

from huggingface_hub import hf_hub_download

# Fix for Windows: register paths to CUDA DLLs before importing llama_cpp
import sys
if sys.platform == "win32":
    import site
    import glob
    for site_pkg in site.getsitepackages():
        for nv_bin in glob.glob(os.path.join(site_pkg, "nvidia", "*", "bin")):
            os.add_dll_directory(nv_bin)
            os.environ["PATH"] = nv_bin + os.pathsep + os.environ["PATH"]

from llama_cpp import Llama

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
#  Default Model: Qwen2.5-7B-Instruct GGUF (Q3_K_M - single file, ~3.0 GB VRAM on GPU)
#  Q3_K_M fits perfectly within 11 GB alongside Whisper large-v3 (~1.6 GB).
#  Q4_K_M for 7B is split into 2 shards (not suitable for hf_hub_download directly).
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_REPO_ID = "Qwen/Qwen2.5-7B-Instruct-GGUF"
DEFAULT_FILENAME = "qwen2.5-7b-instruct-q3_k_m.gguf"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ──────────────────────────────────────────────────────────────────────────────
#  Allowed Values Lists
# ──────────────────────────────────────────────────────────────────────────────
TOP_100_JOBS = [
    "Комп'ютерна діагностика",
    "Заміна оливи ДВЗ + масляний фільтр",
    "Комплексна діагностика",
    "Ендоскопія",
    "Заміна повітряного фільтра ДВЗ",
    "Заміна фільтра салону в салонному відділенні",
    "Заміна сайлентблоку",
    "Зняття / встановлення важеля",
    "Заміна еластичної муфти карданного валу",
    "Слюсарні роботи",
    "Діагностика підвіски (ВИКОРИСТОВУЄМ КОМПЛЕКСНУ)",
    "Зняття / встановлення важеля прд.",
    "Заміна амортизатора переднього",
    "Заміна оливи АКПП",
    "Мийка / чистка деталі",
    "Зняття / встановлення повітряного патрубка",
    "Заміна охолоджувальної рідини",
    "Заміна гальмівної рідини з прокачкою",
    "Заміна оливи в зд. редукторі",
    "Кодування опцій",
    "Заміна амортизатора зд.",
    "Заміна гальмівних дисків та колодок прд.",
]

VALID_RESULT_STATUSES = [
    "Запис",
    "Повторно консультація",
    "Передано іншому філіалу",
    "Передзвонити",
    "Інше",
]

# ──────────────────────────────────────────────────────────────────────────────
#  JSON Schema (Grammar) to strictly constrain LLM output
# ──────────────────────────────────────────────────────────────────────────────
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "call_type": {"type": "string", "enum": ["Вхідний", "Вихідний"]},
        "manager_name": {"type": "string"},
        "greeting": {"type": "integer", "enum": [0, 1]},
        "car_body": {"type": "integer", "enum": [0, 1]},
        "car_year": {"type": "integer", "enum": [0, 1]},
        "car_mileage": {"type": "integer", "enum": [0, 1]},
        "offer_diagnostics": {"type": "integer", "enum": [0, 1]},
        "ask_history": {"type": "integer", "enum": [0, 1]},
        "appointment_date": {"type": "string"},
        "farewell": {"type": "integer", "enum": [0, 1]},
        "top_100_job": {
            "type": "string",
            "enum": [
                "Комп'ютерна діагностика", "Заміна оливи ДВЗ + масляний фільтр", "Комплексна діагностика", "Ендоскопія",
                "Заміна повітряного фільтра ДВЗ", "Заміна фільтра салону в салонному відділенні", "Заміна сайлентблоку",
                "Зняття / встановлення важеля", "Заміна еластичної муфти карданного валу", "Слюсарні роботи",
                "Діагностика підвіски (ВИКОРИСТОВУЄМ КОМПЛЕКСНУ)", "Зняття / встановлення важеля прд.",
                "Заміна амортизатора переднього", "Заміна оливи АКПП", "Мийка / чистка деталі",
                "Зняття / встановлення повітряного патрубка", "Заміна охолоджувальної рідини",
                "Заміна гальмівної рідини з прокачкою", "Заміна оливи в зд. редукторі", "Кодування опцій",
                "Заміна амортизатора зд.", "Заміна гальмівних дисків та колодок прд.", "-"
            ]
        },
        "result_status": {
            "type": "string",
            "enum": ["Запис", "Повторно консультація", "Передано іншому філіалу", "Передзвонити", "Інше"]
        },
        "score": {"type": "integer"},
        "parts_source": {"type": "string", "enum": ["клієнта", "наші", "-"]},
        "comment": {"type": "string"}
    },
    "required": [
        "call_type", "manager_name", "greeting", "car_body", "car_year", "car_mileage",
        "offer_diagnostics", "ask_history", "appointment_date", "farewell", "top_100_job",
        "result_status", "score", "parts_source", "comment"
    ]
}

# ──────────────────────────────────────────────────────────────────────────────
#  System prompt for LLM (according to specifications)
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Ти — експерт-аудитор відділу контролю якості (QA) мережі автосервісів (СТО). "
    "Твоє завдання — проаналізувати транскрибацію телефонної розмови між менеджером СТО та клієнтом. "
    "Розмова може містити українську мову, російську мову, суржик та специфічний сленг автомеханіків. "
    "Ти повинен повернути результат СУВОРО у форматі JSON без жодного додаткового тексту чи markdown-розмітки.\n\n"
    "Правила аналізу для полів JSON:\n"
    "- `call_type`: Визнач, чи це 'Вхідний' (клієнт дзвонить сам) чи 'Вихідний' (менеджер дзвонить).\n"
    "- `manager_name`: Ім'я менеджера. Якщо не назвав, поверни '-'.\n"
    "- `greeting`: 1 якщо менеджер привітався, інакше 0.\n"
    "- `car_body`: 1 якщо менеджер дізнався тип кузова (навіть непрямо, наприклад, якщо клієнт сказав "
    "'Ланос' — це седан, значить кузов відомий). Якщо не обговорювали — 0.\n"
    "- `car_year`: 1 якщо рік авто був названий, 0 якщо ні.\n"
    "- `car_mileage`: 1 якщо пробіг обговорювався, 0 якщо ні.\n"
    "- `offer_diagnostics`: 1 якщо менеджер запропонував діагностику/перевірку, 0 якщо ні.\n"
    "- `ask_history`: 1 якщо менеджер запитав, що робили з авто раніше або де обслуговувались, 0 якщо ні.\n"
    "- `appointment_date`: Дата або час запису (наприклад 'завтра на 10:00'). Якщо не записались — порожній рядок ''.\n"
    "- `farewell`: 1 якщо менеджер попрощався, 0 якщо ні.\n"
    "- `top_100_job`: Вибери рівно ОДНУ послугу з цього списку Топ-100: "
    "['Комп'ютерна діагностика', 'Заміна оливи ДВЗ + масляний фільтр', 'Комплексна діагностика', 'Ендоскопія', "
    "'Заміна повітряного фільтра ДВЗ', 'Заміна фільтра салону в салонному відділенні', 'Заміна сайлентблоку', "
    "'Зняття / встановлення важеля', 'Заміна еластичної муфти карданного валу', 'Слюсарні роботи', "
    "'Діагностика підвіски (ВИКОРИСТОВУЄМ КОМПЛЕКСНУ)', 'Зняття / встановлення важеля прд.', "
    "'Заміна амортизатора переднього', 'Заміна оливи АКПП', 'Мийка / чистка деталі', "
    "'Зняття / встановлення повітряного патрубка', 'Заміна охолоджувальної рідини', "
    "'Заміна гальмівної рідини з прокачкою', 'Заміна оливи в зд. редукторі', 'Кодування опцій', "
    "'Заміна амортизатора зд.', 'Заміна гальмівних дисків та колодок прд.']. "
    "Поверни лише точну назву послуги як рядок (без масивів і списків). Якщо жодна не згадувалась — поверни '-'.\n"
    "- `result_status`: Суворо одне з: 'Запис', 'Повторно консультація', "
    "'Передано іншому філіалу', 'Передзвонити', 'Інше'.\n"
    "- `score`: Твоя загальна оцінка якості роботи менеджера від 1 до 10. Оцінюй емпатію, вирішення конфліктів та задоволеність клієнта. Будь лояльним до дрібниць, але суворо знімай бали за пасивну агресію, небажання слухати клієнта чи погану роботу з запереченнями.\n"
    "- `parts_source`: Суворо 'клієнта', 'наші' або '-' (якщо запчастини взагалі не обговорювались).\n"
    "- `comment`: Твій ОБОВ'ЯЗКОВИЙ короткий коментар (1-3 речення) про те, як менеджер впорався. Якщо була конфліктна ситуація, опиши, наскільки добре він її вирішив. Це поле МАЄ БУТИ ЗАВЖДИ!"
)


# ──────────────────────────────────────────────────────────────────────────────
#  Result Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """Result of the LLM analysis of a single call (15 JSON fields)."""

    call_type: str           # "Вхідний" / "Вихідний"
    manager_name: str        # ім'я або "-"
    greeting: int            # 0 або 1
    car_body: int            # 0 або 1
    car_year: int            # 0 або 1
    car_mileage: int         # 0 або 1
    offer_diagnostics: int   # 0 або 1
    ask_history: int         # 0 або 1
    appointment_date: str    # дата/час або ""
    farewell: int            # 0 або 1
    top_100_job: str         # назва послуги або "-"
    result_status: str       # один із VALID_RESULT_STATUSES
    score: int               # 1–10
    parts_source: str        # "клієнта" / "наші" / "-"
    comment: str             # короткий коментар


def _default_result(reason: str = "Аналіз не виконано.") -> AnalysisResult:
    """Returns a default result for empty or unrecognized text."""
    return AnalysisResult(
        call_type="Вхідний",
        manager_name="-",
        greeting=0,
        car_body=0,
        car_year=0,
        car_mileage=0,
        offer_diagnostics=0,
        ask_history=0,
        appointment_date="",
        farewell=0,
        top_100_job="-",
        result_status="Інше",
        score=1,
        parts_source="-",
        comment=reason,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Main Analyzer Class
# ──────────────────────────────────────────────────────────────────────────────

class ConversationAnalyzer:
    """
    Analyzes conversation text using llama-cpp-python (local GGUF model).

    On the first run, it automatically downloads the model (~1.9 GB for Q4_K_M)
    via huggingface_hub and saves it in the `models/` directory next to the script.
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = DEFAULT_FILENAME,
        models_dir: str = MODELS_DIR,
        n_ctx: int = 8192,       # limited to protect against OOM for long conversations
        n_gpu_layers: int = -1,  # -1 = all layers on GPU (CUDA)
        n_threads: int | None = None,
    ) -> None:
        model_path = self._ensure_model(repo_id, filename, models_dir)
        logger.info("Initializing Llama from model: %s (GPU layers: %s)", model_path, n_gpu_layers)
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,   # load all layers to CUDA
            n_threads=n_threads or max(1, (os.cpu_count() or 4) - 1),
            verbose=False,
        )
        logger.info("llama-cpp model successfully loaded to GPU.")

    # ------------------------------------------------------------------ #
    #  Model Loading
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ensure_model(repo_id: str, filename: str, models_dir: str) -> str:
        """Downloads the GGUF model if it is not already available locally."""
        os.makedirs(models_dir, exist_ok=True)
        local_path = os.path.join(models_dir, filename)
        if os.path.exists(local_path):
            logger.info("Model already exists locally: %s", local_path)
            return local_path
        logger.info(
            "Model not found locally. Downloading from Hugging Face: %s / %s ...",
            repo_id,
            filename,
        )
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=models_dir,
        )
        logger.info("Model downloaded: %s", downloaded)
        return downloaded
    # ------------------------------------------------------------------ #
    #  Analysis
    # ------------------------------------------------------------------ #

    # Maximum length of text passed to the LLM (in characters).
    # Protects against exceeding n_ctx for long conversations.
    # At n_ctx=8192 and ~3 characters/token -> ~20,000 characters is the limit.
    # 12000 characters is a safe maximum for the text (leaving room for prompt and response).
    _MAX_TEXT_CHARS = 12000

    def analyze(self, text: str) -> AnalysisResult:
        """
        Performs LLM analysis of the conversation text and returns an AnalysisResult.

        If the text is empty or the LLM returns invalid JSON, returns a default result.
        The text is automatically truncated to _MAX_TEXT_CHARS to protect against OOM.
        """
        if not text or not text.strip():
            logger.warning("Empty text - returning default result.")
            return _default_result("Conversation text missing or not recognized.")

        # Truncate text to avoid exceeding the model's context limit
        if len(text) > self._MAX_TEXT_CHARS:
            logger.warning(
                "Transcription text is %d characters - truncating to %d to protect against OOM.",
                len(text), self._MAX_TEXT_CHARS,
            )
            text = text[:self._MAX_TEXT_CHARS]

        logger.info("Performing LLM analysis of text (%d characters) on GPU...", len(text))
        try:
            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Проаналізуй наступну транскрипцію телефонного дзвінка:\n\n"
                            + text
                        ),
                    },
                ],
                response_format={
                    "type": "json_object",
                    "schema": RESPONSE_SCHEMA
                },
                temperature=0.1,
                max_tokens=1024,
            )
            raw_json: str = response["choices"][0]["message"]["content"]
            logger.debug("LLM response: %s", raw_json)
            data: dict = json.loads(raw_json)
        except Exception as exc:
            logger.error("LLM or JSON parsing error: %s", exc)
            return _default_result(f"Analysis error: {exc}")

        return self._parse_result(data)

    def analyze_bilingual(self, text_uk: str, text_ru: str) -> AnalysisResult:
        """
        Performs combined LLM analysis of two texts (Ukrainian + Russian).
        
        The AI receives both texts and autonomously decides how to use them
        for the most accurate analysis.

        Args:
            text_uk: Transcription text in Ukrainian
            text_ru: Transcription text in Russian

        Returns:
            AnalysisResult with the combined analysis
        """
        if (not text_uk or not text_uk.strip()) and (not text_ru or not text_ru.strip()):
            logger.warning("Both texts are empty - returning default result.")
            return _default_result("Conversation text missing or not recognized.")

        # For bilingual processing: truncate even more to have a safety margin
        # Each text takes ~45% of the total limit, the rest is for the prompt
        max_chars_per_lang = int(self._MAX_TEXT_CHARS * 0.45)
        
        if len(text_uk) > max_chars_per_lang:
            logger.warning("Ukrainian text %d characters - truncating to %d.", len(text_uk), max_chars_per_lang)
            text_uk = text_uk[:max_chars_per_lang]
        
        if len(text_ru) > max_chars_per_lang:
            logger.warning("Russian text %d characters - truncating to %d.", len(text_ru), max_chars_per_lang)
            text_ru = text_ru[:max_chars_per_lang]

        # Combined prompt for AI
        combined_prompt = (
            "Проаналізуй наступну телефонну розмову:\n\n"
            "УКРАЇНСЬКА ВЕРСІЯ:\n"
            + text_uk + "\n\n"
            "РОСІЙСЬКА ВЕРСІЯ:\n"
            + text_ru + "\n\n"
            "Використай обидва варіанти для найточнішого розуміння. "
            "Комбінуй інформацію з обох версій для повного аналізу."
        )

        logger.info(
            "Performing combined LLM analysis (UK: %d chars, RU: %d chars) on GPU...",
            len(text_uk), len(text_ru),
        )
        try:
            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": combined_prompt,
                    },
                ],
                response_format={
                    "type": "json_object",
                    "schema": RESPONSE_SCHEMA
                },
                temperature=0.1,
                max_tokens=1024,
            )
            raw_json: str = response["choices"][0]["message"]["content"]
            logger.debug("LLM response: %s", raw_json)
            data: dict = json.loads(raw_json)
        except Exception as exc:
            logger.error("LLM or JSON parsing error: %s", exc)
            return _default_result(f"Analysis error: {exc}")

        return self._parse_result(data)


    # ------------------------------------------------------------------ #
    #  Parsing and Response Validation
    # ------------------------------------------------------------------ #

    def _parse_result(self, data: dict) -> AnalysisResult:
        """Parses the JSON dictionary from the LLM response into an AnalysisResult with field validation."""

        def safe_int(val, default: int = 0) -> int:
            try:
                return int(val)
            except (TypeError, ValueError):
                return default

        def safe_str(val, default: str = "-") -> str:
            return str(val).strip() if val is not None else default

        score = max(1, min(10, safe_int(data.get("score"), 5)))

        result_status = safe_str(data.get("result_status"), "Інше")
        if result_status not in VALID_RESULT_STATUSES:
            result_status = "Інше"

        top_100_job = safe_str(data.get("top_100_job"), "-")
        if top_100_job not in TOP_100_JOBS:
            top_100_job = "-"

        parts_source = safe_str(data.get("parts_source"), "-")
        if parts_source not in ("клієнта", "наші", "-"):
            parts_source = "-"

        return AnalysisResult(
            call_type=safe_str(data.get("call_type"), "Вхідний"),
            manager_name=safe_str(data.get("manager_name"), "-"),
            greeting=safe_int(data.get("greeting")),
            car_body=safe_int(data.get("car_body")),
            car_year=safe_int(data.get("car_year")),
            car_mileage=safe_int(data.get("car_mileage")),
            offer_diagnostics=safe_int(data.get("offer_diagnostics")),
            ask_history=safe_int(data.get("ask_history")),
            appointment_date=safe_str(data.get("appointment_date"), ""),
            farewell=safe_int(data.get("farewell")),
            top_100_job=top_100_job,
            result_status=result_status,
            score=score,
            parts_source=parts_source,
            comment=safe_str(data.get("comment"), "No comment provided."),
        )
