"""
Модуль ШІ-аналізу розмов.

Використовує llama-cpp-python з локальною GGUF-моделлю (Qwen2.5 або аналог)
для аналізу транскрипцій дзвінків та повернення структурованого JSON-результату.
При першому запуску автоматично завантажує модель через huggingface_hub.
"""

import json
import logging
import os
from dataclasses import dataclass

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
#  Модель за замовчуванням: Qwen2.5-3B-Instruct GGUF (Q4_K_M — баланс якість/розмір)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
DEFAULT_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ──────────────────────────────────────────────────────────────────────────────
#  Списки допустимих значень
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
#  System prompt для LLM (відповідно до специфікації)
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
    "- `top_100_job`: Знайди найбільш схожу послугу зі списку Топ-100 ("
    "Комп'ютерна діагностика, Заміна оливи ДВЗ + масляний фільтр, Комплексна діагностика, Ендоскопія, "
    "Заміна повітряного фільтра ДВЗ, Заміна фільтра салону в салонному відділенні, Заміна сайлентблоку, "
    "Зняття / встановлення важеля, Заміна еластичної муфти карданного валу, Слюсарні роботи, "
    "Діагностика підвіски (ВИКОРИСТОВУЄМ КОМПЛЕКСНУ), Зняття / встановлення важеля прд., "
    "Заміна амортизатора переднього, Заміна оливи АКПП, Мийка / чистка деталі, "
    "Зняття / встановлення повітряного патрубка, Заміна охолоджувальної рідини, "
    "Заміна гальмівної рідини з прокачкою, Заміна оливи в зд. редукторі, Кодування опцій, "
    "Заміна амортизатора зд., Заміна гальмівних дисків та колодок прд.). "
    "Якщо жодна не згадувалась — поверни '-'.\n"
    "- `result_status`: Суворо одне з: 'Запис', 'Повторно консультація', "
    "'Передано іншому філіалу', 'Передзвонити', 'Інше'.\n"
    "- `score`: Твоя загальна оцінка якості роботи менеджера від 1 до 10.\n"
    "- `parts_source`: Суворо 'клієнта', 'наші' або '-' (якщо запчастини взагалі не обговорювались).\n"
    "- `comment`: Твій короткий коментар (до 2 речень) про те, що менеджер зробив добре, а що пропустив."
)


# ──────────────────────────────────────────────────────────────────────────────
#  Датакласи результату
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """Результат LLM-аналізу однієї розмови (15 полів JSON)."""

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
    """Повертає результат за замовчуванням для порожнього або нерозпізнаного тексту."""
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
#  Головний клас аналізатора
# ──────────────────────────────────────────────────────────────────────────────

class ConversationAnalyzer:
    """
    Аналізує текст розмови за допомогою llama-cpp-python (локальна GGUF-модель).

    При першому запуску автоматично завантажує модель (~1.9 ГБ для Q4_K_M)
    через huggingface_hub і зберігає її у папці `models/` поруч із скриптом.
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = DEFAULT_FILENAME,
        models_dir: str = MODELS_DIR,
        n_ctx: int = 32768,
        n_threads: int | None = None,
    ) -> None:
        model_path = self._ensure_model(repo_id, filename, models_dir)
        logger.info("Ініціалізація Llama з моделі: %s", model_path)
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads or max(1, (os.cpu_count() or 4) - 1),
            verbose=False,
        )
        logger.info("Модель llama-cpp успішно завантажено.")

    # ------------------------------------------------------------------ #
    #  Завантаження моделі
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ensure_model(repo_id: str, filename: str, models_dir: str) -> str:
        """Завантажує GGUF-модель, якщо вона ще відсутня локально."""
        os.makedirs(models_dir, exist_ok=True)
        local_path = os.path.join(models_dir, filename)
        if os.path.exists(local_path):
            logger.info("Модель вже є локально: %s", local_path)
            return local_path
        logger.info(
            "Модель не знайдена локально. Завантаження з Hugging Face: %s / %s ...",
            repo_id,
            filename,
        )
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=models_dir,
        )
        logger.info("Модель завантажено: %s", downloaded)
        return downloaded

    # ------------------------------------------------------------------ #
    #  Аналіз
    # ------------------------------------------------------------------ #

    def analyze(self, text: str) -> AnalysisResult:
        """
        Виконує LLM-аналіз тексту розмови та повертає AnalysisResult.

        Якщо текст порожній або LLM повертає невалідний JSON — повертає дефолтний результат.
        """
        if not text or not text.strip():
            logger.warning("Порожній текст — повертаємо результат за замовчуванням.")
            return _default_result("Текст розмови відсутній або не розпізнано.")

        logger.info("Виконується LLM-аналіз тексту (%d символів)...", len(text))
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
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=512,
            )
            raw_json: str = response["choices"][0]["message"]["content"]
            logger.debug("LLM відповідь: %s", raw_json)
            data: dict = json.loads(raw_json)
        except Exception as exc:
            logger.error("Помилка LLM або парсингу JSON: %s", exc)
            return _default_result(f"Помилка аналізу: {exc}")

        return self._parse_result(data)

    # ------------------------------------------------------------------ #
    #  Парсинг та валідація відповіді
    # ------------------------------------------------------------------ #

    def _parse_result(self, data: dict) -> AnalysisResult:
        """Парсить словник із JSON-відповіді LLM у AnalysisResult з валідацією полів."""

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
            comment=safe_str(data.get("comment"), "Коментар відсутній."),
        )
