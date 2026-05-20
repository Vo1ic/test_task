"""
Модуль для запису даних у Google Sheets та їх форматування.

Відповідає за формування рядка з 21 колонки відповідно до специфікації,
червоне форматування проблемних коментарів та підрахунок балів у підсумку.
"""

import logging
import re
from datetime import datetime

import gspread
from gspread.utils import rowcol_to_a1
from gspread_formatting import CellFormat, Color, format_cell_range

from analyzer import AnalysisResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
#  Заголовки таблиці (21 колонка)
# ──────────────────────────────────────────────────────────────────────────────
HEADERS = [
    "Дата",                                        # 1
    "Тип звернення",                               # 2
    "Номер телефону",                              # 3
    "Філія",                                       # 4  (порожньо за ТЗ)
    "Менеджер",                                    # 5
    "Початок розмови, представлення",              # 6
    "Чи дізнався менеджер кузов автомобіля",       # 7
    "Чи дізнався менеджер рік автомобіля",         # 8
    "Чи дізнався менеджер пробіг",                 # 9
    "Пропозиція про комплексну діагностику",       # 10
    "Дізнався які роботи робилися раніше",         # 11
    "Запис на сервіс Дата",                        # 12
    "Завершення розмови прощання",                 # 13
    "Яка робота з топ 100",                        # 14
    "Чи дотримувався всіх інструкцій з топ 100",  # 15 (порожньо за ТЗ)
    "Яких рекомендацій не дотримувався",           # 16 (порожньо за ТЗ)
    "Результат",                                   # 17
    "Оцінка",                                      # 18
    "Запчастини",                                  # 19
    "Коментар",                                    # 20
    "Фінальний рахунок",                           # 21
]

# Індекси (1-based) для форматування та формул
COMMENT_COL_INDEX = HEADERS.index("Коментар") + 1        # 20
FINAL_SCORE_COL_INDEX = HEADERS.index("Фінальний рахунок") + 1  # 21

# Червоний фон для проблемних коментарів (оцінка < 7)
RED_COLOR = Color(red=1.0, green=0.0, blue=0.0)

# Поріг оцінки: >= 7 → 1 (ОК), < 7 → 0 (проблема)
SCORE_THRESHOLD = 7


# ──────────────────────────────────────────────────────────────────────────────
#  Парсинг назви файлу
# ──────────────────────────────────────────────────────────────────────────────

def _parse_filename(filename: str) -> tuple[str, str]:
    """
    Парсить назву файлу формату: 2025-07-14_14-48_0974747746_incoming.mp3

    Повертає: (дата у форматі 'ДД.ММ.РРРР', номер_телефону).
    Якщо шаблон не збігається — дата = поточна, номер = порожній рядок.
    """
    # Дата (перші три групи цифр через дефіс)
    date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", filename)
    if date_match:
        year, month, day = date_match.groups()
        call_date = f"{day}.{month}.{year}"
    else:
        call_date = datetime.now().strftime("%d.%m.%Y")
        logger.warning("Не вдалося витягти дату з назви файлу: %s", filename)

    # Номер телефону (10 і більше цифр підряд між підкресленнями або перед крапкою)
    phone_match = re.search(r"_(\d{10,})[_.]", filename)
    phone = phone_match.group(1) if phone_match else ""

    return call_date, phone


# ──────────────────────────────────────────────────────────────────────────────
#  Головний клас
# ──────────────────────────────────────────────────────────────────────────────

class SheetsUpdater:
    """
    Клас для оновлення Google Sheets таблиці результатами аналізу дзвінків.

    Формує рядок із 21 елемента відповідно до специфікації та записує його
    в таблицю. Проблемні дзвінки (оцінка < 7) виділяються червоним кольором
    у комірці 'Коментар' для зручного реагування.
    """

    def __init__(self, spreadsheet: gspread.Spreadsheet) -> None:
        self._spreadsheet = spreadsheet
        self._worksheet = self._get_or_create_worksheet()
        self._ensure_headers()

    # ------------------------------------------------------------------ #
    #  Підготовка аркуша
    # ------------------------------------------------------------------ #

    def _get_or_create_worksheet(self) -> gspread.Worksheet:
        """Отримує перший аркуш або створює новий із потрібними розмірами."""
        try:
            return self._spreadsheet.sheet1
        except Exception:
            return self._spreadsheet.add_worksheet(title="Звіт", rows=1000, cols=25)

    def _ensure_headers(self) -> None:
        """Записує заголовки у перший рядок, якщо таблиця порожня."""
        first_row = self._worksheet.row_values(1)
        if not first_row:
            self._worksheet.append_row(HEADERS, value_input_option="USER_ENTERED")
            logger.info("Заголовки таблиці (21 колонка) записано.")

    # ------------------------------------------------------------------ #
    #  Додавання запису
    # ------------------------------------------------------------------ #

    def add_call_record(
        self,
        filename: str,
        transcription_text: str,
        analysis: AnalysisResult,
    ) -> None:
        """
        Формує рядок із 21 елемента та додає його до таблиці.

        Логіка фінального рахунку: 1 якщо оцінка >= 7, інакше 0.
        Якщо оцінка < 7 — комірка 'Коментар' фарбується у червоний.

        Аргументи transcription_text не включається у таблицю — він вже
        збережений у .txt файлі на Drive поруч із аудіо.
        """
        call_date, phone_number = _parse_filename(filename)
        final_score = 1 if analysis.score >= SCORE_THRESHOLD else 0

        row_data = [
            call_date,                   # 1  Дата
            analysis.call_type,          # 2  Тип звернення
            phone_number,                # 3  Номер телефону
            "",                          # 4  Філія (порожньо за ТЗ)
            analysis.manager_name,       # 5  Менеджер
            analysis.greeting,           # 6  Представлення (0/1)
            analysis.car_body,           # 7  Кузов (0/1)
            analysis.car_year,           # 8  Рік (0/1)
            analysis.car_mileage,        # 9  Пробіг (0/1)
            analysis.offer_diagnostics,  # 10 Діагностика (0/1)
            analysis.ask_history,        # 11 Роботи раніше (0/1)
            analysis.appointment_date,   # 12 Запис — дата/час або ""
            analysis.farewell,           # 13 Прощання (0/1)
            analysis.top_100_job,        # 14 Топ-100 робота або "-"
            "",                          # 15 Дотримувався інструкцій (порожньо за ТЗ)
            "",                          # 16 Рекомендації (порожньо за ТЗ)
            analysis.result_status,      # 17 Результат
            analysis.score,              # 18 Оцінка (1–10)
            analysis.parts_source,       # 19 Запчастини
            analysis.comment,            # 20 Коментар
            final_score,                 # 21 Фінальний рахунок (0 або 1)
        ]

        self._worksheet.append_row(row_data, value_input_option="USER_ENTERED")
        logger.info("Рядок для '%s' додано до таблиці.", filename)

        # Якщо оцінка нижча за поріг — фарбуємо комірку 'Коментар' у червоний
        if analysis.score < SCORE_THRESHOLD:
            last_row = len(self._worksheet.get_all_values())
            comment_cell = rowcol_to_a1(last_row, COMMENT_COL_INDEX)
            try:
                fmt = CellFormat(backgroundColor=RED_COLOR)
                format_cell_range(self._worksheet, comment_cell, fmt)
                logger.info(
                    "Комірку '%s' зафарбовано у червоний (оцінка=%d < %d).",
                    comment_cell,
                    analysis.score,
                    SCORE_THRESHOLD,
                )
            except Exception as exc:
                logger.warning("Не вдалося застосувати червоне форматування: %s", exc)

    # ------------------------------------------------------------------ #
    #  Підсумкова формула
    # ------------------------------------------------------------------ #

    def add_total_sum_formula(self) -> None:
        """
        Додає формулу =SUM(...) у кінці стовпця 'Фінальний рахунок'
        для підрахунку загальної кількості якісних дзвінків.
        """
        all_values = self._worksheet.get_all_values()
        last_data_row = len(all_values)

        if last_data_row < 2:
            logger.warning("Немає даних для підрахунку суми.")
            return

        # Отримуємо літеру стовпця (наприклад 'U' для 21-ї колонки)
        col_letter = rowcol_to_a1(1, FINAL_SCORE_COL_INDEX)[:-1]
        sum_formula = f"=SUM({col_letter}2:{col_letter}{last_data_row})"
        total_row = last_data_row + 1

        self._worksheet.update_cell(total_row, FINAL_SCORE_COL_INDEX, sum_formula)
        logger.info(
            "Формулу суми '%s' записано в рядок %d.",
            sum_formula,
            total_row,
        )

    # ------------------------------------------------------------------ #
    #  Властивості
    # ------------------------------------------------------------------ #

    @property
    def spreadsheet_url(self) -> str:
        """Повертає URL таблиці для відображення в логах."""
        return self._spreadsheet.url
