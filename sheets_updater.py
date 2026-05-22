"""
Module for writing and formatting data in Google Sheets.

Responsible for assembling a 21-column row according to the specification,
applying red formatting to problematic comments, and calculating the final score sum.
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
#  Table Headers (21 columns)
# ──────────────────────────────────────────────────────────────────────────────
HEADERS = [
    "Дата",                                        # 1: Date
    "Тип звернення",                               # 2: Call Type
    "Номер телефону",                              # 3: Phone Number
    "Філія",                                       # 4: Branch (empty per spec)
    "Менеджер",                                    # 5: Manager Name
    "Початок розмови, представлення",              # 6: Greeting
    "Чи дізнався менеджер кузов автомобіля",       # 7: Car Body
    "Чи дізнався менеджер рік автомобіля",         # 8: Car Year
    "Чи дізнався менеджер пробіг",                 # 9: Car Mileage
    "Пропозиція про комплексну діагностику",       # 10: Offer Diagnostics
    "Дізнався які роботи робилися раніше",         # 11: Ask History
    "Запис на сервіс Дата",                        # 12: Appointment Date
    "Завершення розмови прощання",                 # 13: Farewell
    "Яка робота з топ 100",                        # 14: Top 100 Job
    "Чи дотримувався всіх інструкцій з топ 100",  # 15: Followed instructions (empty per spec)
    "Яких рекомендацій не дотримувався",           # 16: Missed recommendations (empty per spec)
    "Результат",                                   # 17: Result Status
    "Оцінка",                                      # 18: Score
    "Запчастини",                                  # 19: Parts Source
    "Коментар",                                    # 20: Comment
    "Фінальний рахунок",                           # 21: Final Score
]

# 1-based indices for formatting and formulas
COMMENT_COL_INDEX = HEADERS.index("Коментар") + 1        # 20
FINAL_SCORE_COL_INDEX = HEADERS.index("Фінальний рахунок") + 1  # 21

# Red background for problematic comments (score < 7)
RED_COLOR = Color(red=1.0, green=0.0, blue=0.0)

# Score threshold: >= 7 → 1 (OK), < 7 → 0 (problem)
SCORE_THRESHOLD = 7


# ──────────────────────────────────────────────────────────────────────────────
#  Filename Parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_filename(filename: str) -> tuple[str, str]:
    """
    Parses a filename with the format: 2025-07-14_14-48_0974747746_incoming.mp3

    Returns: (date in 'DD.MM.YYYY' format, phone_number).
    If the pattern doesn't match, returns the current date and an empty string.
    """
    # Date (first three digit groups separated by hyphen)
    date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", filename)
    if date_match:
        year, month, day = date_match.groups()
        call_date = f"{day}.{month}.{year}"
    else:
        call_date = datetime.now().strftime("%d.%m.%Y")
        logger.warning("Failed to extract date from filename: %s", filename)

    # Phone number (10 or more consecutive digits between underscores or before a dot)
    phone_match = re.search(r"_(\d{10,})[_.]", filename)
    phone = phone_match.group(1) if phone_match else ""
    
    # To prevent Google Sheets from removing a leading 0 (treating it as a number),
    # prepend an apostrophe to the string. It remains hidden in the UI but preserves the 0.
    if phone:
        phone = f"'{phone}"

    return call_date, phone


# ──────────────────────────────────────────────────────────────────────────────
#  Main Class
# ──────────────────────────────────────────────────────────────────────────────

class SheetsUpdater:
    """
    Class for updating the Google Sheets spreadsheet with call analysis results.

    Formats a 21-element row according to the spec and writes it to the sheet.
    Problematic calls (score < 7) are highlighted in red in the 'Comment' cell
    for easy visibility.
    """

    def __init__(self, spreadsheet: gspread.Spreadsheet) -> None:
        self._spreadsheet = spreadsheet
        self._worksheet = self._get_or_create_worksheet()
        self._ensure_headers()

    # ------------------------------------------------------------------ #
    #  Sheet Preparation
    # ------------------------------------------------------------------ #

    def _get_or_create_worksheet(self) -> gspread.Worksheet:
        """Retrieves the first worksheet or creates a new one with the required dimensions."""
        try:
            return self._spreadsheet.sheet1
        except Exception:
            return self._spreadsheet.add_worksheet(title="Звіт", rows=1000, cols=25)

    def _ensure_headers(self) -> None:
        """Writes headers to the first row if the sheet is empty."""
        first_row = self._worksheet.row_values(1)
        if not first_row:
            self._worksheet.append_row(HEADERS, value_input_option="USER_ENTERED")
            logger.info("Table headers (21 columns) written successfully.")

    # ------------------------------------------------------------------ #
    #  Adding Records
    # ------------------------------------------------------------------ #

    def add_call_record(
        self,
        filename: str,
        transcription_text: str,
        analysis: AnalysisResult,
    ) -> None:
        """
        Formats a 21-element row and appends it to the table.

        Final score logic: 1 if score >= 7, otherwise 0.
        If score < 7, the 'Comment' cell is colored red.

        The `transcription_text` argument is not included in the table; it is
        already saved in a .txt file on Drive next to the audio.
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
        logger.info("Row for '%s' appended to the table.", filename)

        # If the score is below the threshold, highlight the 'Comment' cell in red
        if analysis.score < SCORE_THRESHOLD:
            last_row = len(self._worksheet.get_all_values())
            comment_cell = rowcol_to_a1(last_row, COMMENT_COL_INDEX)
            try:
                fmt = CellFormat(backgroundColor=RED_COLOR)
                format_cell_range(self._worksheet, comment_cell, fmt)
                logger.info(
                    "Cell '%s' highlighted in red (score=%d < %d).",
                    comment_cell,
                    analysis.score,
                    SCORE_THRESHOLD,
                )
            except Exception as exc:
                logger.warning("Failed to apply red formatting: %s", exc)

    # ------------------------------------------------------------------ #
    #  Summary Formula
    # ------------------------------------------------------------------ #

    def add_total_sum_formula(self) -> None:
        """
        Appends a =SUM(...) formula at the end of the 'Final Score' column
        to calculate the total number of quality calls.
        """
        all_values = self._worksheet.get_all_values()
        last_data_row = len(all_values)

        if last_data_row < 2:
            logger.warning("No data available to calculate sum.")
            return

        # Get the column letter (e.g., 'U' for the 21st column)
        col_letter = rowcol_to_a1(1, FINAL_SCORE_COL_INDEX)[:-1]
        sum_formula = f"=SUM({col_letter}2:{col_letter}{last_data_row})"
        total_row = last_data_row + 1

        self._worksheet.update_cell(total_row, FINAL_SCORE_COL_INDEX, sum_formula)
        logger.info(
            "Sum formula '%s' written to row %d.",
            sum_formula,
            total_row,
        )

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def spreadsheet_url(self) -> str:
        """Returns the spreadsheet URL for logging purposes."""
        return self._spreadsheet.url
