# Auto Service QA Call Audit

A project for automating the audit of phone conversations between auto service managers and clients. The script integrates with Google Drive, downloads call recordings, transcribes them using a GPU-accelerated pipeline (pyannote.audio, faster-whisper), analyzes them via a local neural network (Qwen2.5-7B) on the GPU, and generates a detailed report table in Google Sheets.

The entire process is strictly optimized for GPUs with 12 GB VRAM (total memory consumption up to ~11 GB).

---

## 💻 Detailed Usage Instructions

### 1. Google Cloud Preparation (Obtaining Access)
Since the script creates files on your behalf in your Google Drive, you need to obtain the `client_secret.json` authorization file:
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or select an existing one).
3. On the left menu, open **APIs & Services** -> **Library**. Find and enable two APIs:
   - **Google Drive API**
   - **Google Sheets API**
4. Go to **APIs & Services** -> **OAuth consent screen**:
   - Choose **External**, click Create.
   - Fill in the required fields (app name, your email).
   - On the "Test users" step, **add your email** (e.g., `your.email@gmail.com`).
5. Go to **APIs & Services** -> **Credentials**:
   - Click **+ CREATE CREDENTIALS** -> **OAuth client ID**.
   - Select **Application type**: `Desktop app`.
   - Click **Create**, then download the JSON file (⬇️ button).
6. Rename the downloaded file to `client_secret.json` and place it in the root folder of this project.

### 2. Folder Configuration
In `google_services.py` (lines 29-32), specify the correct Google Drive folder IDs:
- `SHARED_FOLDER_ID`: The ID of the shared folder where the script will **read** the original audio files from.
- `REPORT_FOLDER_ID`: The ID of your workspace folder where the script will **save** audio copies, transcription files (`.txt`), and the final report (Google Sheets).

*(The folder ID is the long string of characters in the browser URL after `folders/`)*.

### 3. Environment Variables (.env)
The project relies on environment variables for sensitive tokens.
1. Copy the provided `.env.example` file to a new file named `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and fill in your Hugging Face token.
   - Register on [Hugging Face](https://huggingface.co/) and create an Access Token (Read permissions).
   - Accept the terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).

### 4. Dependency Installation and GPU Setup
The project uses CUDA for maximum acceleration of transcription and LLM analysis.
Run the setup script (it will detect your CUDA version, create a virtual environment `python -m venv .venv`, and install all necessary packages, pinning exact working versions like PyTorch with CUDA and llama-cpp-python):
```powershell
.\setup.ps1
```

### 5. First Run and Authorization
Activate the virtual environment and run the main script:
```powershell
.\.venv\Scripts\activate
python main.py
```
**What happens on the first run:**
1. The script will automatically download the `Qwen2.5-7B-Instruct-Q3_K_M.gguf` neural network (~3 GB) and the transcription models (Whisper large-v3, pyannote). This is done only once.
2. A browser window will open asking you to log into your Google account.
3. Since your app is in testing status, Google may warn: *"Google hasn't verified this app"*. Click **Advanced** -> **Go to [Name] (unsafe)** and grant the required Drive and Sheets permissions.
4. The script will save `token.json` in the project folder so the browser won't open on subsequent runs.

### 6. How the Script Works (Workflow)
After a successful launch, the script operates fully automatically in a pipeline mode:
1. Creates a new Google Spreadsheet in your `REPORT_FOLDER_ID` folder.
2. Locates all `.mp3` files in the shared folder.
3. Sequentially processes each call:
   - Copies the `mp3` to your workspace folder.
   - Transcribes the audio locally on the GPU (Diarization -> Whisper ASR) with smart VRAM management.
   - Uploads the structured dialogue text (`.txt`) to Google Drive next to the copied audio file.
   - Analyzes the conversation against service station quality criteria using the local AI (Qwen2.5-7B) on the GPU.
   - Appends a row with the scores to the Google Spreadsheet.
4. If the overall call score is < 7 points, the comment cell is automatically highlighted in red.
5. After processing all files, the script adds a `=SUM()` formula to calculate the final score and outputs a convenient link to the generated report in the console.
