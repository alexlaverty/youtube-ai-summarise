# YouTube Video Summarizer using Ollama or Gemini

This Python script extracts subtitles from a YouTube video or an entire playlist, sends the transcript to either a locally running Ollama instance or the Google Gemini API, generates a detailed list of key points, and saves the results to a text file.

## Features

* Processes single YouTube videos or entire playlists.
* Automatically fetches available subtitles (prioritizes languages set in config or command line).
* Cleans subtitle text by removing timestamps and annotations.
* Supports two summarization engines:
    * **Ollama:** Uses a local Ollama instance.
    * **Gemini:** Uses the Google Gemini API (requires API key).
* Configuration managed via `config.yml` (API keys, model preferences, URLs, etc.).
* Command-line arguments to override configuration and select engine.
* Appends results (Title, URL, Detailed Points) to a specified output file.
* Includes error handling and progress reporting.
* Configurable delay between processing videos in a playlist.

## Prerequisites

* **Python 3.x:** Ensure you have Python installed.
* **Ollama (Optional):** If using the `ollama` engine, you need a running Ollama instance accessible from where you run the script. Download and install it from [https://ollama.com/](https://ollama.com/). Make sure the desired model is pulled (e.g., `ollama pull llama3:latest`).
* **Google Gemini API Key (Optional):** If using the `gemini` engine, you need an API key from Google AI Studio ([https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)).
* **Required Python Libraries:** Install the necessary libraries using pip.

## Installation

1.  **Clone the repository (or download the script):**
    ```bash
    # If you have a git repo setup
    # git clone <your-repo-url>
    # cd <your-repo-directory>

    # Or just ensure you have the script (e.g., app.py) and create config.yml
    ```

2.  **Create `requirements.txt`:**
    Create a file named `requirements.txt` with the following content:
    ```txt
    requests
    pytubefix
    google-generativeai
    PyYAML
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create `config.yml`:**
    Create a file named `config.yml` in the same directory as the script. You can copy and paste the example below and modify it with your settings.
    ```yaml
    # config.yml Example

    # Default summarization engine ('ollama' or 'gemini')
    engine: ollama

    # Ollama settings (used if engine is 'ollama')
    ollama_url: "http://localhost:11434/api/chat"
    ollama_model: "llama3:latest" # Make sure you have pulled this model

    # Gemini settings (used if engine is 'gemini')
    # Get your key from [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
    gemini_api_key: "YOUR_GEMINI_API_KEY_HERE" # IMPORTANT: Keep this secure!

    # General settings
    language_preference: ['en', 'a.en'] # Preferred subtitle languages (ISO 639-1 codes)
    output_file: "output.txt"
    playlist_video_delay_seconds: 2 # Delay between videos in a playlist

    ```
    **Important:** Replace `"YOUR_GEMINI_API_KEY_HERE"` with your actual API key if you plan to use the Gemini engine. Consider using environment variables or other secure methods for managing API keys in production environments.

## Configuration

The script reads its configuration from `config.yml`. The key settings are:

* `engine`: Specifies the default summarization engine (`ollama` or `gemini`). Can be overridden with the `--engine` argument.
* `ollama_url`: The URL for your local Ollama API endpoint.
* `ollama_model`: The specific Ollama model to use (ensure it's pulled).
* `gemini_api_key`: Your Google Gemini API Key. **Required** if using the `gemini` engine.
* `language_preference`: A list of preferred language codes for subtitles (e.g., `en` for English, `es` for Spanish, `a.en` for auto-generated English).
* `output_file`: The default file where results are saved.
* `playlist_video_delay_seconds`: Time in seconds to wait between processing videos in a playlist to avoid rate limiting.

Most configuration values can be overridden using command-line arguments.

## Usage

Ensure your Ollama instance is running if you intend to use the `ollama` engine.

**Command-line arguments:**

* `-u URL` or `--url URL`: Process a single YouTube video URL.
* `-p URL` or `--playlist URL`: Process all videos in a YouTube playlist URL.
* `--engine {ollama,gemini}`: Override the summarization engine specified in `config.yml`.
* `-o FILE` or `--output-file FILE`: Specify the output file path (overrides `config.yml`).
* `-m MODEL` or `--model MODEL`: Specify the Ollama model name (overrides `config.yml`, only applies if `engine` is `ollama`).
* `--ollama-url URL`: Specify the Ollama API endpoint URL (overrides `config.yml`, only applies if `engine` is `ollama`).
* `--gemini-api-key KEY`: Provide your Gemini API key via command line (overrides `config.yml`, only applies if `engine` is `gemini`). Use quotes if your key contains special characters.
* `-l LANG [LANG ...]` or `--lang LANG [LANG ...]`: Specify preferred subtitle language codes (overrides `config.yml`).

**Examples:**

1.  **Process a single video using the default engine (from `config.yml`):**
    ```bash
    python app.py -u "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    ```

2.  **Process a playlist using the Gemini engine (API key must be in `config.yml` or provided):**
    ```bash
    python app.py --engine gemini -p "https://www.youtube.com/playlist?list=PL..."
    ```

3.  **Process a playlist using Gemini, overriding the API key and output file:**
    ```bash
    python app.py --engine gemini --gemini-api-key "AIzaSy..." -p "https://www.youtube.com/playlist?list=PL..." -o gemini_summaries.txt
    ```

4.  **Process a single video using Ollama with a specific model, preferring Spanish subtitles:**
    ```bash
    python app.py --engine ollama -m mistral:latest -u "https://www.youtube.com/watch?v=..." -l es en
    ```

## Output

The script appends the results for each processed video to the specified output file (default: `output.txt` from `config.yml` or via `-o`). Each entry follows this format: