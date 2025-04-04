# YouTube Video Summarizer using Ollama

This Python script extracts subtitles from a YouTube video or an entire playlist, sends the transcript to a locally running Ollama instance, and generates a bulleted list of key points, saving the results to a text file.

## Features

*   Processes single YouTube videos or entire playlists.
*   Automatically fetches available subtitles (prioritizes English by default).
*   Cleans subtitle text by removing timestamps and annotations.
*   Uses a local Ollama instance for generating summaries (key points).
*   Configurable Ollama model, API endpoint, and subtitle language preference.
*   Appends results (Title, URL, Key Points) to a specified output file.
*   Includes basic error handling and progress reporting.

## Prerequisites

*   **Python 3.x:** Ensure you have Python installed.
*   **Ollama:** You need a running Ollama instance accessible from where you run the script. Download and install it from [https://ollama.com/](https://ollama.com/). Make sure the desired model (default: `codeqwen:latest`) is pulled (`ollama pull codeqwen:latest`).
*   **Required Python Libraries:** Install the necessary libraries using pip.

## Installation

1.  **Clone the repository (or download the script):**
    ```bash
    # If you have a git repo setup
    # git clone <your-repo-url>
    # cd <your-repo-directory>

    # Or just ensure you have app.py and requirements.txt
    ```

2.  **Install dependencies:**
    Create a `requirements.txt` file with the following content if you don't have one:
    ```txt
    requests
    pytubefix
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you don't have a `requirements.txt` file, you can install them directly: `pip install requests pytubefix`)*

## Configuration

The script uses the following defaults, which can be overridden via command-line arguments:

*   **Ollama API URL:** `http://localhost:11434/api/chat` (change with `--ollama-url`)
*   **Ollama Model:** `codeqwen:latest` (change with `--model`) - *Make sure this model is available in your Ollama instance!*
*   **Subtitle Language Preference:** `['en', 'a.en']` (English, Auto-generated English) (change with `--lang`)
*   **Output File:** `output.txt` (change with `--output-file`)

## Usage

Make sure your Ollama instance is running before executing the script.

**Command-line arguments:**

*   `-u URL` or `--url URL`: Process a single YouTube video URL.
*   `-p URL` or `--playlist URL`: Process all videos in a YouTube playlist URL.
*   `-o FILE` or `--output-file FILE`: Specify the output file path (default: `output.txt`).
*   `-m MODEL` or `--model MODEL`: Specify the Ollama model name (default: `codeqwen:latest`).
*   `--ollama-url URL`: Specify the Ollama API endpoint URL (default: `http://localhost:11434/api/chat`).
*   `-l LANG [LANG ...]` or `--lang LANG [LANG ...]`: Specify preferred subtitle language codes (default: `en a.en`).

**Examples:**

1.  **Process a single video using default settings:**
    ```bash
    python app.py -u "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    ```

2.  **Process a playlist, saving to a different file and using a different model:**
    ```bash
    python app.py -p "https://www.youtube.com/playlist?list=PL..." -o my_summaries.txt -m llama3:latest
    ```

3.  **Process a single video, preferring Spanish subtitles first:**
    ```bash
    python app.py -u "https://www.youtube.com/watch?v=..." -l es en
    ```

## Output

The script appends the results for each processed video to the specified output file (default: `output.txt`). Each entry follows this format:

```
--- Video Start ---
Title: [Video Title]
URL: [Video URL]

Key Points:
- [Generated Key Point 1]
- [Generated Key Point 2]
- ...
--- Video End ---

```

## Notes

*   The quality of the summary depends heavily on the quality of the available subtitles and the capabilities of the chosen Ollama model.
*   Ensure your Ollama instance is running and accessible at the specified URL.
*   Some videos may not have subtitles available, or only auto-generated ones. The script attempts to handle this but may fail if no captions are found.
*   Processing long playlists can take a significant amount of time.
