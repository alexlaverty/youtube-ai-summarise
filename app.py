import argparse
import requests
import re
import sys
import os
from typing import Tuple, List, Optional, Dict, Any
from pytubefix import YouTube, Playlist
from pprint import pprint
import time
import yaml
import google.generativeai as genai

# --- Load Configuration from YAML ---
def load_config(filepath="config.yml"):
    try:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Warning: Configuration file '{filepath}' not found. Using default values.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file '{filepath}': {e}. Using default values.")
        return {}

config = load_config()

# --- Configuration ---
DEFAULT_OLLAMA_URL = config.get("ollama_url", "http://localhost:11434/api/chat")
DEFAULT_OLLAMA_MODEL = config.get("ollama_model", "codeqwen:latest")
DEFAULT_LANG_PREFERENCE = config.get("language_preference", ['en', 'a.en'])
DEFAULT_OUTPUT_FILE = config.get("output_file", "output.txt")
PLAYLIST_VIDEO_DELAY_SECONDS = config.get("playlist_video_delay_seconds", 1)
GEMINI_API_KEY = config.get("gemini_api_key", "")

# --- Helper Functions ---

def clean_subtitle_text(srt_text: str) -> str:
    """
    Cleans SRT subtitle text by removing timestamps, sequence numbers,
    and common annotations like [Music] or [Applause]. Joins lines
    into a single coherent block of text.
    """
    cleaned = re.sub(r'^\d+\s*$', '', srt_text, flags=re.MULTILINE)
    cleaned = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\[.*?\]|\(.*?\)', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<.*?>', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

# --- Core Functions ---

def extract_subtitles(youtube_url: str, lang_preference: List[str] = DEFAULT_LANG_PREFERENCE) -> Tuple[str, str]:
    """
    Extracts and cleans subtitles from a YouTube video.

    Args:
        youtube_url: The URL of the YouTube video.
        lang_preference: A list of language codes to try in order (e.g., ['en', 'a.en']).

    Returns:
        A tuple containing: (video_title, cleaned_subtitles_string).

    Raises:
        ValueError: If no suitable subtitles are found.
        RuntimeError: If there's an issue downloading or processing.
    """
    try:
        try:
            yt = YouTube(youtube_url)
            video_title = yt.title
        except Exception as init_err:
            raise RuntimeError(f"Failed to initialize YouTube object for {youtube_url}. It might be invalid, private, or unavailable. Error: {init_err}")

        print(f"Processing video: '{video_title}' ({youtube_url})")

        caption_to_fetch = None
        selected_lang_code = None
        for lang_code in lang_preference:
            caption = yt.captions.get(lang_code)
            if caption:
                caption_to_fetch = caption
                selected_lang_code = lang_code
                break

        if not caption_to_fetch:
            if yt.captions:
                fallback_caption = list(yt.captions.values())[0]
                print(f"Warning: Preferred languages {lang_preference} not found for '{video_title}'. Falling back to: '{fallback_caption.code}'")
                caption_to_fetch = fallback_caption
                selected_lang_code = fallback_caption.code
            else:
                raise ValueError(f"No subtitles available for video '{video_title}' ({youtube_url}).")

        raw_subtitles_srt = caption_to_fetch.generate_srt_captions()

        if not raw_subtitles_srt:
            raise ValueError(f"Subtitle generation for '{selected_lang_code}' returned empty for video '{video_title}'.")

        cleaned_subtitles = clean_subtitle_text(raw_subtitles_srt)

        if not cleaned_subtitles:
            raise ValueError(f"Subtitles were empty after cleaning for video '{video_title}'.")

        return video_title, cleaned_subtitles

    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError)):
            raise e
        else:
            raise RuntimeError(f"Failed to extract/clean subtitles from {youtube_url}: {e}")


def generate_key_points_with_ollama(
    subtitles: str,
    video_title: str,
    model_name: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL
    ) -> str:
    """
    Sends cleaned subtitles to a local Ollama instance to generate key points.
    """
    try:
        system_prompt = (
            "You are an expert assistant specialized in analyzing video transcripts. "
            "Your task is to identify and list the main key points discussed in a video, "
            "based *only* on the provided subtitles transcript. "
            "Format the output as a concise, easy-to-read bulleted list."
        )
        user_prompt = (
            f"I have the subtitles from a YouTube video titled \"{video_title}\". "
            "I don't have time to watch the video. Please analyze the following subtitle text "
            "and provide a bulleted list of the key points or main topics discussed. "
            "Focus on the core message and important information presented. "
            "If the title is a question attempt to answer the question based off the provided subtitle text. "
            "Provide at least 10 to 50 key points from the subtitles.\n\n"
            "--- Subtitle Transcript ---\n"
            f"{subtitles}\n"
            "--- End Transcript ---\n\n"
            "Key points:"
        )
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            # "options": { "temperature": 0.7 } # Optional parameters
        }

        print(f"Sending request to Ollama model '{model_name}' for '{video_title}'...")

        headers = {"Content-Type": "application/json"}
        response = requests.post(ollama_url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()

        if "message" in data and "content" in data["message"]:
            key_points = data["message"]["content"].strip()
            if not key_points:
                return "Ollama returned an empty response."

            print(f"Successfully generated key points for '{video_title}':\n")
            for i, line in enumerate(key_points.split("\n"), start=1):
                if line.strip():
                    print(f"{line.strip()}")

            return key_points
        elif "error" in data:
            raise RuntimeError(f"Ollama API returned an error for '{video_title}': {data['error']}")
        else:
            print(f"Warning: Unexpected response structure from Ollama for '{video_title}':")
            pprint(data)
            return data.get("response", "No summary content found in response.")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error connecting to Ollama at {ollama_url} for '{video_title}': {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to generate key points using Ollama for '{video_title}': {e}")


def generate_summary_with_gemini(subtitles: str, video_title: str, api_key: str) -> str:
    """
    Generates a detailed, nested list of key points from the subtitles using the Google Gemini API.
    """
    if not api_key:
        raise ValueError("Gemini API key is not provided in the configuration.")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro-latest') # Or 'gemini-1.0-pro'

        # --- Start: Updated Prompt ---
        prompt = f"""You are a highly detailed analysis assistant. Your task is to meticulously analyze the provided video transcript and extract the specific information discussed. Do not provide a general summary. Instead, identify the main topics and for each topic, list the key details, arguments, examples, or facts presented in the transcript.

Use *only* the information present in the transcript below.

Video Title: "{video_title}"

--- Subtitle Transcript ---
{subtitles}
--- End Transcript ---

Please structure your output as a nested bulleted list. Each top-level bullet should represent a main topic discussed. Under each main topic, use indented bullets (like '-' or '*') to list the specific points, details, arguments, examples, conclusions, or statements made about that topic in the video. Be specific about *what was said*.

Example Structure:
* Main Topic 1 discussed in the video
    - Specific detail 'a' mentioned about Topic 1 (e.g., "They mentioned the sky is blue").
    - Specific detail 'b' mentioned about Topic 1 (e.g., "An example given was X, leading to conclusion Y").
* Main Topic 2 discussed in the video
    - Specific argument 'x' presented regarding Topic 2 (e.g., "The speaker argued that...").
    - Specific fact 'y' shared about Topic 2 (e.g., "It was stated that the project started on Z date").

Detailed Key Points Extraction:"""
        # --- End: Updated Prompt ---

        print(f"Sending request to Gemini API ({model.model_name}) for '{video_title}' to extract detailed points...")
        response = model.generate_content(prompt)

        if not response.parts:
             finish_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
             if finish_reason != "SAFETY":
                 print(f"Warning: Gemini response for '{video_title}' was empty or blocked. Finish Reason: {finish_reason}", file=sys.stderr)
                 return f"Gemini API did not return details. (Reason: {finish_reason})"
             else:
                 print(f"Warning: Gemini response for '{video_title}' was blocked due to safety settings.", file=sys.stderr)
                 return "Gemini API did not return details due to safety settings."

        detailed_points = response.text.strip()

        if detailed_points:
            print(f"Successfully extracted detailed points for '{video_title}' using Gemini:\n{detailed_points}\n")
            return detailed_points
        else:
            print(f"Warning: Gemini API returned empty details string for '{video_title}'.")
            return "Gemini API returned empty details."

    except google.api_core.exceptions.NotFound as e:
         raise RuntimeError(f"Failed to generate details using Gemini API for '{video_title}': Model not found or API key issue. Check model name and API key. Original error: {e}")
    except Exception as e:
        print(f"Detailed Gemini Error Traceback for '{video_title}':\n{traceback.format_exc()}", file=sys.stderr)
        raise RuntimeError(f"Failed to generate details using Gemini API for '{video_title}': An unexpected error occurred: {e}")

def save_output(filepath: str, title: str, key_points: str, url: str):
    """Appends the video title, URL, and key points to the specified file."""
    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"--- Video Start ---\n")
            f.write(f"Title: {title}\n")
            f.write(f"URL: {url}\n\n")
            f.write("Key Points:\n")
            f.write(f"{key_points}\n")
            f.write(f"--- Video End ---\n\n")
        print(f"Saved key points for '{title}' to {filepath}")
    except IOError as e:
        print(f"Error: Failed to write to output file {filepath}: {e}", file=sys.stderr)


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Extract YouTube subtitles and generate key points/summaries using Ollama or Gemini for a single video or a playlist.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-u", "--url", help="URL of a single YouTube video.")
    input_group.add_argument("-p", "--playlist", help="URL of a YouTube playlist.")

    parser.add_argument(
        "-o", "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="File path to save the output."
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Name of the Ollama model to use (if engine is ollama)."
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="URL for the Ollama API endpoint (if engine is ollama)."
    )
    parser.add_argument(
        "-l", "--lang",
        nargs='+',
        default=DEFAULT_LANG_PREFERENCE,
        help="Preferred language codes for subtitles, in order of priority."
    )
    parser.add_argument(
        "--engine",
        choices=['ollama', 'gemini'],
        help="Override the summarization engine specified in config.yml."
    )
    parser.add_argument(
        "--gemini-api-key",
        help="Optional: Your Google Gemini API key. If not provided, the key from config.yml will be used (if engine is Gemini)."
    )

    args = parser.parse_args()

    # Determine the engine to use, prioritizing the command-line argument
    engine_to_use = args.engine if args.engine else config.get("engine", "ollama")

    urls_to_process = []
    is_playlist = False

    if args.playlist:
        is_playlist = True
        try:
            print(f"Fetching playlist information from: {args.playlist}")
            pl = Playlist(args.playlist)
            print(f"Processing playlist: '{pl.title}' ({len(pl.video_urls)} videos)")
            if not pl.video_urls:
                print(f"Error: Playlist URL {args.playlist} seems valid but contains no videos.", file=sys.stderr)
                sys.exit(1)
            urls_to_process = list(pl.video_urls)
        except Exception as e:
            print(f"Error: Failed to process playlist URL {args.playlist}: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.url:
        urls_to_process.append(args.url)
    else:
        print("Error: You must provide either --url or --playlist.", file=sys.stderr)
        sys.exit(1)

    print(f"\n--- Starting YouTube Subtitle Summarization ---")
    print(f"Output will be saved to: {args.output_file}")
    print(f"Using summarization engine: {engine_to_use}")
    if engine_to_use == 'ollama':
        print(f"Using Ollama model: {args.model} at {args.ollama_url}")
    elif engine_to_use == 'gemini':
        print(f"Using Gemini API.")
    print(f"Preferred subtitle languages: {args.lang}")

    total_videos = len(urls_to_process)
    processed_count = 0
    error_count = 0

    for i, video_url in enumerate(urls_to_process):
        print(f"\n[{i+1}/{total_videos}] Processing URL: {video_url}")
        try:
            video_title, subtitles = extract_subtitles(video_url, lang_preference=args.lang)

            if len(subtitles) < 50:
                print(f"Warning: Cleaned subtitles for '{video_title}' are very short ({len(subtitles)} chars). Summary might be limited.")
                if len(subtitles) == 0:
                    raise ValueError("Subtitles are empty after cleaning. Cannot proceed.")

            summary_text = ""
            engine_used = engine_to_use

            if engine_to_use == 'ollama':
                summary_text = generate_key_points_with_ollama(
                    subtitles=subtitles,
                    video_title=video_title,
                    model_name=args.model,
                    ollama_url=args.ollama_url
                )
            elif engine_to_use == 'gemini':
                gemini_key_to_use = args.gemini_api_key if args.gemini_api_key else GEMINI_API_KEY
                if not gemini_key_to_use:
                    raise ValueError("Gemini API key is required when using the Gemini engine. Provide it via --gemini-api-key or in config.yml.")
                summary_text = generate_summary_with_gemini(
                    subtitles=subtitles,
                    video_title=video_title,
                    api_key=gemini_key_to_use
                )

            save_output(args.output_file, video_title, summary_text, video_url, engine=engine_used)
            processed_count += 1

        except (ValueError, RuntimeError) as e:
            print(f"Error processing video {video_url}: {e}", file=sys.stderr)
            error_count += 1
        except Exception as e:
            print(f"An unexpected error occurred processing video {video_url}: {e}", file=sys.stderr)
            error_count += 1
        finally:
            if is_playlist and i < total_videos - 1:
                print(f"Waiting for {PLAYLIST_VIDEO_DELAY_SECONDS}s before next video...")
                time.sleep(PLAYLIST_VIDEO_DELAY_SECONDS)


    print("\n--- Processing Complete ---")
    print(f"Total videos attempted: {total_videos}")
    print(f"Successfully processed and saved: {processed_count}")
    print(f"Errors encountered: {error_count}")
    if error_count > 0:
        print(f"Check logs or console output for details on errors.")
    print(f"Output saved in: {args.output_file}")


if __name__ == "__main__":
    main()