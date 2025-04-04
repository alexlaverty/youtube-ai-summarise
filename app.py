import argparse
import requests
import re
import sys
import os # Added for file path operations
from typing import Tuple, List, Optional, Dict, Any
# Import Playlist along with YouTube
from pytubefix import YouTube, Playlist
from pprint import pprint
import time # Added for potential rate limiting/pauses

# --- Configuration ---
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "codeqwen:latest" # CHANGE THIS to your preferred Ollama model
DEFAULT_LANG_PREFERENCE = ['en', 'a.en']
DEFAULT_OUTPUT_FILE = "output.txt"
# Optional: Add a small delay between processing videos in a playlist
PLAYLIST_VIDEO_DELAY_SECONDS = 1

# --- Helper Functions ---

def clean_subtitle_text(srt_text: str) -> str:
    """
    Cleans SRT subtitle text by removing timestamps, sequence numbers,
    and common annotations like [Music] or [Applause]. Joins lines
    into a single coherent block of text.
    """
    # Remove sequence numbers
    cleaned = re.sub(r'^\d+\s*$', '', srt_text, flags=re.MULTILINE)
    # Remove timestamps
    cleaned = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\s*', '', cleaned, flags=re.MULTILINE)
    # Remove common annotations (e.g., [Music], [Applause], (Laughter)) - case insensitive
    cleaned = re.sub(r'\[.*?\]|\(.*?\)', '', cleaned, flags=re.IGNORECASE)
    # Remove HTML tags just in case (like <i>, <b>)
    cleaned = re.sub(r'<.*?>', '', cleaned)
    # Consolidate multiple newlines/spaces into single spaces
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
        # Adding a check to ensure pytubefix doesn't hang on invalid URLs sometimes
        # Note: This might add a small overhead but increases robustness
        try:
            yt = YouTube(youtube_url)
            # Accessing title early forces connection/validation
            video_title = yt.title
        except Exception as init_err:
             raise RuntimeError(f"Failed to initialize YouTube object for {youtube_url}. It might be invalid, private, or unavailable. Error: {init_err}")

        print(f"Processing video: '{video_title}' ({youtube_url})")
        # Optional: Reduce verbosity for playlist processing
        # print("Available captions:")
        # pprint(yt.captions.keys())

        caption_to_fetch = None
        selected_lang_code = None
        for lang_code in lang_preference:
            caption = yt.captions.get(lang_code)
            if caption:
                # print(f"Found preferred caption: '{lang_code}'")
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

        # print(f"Fetching subtitles for language code: '{selected_lang_code}'...")
        raw_subtitles_srt = caption_to_fetch.generate_srt_captions()

        if not raw_subtitles_srt:
             raise ValueError(f"Subtitle generation for '{selected_lang_code}' returned empty for video '{video_title}'.")

        # print("Cleaning subtitles...")
        cleaned_subtitles = clean_subtitle_text(raw_subtitles_srt)

        if not cleaned_subtitles:
            raise ValueError(f"Subtitles were empty after cleaning for video '{video_title}'.")

        return video_title, cleaned_subtitles

    except Exception as e:
        # Re-raise specific errors or generalize
        if isinstance(e, (ValueError, RuntimeError)):
             raise e # Pass through our specific errors
        else:
             # Catch other potential pytube/network errors
             raise RuntimeError(f"Failed to extract/clean subtitles from {youtube_url}: {e}")


def generate_key_points_with_ollama(
    subtitles: str,
    video_title: str,
    model_name: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL
    ) -> str:
    """
    Sends cleaned subtitles to a local Ollama instance to generate key points.
    (Prompt remains the same as before, ensure it meets your needs)
    """
    try:
        # system_prompt = (
        #     "You are an expert assistant specialized in analyzing video transcripts. "
        #     "Your task is to identify and list the main key points discussed in a video, "
        #     "based *only* on the provided subtitles transcript. "
        #     "Format the output as a concise, easy-to-read bulleted list. Ensure each point is distinct and informative. Provide at least 10 key points"
        # )
        # user_prompt = (
        #     f"I have the subtitles from a YouTube video titled \"{video_title}\". "
        #     "I don't have time to watch the video. Please analyze the following subtitle text "
        #     "and provide a bulleted list of the key points or main topics discussed. "
        #     "Focus on the core message and important information presented.\n\n"
        #     "--- Subtitle Transcript ---\n"
        #     f"{subtitles}\n"
        #     "--- End Transcript ---\n\n"
        #     "Key points:"
        # )
        # --- Optimized Prompt ---
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
            "Focus on the core message and important information presented. Provide at least 10 key points from the subtitles.\n\n"
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
            #print(f"Successfully generated key points for '{video_title}'.")
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
        description="Extract YouTube subtitles and generate key points using Ollama for a single video or a playlist.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Group for mutually exclusive arguments: either --url or --playlist
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
        help="Name of the Ollama model to use."
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="URL for the Ollama API endpoint."
    )
    parser.add_argument(
        "-l", "--lang",
        nargs='+',
        default=DEFAULT_LANG_PREFERENCE,
        help="Preferred language codes for subtitles, in order of priority."
    )

    args = parser.parse_args()

    urls_to_process = []
    is_playlist = False

    # --- Determine Video URLs ---
    if args.playlist:
        is_playlist = True
        try:
            print(f"Fetching playlist information from: {args.playlist}")
            pl = Playlist(args.playlist)
            # Accessing title forces loading playlist metadata
            print(f"Processing playlist: '{pl.title}' ({len(pl.video_urls)} videos)")
            if not pl.video_urls:
                 print(f"Error: Playlist URL {args.playlist} seems valid but contains no videos.", file=sys.stderr)
                 sys.exit(1)
            urls_to_process = list(pl.video_urls) # Get all video URLs
        except Exception as e:
            print(f"Error: Failed to process playlist URL {args.playlist}: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.url:
        urls_to_process.append(args.url)
    else:
        # This case should not happen due to the mutually exclusive group, but added for safety
        print("Error: You must provide either --url or --playlist.", file=sys.stderr)
        sys.exit(1)

    print(f"\n--- Starting YouTube Subtitle Summarization ---")
    print(f"Output will be saved to: {args.output_file}")
    print(f"Using Ollama model: {args.model} at {args.ollama_url}")
    print(f"Preferred subtitle languages: {args.lang}")

    total_videos = len(urls_to_process)
    processed_count = 0
    error_count = 0

    # --- Process Each Video URL ---
    for i, video_url in enumerate(urls_to_process):
        print(f"\n[{i+1}/{total_videos}] Processing URL: {video_url}")
        try:
            # 1. Extract and Clean Subtitles
            video_title, subtitles = extract_subtitles(video_url, lang_preference=args.lang)
            # print(f"Subtitles extracted and cleaned successfully for '{video_title}'.")
            # print(f"Total cleaned subtitle length: {len(subtitles)} characters.")

            if len(subtitles) < 50:
                print(f"Warning: Cleaned subtitles for '{video_title}' are very short ({len(subtitles)} chars). Summary might be limited.")
                if len(subtitles) == 0:
                    raise ValueError("Subtitles are empty after cleaning. Cannot proceed.")

            # 2. Generate Key Points with Ollama
            key_points = generate_key_points_with_ollama(
                subtitles=subtitles,
                video_title=video_title,
                model_name=args.model,
                ollama_url=args.ollama_url
            )

            # 3. Save Results to File
            save_output(args.output_file, video_title, key_points, video_url)
            processed_count += 1

        except (ValueError, RuntimeError) as e:
            # Catch errors specific to subtitle extraction or Ollama processing for *this* video
            print(f"Error processing video {video_url}: {e}", file=sys.stderr)
            error_count += 1
        except Exception as e:
            # Catch any other unexpected errors for *this* video
            print(f"An unexpected error occurred processing video {video_url}: {e}", file=sys.stderr)
            error_count += 1
        finally:
            # Optional: Add a delay between videos in a playlist
            if is_playlist and i < total_videos - 1: # Don't sleep after the last video
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