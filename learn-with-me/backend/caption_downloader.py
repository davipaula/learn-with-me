import json
import os
from typing import List

from tqdm import tqdm

from backend.language_detection import LanguageDetection

language_detection = LanguageDetection()


def main() -> None:
    video_ids = get_english_video_ids()

    download_captions(video_ids)


def download_captions(video_ids: List[str]) -> None:
    base_url = 'https://www.youtube.com/watch?v='
    lang = "en"

    os.chdir("../data/raw/")

    for video_id in tqdm(video_ids):
        url = base_url + video_id
        download_cmd = ["youtube-dl", "--skip-download", "--write-sub",
                        "--sub-lang", lang, url]
        os.system(" ".join(download_cmd))


def get_english_video_ids() -> List[str]:
    with open('../data/processed/results.jsonl', 'r') as results_file:
        results = list(results_file)

    video_list = [json.loads(result) for result in results]
    video_ids = [video["video_id"] for video in video_list if is_english(video["title"])]

    return video_ids


def is_english(title: str) -> bool:
    language = language_detection.get_language(title)

    return language["language"] == "en"


if __name__ == "__main__":
    main()
