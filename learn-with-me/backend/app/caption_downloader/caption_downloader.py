import json
import os
from typing import List

from tqdm import tqdm

# from learn_with_me.backend.caption_downloader.language_detection import (
#     LanguageDetection,
# )
#
# language_detection = LanguageDetection()


def run(
    video_ids_path: str, base_url: str, lang: str, should_check_english: bool = False
) -> None:
    video_ids = get_video_ids(video_ids_path, should_check_english)

    print("Downloading video_captions")
    download_captions(video_ids, base_url, lang)

    # print("Downloading audios")
    # download_audios(video_ids, base_url, lang)


def download_audios(video_ids: List[str], base_url: str, lang: str) -> None:
    os.chdir("../../data/raw/audio/")

    for video_id in video_ids:
        url = base_url + video_id
        download_cmd = [
            "youtube-dl",
            "--write-sub",
            "--extract-audio",
            "--audio-format 'mp3'",
            "-o",
            f"{video_id}.%(ext)s",
            url,
        ]
        os.system(" ".join(download_cmd))


def download_captions(video_ids: List[str], base_url: str, lang: str) -> None:
    os.chdir("../../data/raw/caption/")

    for video_id in video_ids:
        url = base_url + video_id
        file_name = video_id.rsplit("/talks/")[1]
        download_cmd = [
            "youtube-dl",
            "--skip-download",
            "--write-sub",
            "--sub-lang",
            lang,
            "-o",
            f"'{file_name}.%(ext)s'",
            url,
        ]
        os.system(" ".join(download_cmd))


def get_video_ids(video_ids_path: str, should_check_english: bool = False) -> List[str]:
    with open(video_ids_path, "r") as results_file:
        results = list(results_file)

    video_list = [json.loads(result) for result in results]

    # if should_check_english:
    #     return [video["video_id"] for video in video_list if is_english(video["title"])]

    return [video["id"] for video in video_list]


# # It is not possible to make sure that the videos downloaded from Youtube are in English
# # This function attempts to check if the title is in English. However, the results of this
# # check were not satisfactory. This function is not needed for TED talks
# def is_english(title: str) -> bool:
#     language = language_detection.get_language(title)
#
#     return language["language"] == "en"


if __name__ == "__main__":
    run()
