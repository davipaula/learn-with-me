import json
import os
from collections import defaultdict
from typing import Dict, List

from tqdm import tqdm

import webvtt

from learn_with_me.backend.video_list_retriever.video_caption import VideoCaption

RAW_DATA_FOLDER = "/Users/dnascimentodepau/Documents/personal/projects/learn-with-me/learn_with_me/data/raw/caption/"

OUTPUT_FOLDER = "/Users/dnascimentodepau/Documents/personal/projects/learn-with-me/learn_with_me/data/processed/caption/dataset.jsonl"


def run():
    caption_files = [
        os.fsdecode(file)
        for file in os.listdir(RAW_DATA_FOLDER)
        if os.fsdecode(file).endswith(".vtt")
    ]

    captions = create_captions_dataset(caption_files)

    _save_as_json(captions)


def create_captions_dataset(file_names: List[str]) -> List[VideoCaption]:
    captions = []

    for file_name in tqdm(file_names):
        raw_captions = webvtt.read(RAW_DATA_FOLDER + file_name)

        video_captions = [
            {
                key: value
                for key, value in (
                    ("start", caption.start),
                    ("end", caption.end),
                    ("text", caption.text),
                )
            }
            for caption in raw_captions
        ]

        captions.append(VideoCaption(title=file_name, captions=video_captions))

    return captions


def _save_as_json(video_captions):
    with open(OUTPUT_FOLDER, "w") as output_file:
        for video_caption in tqdm(video_captions):
            json.dump(video_caption.__dict__, output_file)
            output_file.write("\n")


if __name__ == "__main__":
    run()
