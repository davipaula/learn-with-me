import json
import os
from collections import defaultdict
from typing import Dict, List

from tqdm import tqdm

import webvtt

RAW_DATA_FOLDER = "/Users/dnascimentodepau/Documents/personal/projects/learn-with-me/learn_with_me/data/raw/caption/"

OUTPUT_FOLDER = "/Users/dnascimentodepau/Documents/personal/projects/learn-with-me/learn_with_me/data/processed/caption/dataset.jsonl"


def run():
    caption_files = [
        os.fsdecode(file)
        for file in os.listdir(RAW_DATA_FOLDER)
        if os.fsdecode(file).endswith(".vtt")
    ]

    captions = create_captions_dataset(caption_files)

    save_as_json(captions)


def create_captions_dataset(filenames: List[str]) -> Dict[list]:
    captions = defaultdict(list)

    for file in tqdm(filenames):
        raw_captions = webvtt.read(RAW_DATA_FOLDER + file)

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

        captions[file] = video_captions

    return captions


def save_as_json(captions):
    with open(OUTPUT_FOLDER, "w") as output_file:
        print("Saving files")
        for line in tqdm(captions.items()):
            json.dump(line, output_file)
            output_file.write("\n")


if __name__ == "__main__":
    run()
