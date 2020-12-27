import json
import os
from collections import defaultdict
from typing import List

from tqdm import tqdm
# from utils import save_as_json

import webvtt

RAW_DATA_FOLDER = (
    "/Users/dnascimentodepau/Documents/personal/projects/learn_with_me/data/raw/"
)

OUTPUT_FOLDER = (
    "/Users/dnascimentodepau/Documents/personal/projects/learn_with_me/data/processed/dataset.jsonl"
)


def convert_vtt(filenames: List[str]) -> None:
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

    with open(OUTPUT_FOLDER, "w") as output_file:
        print("Saving files")
        for line in tqdm(captions.items()):
            json.dump(line, output_file)
            output_file.write("\n")


if __name__ == "__main__":
    filenames_vtt = [
        os.fsdecode(file)
        for file in os.listdir(RAW_DATA_FOLDER)
        if os.fsdecode(file).endswith(".vtt")
    ]

    print(len(filenames_vtt))

    convert_vtt(filenames_vtt)
