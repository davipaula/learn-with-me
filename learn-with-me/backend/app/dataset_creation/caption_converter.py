import json
import logging
import os
from typing import List

import webvtt
from tqdm import tqdm

from app.video_list_retriever.video_caption import VideoCaption

# Bad hack to access the files
# TODO fix it
os.chdir(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_FOLDER = "../data/raw/caption/"
OUTPUT_FOLDER = "../data/processed/caption/dataset.jsonl"

logger = logging.getLogger(__name__)
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.NOTSET, format=LOG_FORMAT)


def run():
    logger.info("Loading caption files")
    caption_files = [
        os.fsdecode(file)
        for file in os.listdir(RAW_DATA_FOLDER)
        if os.fsdecode(file).endswith(".vtt")
    ]
    logger.info("Caption files loaded")

    captions = create_captions_dataset(caption_files)

    _save_as_json(captions)

    logger.info("Dataset created successfully")


def create_captions_dataset(file_names: List[str]) -> List[VideoCaption]:
    captions = []

    logger.info("Creating captions dataset")
    for file_name in tqdm(file_names):
        raw_captions = webvtt.read(RAW_DATA_FOLDER + file_name)

        video_captions = [
            {
                key: value
                for key, value in (
                    ("start", caption.start),
                    ("end", caption.end),
                    ("text", caption.text),
                    ("normalized_text",),
                )
            }
            for caption in raw_captions
        ]

        captions.append(VideoCaption(title=file_name, captions=video_captions))

    return captions


def _save_as_json(video_captions):
    logger.info("Saving captions dataset")
    with open(OUTPUT_FOLDER, "w") as output_file:
        for video_caption in tqdm(video_captions):
            json.dump(video_caption.__dict__, output_file)
            output_file.write("\n")


if __name__ == "__main__":
    run()
