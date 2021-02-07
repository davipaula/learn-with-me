import json
import logging
import time
from typing import List

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from page_url import Video

logger = logging.getLogger(__name__)
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _save_as_json(topic_videos: List[Video], output_path: str):
    with open(output_path, "w") as output_file:
        print("Saving files")
        for video in tqdm(topic_videos):
            json.dump(video.__dict__, output_file)
            output_file.write("\n")


def run() -> None:
    topics = [
        "technology",
        "entertainment",
        "design",
        "business",
        "science",
        "global issues",
    ]

    logger.info("Obtaining video from topics")
    all_topics_videos = []
    for topic in topics:
        _url = f"https://www.ted.com/talks?language=en&sort=popular&topics%5B%5D={topic}&page="

        number_of_result_pages = _get_number_of_result_pages(_url)
        topic_videos = get_videos_from_topic(
            _url, number_of_result_pages, topic
        )

        all_topics_videos.extend(topic_videos)

    logger.info(f"{len(all_topics_videos)} videos found. Saving results.")
    _save_as_json(all_topics_videos, "../../data/processed/ted_results.jsonl")

    logger.info(f"Process finished successfully")


def _clean_link(link: str) -> str:
    return link.rsplit("?language=en")[0]


def get_videos_from_topic(
    base_url: str, number_of_result_pages: int, topic: str
) -> List[Video]:
    talks_header_class = "f-w:700 h9 m5"

    logger.info(f"Getting videos from {topic} topic")
    videos = []
    for current_page in tqdm(range(1, number_of_result_pages + 1)):
        time.sleep(5)
        current_url = f"{base_url}{current_page}"

        parsed_page = get_parsed_page(current_url)

        talks_header_tags = parsed_page.find_all(
            "h4", {"class": talks_header_class}
        )

        talks_a_tags = [
            talk_header.find("a") for talk_header in talks_header_tags
        ]

        videos_in_page = [
            Video(
                title=a_tag.text, id=_clean_link(a_tag.get("href")), topic=topic
            )
            for a_tag in talks_a_tags
        ]

        videos.extend(videos_in_page)

    return videos


def _get_number_of_result_pages(url: str) -> int:
    parsed_page = get_parsed_page(url)

    tags_class = "pagination__item pagination__link"
    number_of_result_pages = parsed_page.find_all("a", {"class": tags_class})

    if len(number_of_result_pages) == 0:
        return 0

    return int(number_of_result_pages[-1].text)


def get_parsed_page(url: str):
    response = requests.get(url)

    return BeautifulSoup(response.content, "html.parser")


if __name__ == "__main__":
    run()
