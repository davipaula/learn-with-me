from typing import Dict, List

import requests
from bs4 import BeautifulSoup

from learn_with_me.backend.utils import save_as_json


def run() -> None:
    _url = (
        "https://www.ted.com/talks?language=en&sort=popular&topics%5B%5D=business&page="
    )

    number_of_result_pages = get_number_of_result_pages(_url)
    link_tags = get_links(_url, number_of_result_pages)
    links = _clean_links(link_tags)

    save_as_json(links, "../../data/processed/ted_results.jsonl")

    print(f"We found {len(links)} links")
    print(links)


def get_links(url: str, number_of_result_pages: int) -> Dict[str, str]:
    talks_header_class = "f-w:700 h9 m5"

    talks_urls = {}
    for current_page in range(1, number_of_result_pages + 1):
        current_url = f"{url}{current_page}"
        parsed_page = get_parsed_page(current_url)

        talks_header_tags = parsed_page.find_all("h4", {"class": talks_header_class})

        talks_a_tags = [talk_header.find("a") for talk_header in talks_header_tags]
        page_urls = {a_tag.text: a_tag.get("href") for a_tag in talks_a_tags}

        talks_urls.update(page_urls)

    return talks_urls


def get_number_of_result_pages(url: str) -> int:
    parsed_page = get_parsed_page(url)

    tags_class = "pagination__item pagination__link"
    results_pages = parsed_page.find_all("a", {"class": tags_class})

    last_result_page = int(results_pages[-1].text)

    return last_result_page


def get_parsed_page(url: str):
    response = requests.get(url)
    parsed_page = BeautifulSoup(response.content, "html.parser")

    return parsed_page


def _clean_links(links_href: Dict[str, str]) -> List[dict]:
    links = {link[0]: link[1].rsplit("?language=en")[0] for link in links_href.items()}

    return [{"title": link[0], "video_id": link[1]} for link in links.items()]


if __name__ == "__main__":
    run()
