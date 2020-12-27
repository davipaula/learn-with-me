import caption_downloader


def run() -> None:
    _base_url = "https://www.ted.com"
    _video_ids_path = "../../data/processed/ted_results.jsonl"
    _lang = "en"
    _should_check_english = False

    caption_downloader.run(_video_ids_path, _base_url, _lang, _should_check_english)


if __name__ == "__main__":
    run()
