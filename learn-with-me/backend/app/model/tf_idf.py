import json
import logging
import os
from typing import Dict, List

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

# Bad hack to access the files
# TODO fix it
os.chdir(os.path.dirname(os.path.abspath(__file__)))

TED_RESULTS_PATH = "../data/processed/ted_results.jsonl"
CAPTION_DATASET_PATH = "../data/processed/caption/dataset.jsonl"

logger = logging.getLogger(__name__)
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def run(topic: str, n: int = 10):
    captions = get_captions_as_json()

    # TODO preprocess text: noise removal, stop-word removal, lemmatization
    captions_text = get_captions_text(captions)

    # TODO save this model
    logger.info("Training model over corpus")
    tf_idf_vectors = TfidfVectorizer(
        analyzer="word", min_df=0, max_df=0.8, stop_words="english"
    )
    # tf_idf_result = tf_idf_vectors.fit_transform(captions_text)
    tf_idf_vectors.fit(captions_text)

    logger.info(f"Getting most frequent words for topic {topic}")

    topic_text = get_topic_text(topic)
    tf_idf_topic = tf_idf_vectors.transform([topic_text])
    sorted_vectors = sort_tf_idf_vectors(tf_idf_topic.tocoo())

    top_n_words = get_top_n_words_from_topic(
        tf_idf_vectors.get_feature_names(), sorted_vectors, n
    )

    logger.info(top_n_words)

    return top_n_words


def get_topic_text(topic: str) -> str:
    with open(TED_RESULTS_PATH, "r") as json_file:
        json_lines = list(json_file)

    ted_videos = [json.loads(json_line) for json_line in json_lines]

    # Need to normalize file name and video title. Currently they are different
    # TODO think about how to store the id the right way
    videos_in_topic = [
        video["id"].rsplit("/talks/")[1]
        for video in ted_videos
        if video["topic"] == topic
    ]

    # Need to change `dataset.jsonl` to the right json format (key, value). Currently the key is the file name
    captions = get_captions_as_json()

    # TODO create a Caption data class and add methods to avoid this absurd data manipulation
    captions_from_topic = [
        caption
        for caption in captions
        if caption["title"].rsplit(".en.vtt")[0] in videos_in_topic
    ]

    # TODO check why length of videos_in_topic and captions_from_topic do not match
    # print(len(captions_from_topic))

    topic_text = get_captions_text(captions_from_topic)

    return " ".join(topic_text)


def get_captions_text(captions: List[Dict[str, List]]) -> List[str]:
    normalized_dataset = []

    for caption in captions:
        normalized_captions = [
            sentence["text"] for sentence in caption["captions"]
        ]

        normalized_dataset.append(" ".join(normalized_captions))

    return normalized_dataset


def get_captions_as_json() -> List[Dict]:
    with open(CAPTION_DATASET_PATH, "r") as json_file:
        json_list = list(json_file)

    return [json.loads(json_str) for json_str in json_list]


def sort_tf_idf_vectors(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)

    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def get_top_n_words_from_topic(feature_names, sorted_vectors, n: int = 10):
    top_n_vectors = sorted_vectors[:n]

    return {
        feature_names[index]: round(score, 3) for index, score in top_n_vectors
    }


if __name__ == "__main__":
    run()
