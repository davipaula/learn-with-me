import json
from typing import Dict, List

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)

import pandas as pd


def get_most_frequent_words(n: int = 10):
    captions = get_captions_as_json()

    captions_text = get_captions_text(captions)

    # TODO save this model
    tf_idf_vectors = TfidfVectorizer(analyzer="word", min_df=0)
    tf_idf_result = tf_idf_vectors.fit_transform(captions_text)

    # TODO combine captions from the same subject to identify the most important words of this subject
    caption_test = sort_tf_idf_vectors(tf_idf_result[15].tocoo())

    # TODO load model from file
    top_n_words = get_top_n_words_from_vector(
        tf_idf_vectors.get_feature_names(), caption_test, n
    )

    print(top_n_words)


def get_tf_idf_transformer(cv_matrix):
    tf_idf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tf_idf_transformer.fit(cv_matrix)

    return tf_idf_transformer


def get_count_vectorizer(normalized_dataset):
    count_vectorizer = CountVectorizer()

    return (
        count_vectorizer.fit_transform(normalized_dataset),
        count_vectorizer.get_feature_names(),
    )


def get_captions_text(captions: List[Dict[str, List]]) -> List[str]:
    normalized_dataset = []

    for caption in captions:
        normalized_captions = [sentence["text"] for sentence in caption[1]]

        normalized_dataset.append(" ".join(normalized_captions))

    return normalized_dataset


def build_tf_idf_matrix(dataset):
    tf = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=0)
    tf.fit_transform(dataset)

    df = pd.DataFrame()

    return tf


def get_captions_as_json() -> List[Dict]:
    with open("../../data/processed/caption/dataset.jsonl", "r") as json_file:
        json_list = list(json_file)

    return [json.loads(json_str) for json_str in json_list]


def consolidate_sentences(first_sentence, second_sentence):
    return f"{first_sentence} {second_sentence}"


def sort_tf_idf_vectors(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)

    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def get_top_n_words_from_vector(feature_names, sorted_vectors, n: int = 10):
    top_n_vectors = sorted_vectors[:n]

    return {feature_names[index]: round(score, 3) for index, score in top_n_vectors}


if __name__ == "__main__":
    print(get_most_frequent_words())
