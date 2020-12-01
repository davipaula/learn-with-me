import json
from collections import defaultdict
from typing import Dict, List

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def get_most_frequent_words():
    captions = get_captions_as_json()

    normalized_dataset = normalize_dataset(captions)

    ordered_vocabulary = get_ordered_vocabulary(normalized_dataset)

    return json.dumps(ordered_vocabulary)


def get_ordered_vocabulary(normalized_dataset: Dict[str, str]) -> Dict[str, int]:
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit_transform(normalized_dataset)

    vocabulary = count_vectorizer.vocabulary_

    ordered_vocabulary = {
        word: vocabulary[word]
        for word in sorted(
            vocabulary,
            key=vocabulary.get,
            reverse=True,
        )
    }

    return ordered_vocabulary


def normalize_dataset(captions: List[Dict[str, List]]) -> Dict[str, str]:
    normalized_dataset = defaultdict(str)

    for caption in captions:
        title = caption[0]

        normalized_captions = [sentence["text"] for sentence in caption[1]]

        normalized_dataset[title] = " ".join(normalized_captions)

    return normalized_dataset


def build_tf_idf_matrix(dataset):
    tf = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=0)

    return tf.fit_transform(dataset)


def get_captions_as_json() -> List[Dict]:
    with open("./data/processed/dataset.jsonl", "r") as json_file:
        json_list = list(json_file)

    return [json.loads(json_str) for json_str in json_list]


def consolidate_sentences(first_sentence, second_sentence):
    return f"{first_sentence} {second_sentence}"


if __name__ == "__main__":
    print(get_most_frequent_words())
