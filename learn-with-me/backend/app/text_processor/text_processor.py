import logging

import spacy


logger = logging.getLogger(__name__)
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class TextProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def clean_text(self, text: str):
        doc = self.nlp(text)

        return " ".join(token.lemma_ for token in doc if not token.is_stop)

    def clean_texts(self, texts):
        clean_texts = []

        logger.info(f"Starting to process documents")
        for doc in self.nlp.pipe(
            texts,
            disable=[
                "tagger",
                "parser",
                "ner",
                "entity_linker",
                "textcat",
                "entity_ruler",
                "sentencizer",
                "merge_noun_chunks",
                "merge_entities",
                "merge_subtokens",
            ],
        ):

            clean_text = " ".join(
                token.lemma_ for token in doc if not token.is_stop
            )
            clean_texts.append(clean_text)

        logger.info(f"Finished processing documents")

        return clean_texts

