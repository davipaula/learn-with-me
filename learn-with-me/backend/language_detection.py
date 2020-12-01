import spacy
from spacy_langdetect import LanguageDetector


class LanguageDetection:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    def get_language(self, text):
        doc = self.nlp(text)

        return doc._.language
    
if __name__ == "__main__":
    ll = LanguageDetection()
    print(ll.get_language("This is a sentence"))
