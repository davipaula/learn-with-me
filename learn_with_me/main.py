from fastapi import FastAPI

from backend.tf_idf import get_most_frequent_words

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello world"}


@app.get("/frequent-words")
def frequent_words():
    return get_most_frequent_words()
