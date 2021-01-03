from fastapi import APIRouter

from app.model import tf_idf

words_router = r = APIRouter()


@r.get("/topic/{topic}")
def by_topic(topic: str):
    most_important_words = tf_idf.run(topic)

    return {"topic": topic, "words": most_important_words}
