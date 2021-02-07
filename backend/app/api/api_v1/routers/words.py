from fastapi import APIRouter

from app.model import tf_idf

words_router = r = APIRouter()


@r.get("/topics/{topic}/{number_of_words}")
def by_topic(topic: str, number_of_words: int):
    most_important_words = tf_idf.run(topic, number_of_words)

    return {"topic": topic, "words": most_important_words}


@r.get("/topic/videos/")
def get_videos_with_words():
    # topic = "business"
    # words = {
    #     "company",
    #     "want",
    #     "business",
    #     "world",
    #     "actually",
    # }

    # return {"value": get_video_captions(db)}
    return {"value": 10101101}
