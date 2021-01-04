import json

from sqlalchemy import Column, MetaData, String, Table, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from tqdm import tqdm

from app.core import config
from app.db import models as m
from app.db.crud import create_video_caption
from app.db.schemas import VideoCaption
from app.db.session import SessionLocal

CAPTION_DATASET_PATH = "../data/processed/caption/dataset.jsonl"


def migrate_json() -> None:
    # Read json
    # For each record in file, add to DB
    with open(CAPTION_DATASET_PATH, "r") as json_file:
        json_list = list(json_file)

    captions = [json.loads(json_str) for json_str in json_list]

    print(f"{len(captions)} captions in JSONL")

    db = SessionLocal()

    print("Inserting data")
    for caption in tqdm(captions):
        create_video_caption(
            db,
            VideoCaption(title=caption["title"], caption=caption["captions"]),
        )

    db.close()


def create_table():
    engine = create_engine(
        config.SQLALCHEMY_DATABASE_URI,
    )
    metadata = MetaData()

    video_caption = Table(
        "video_caption",
        metadata,
        Column("title", String, primary_key=True),
        Column("caption", JSONB),
    )

    print("Creating tables")
    metadata.drop_all(engine)
    metadata.create_all(engine)


def get_values():
    db = SessionLocal()
    values = db.query(m.VideoCaption).all()

    print(f"{len(values)} records inserted")


if __name__ == "__main__":
    create_table()
    migrate_json()
    get_values()
