from dataclasses import dataclass
from typing import Dict, List


@dataclass
class VideoCaption:
    title: str
    captions: List[Dict]
