import json
from typing import List

from tqdm import tqdm


def save_as_json(content: List[dict], output_path: str) -> None:
    with open(output_path, "w") as output_file:
        print("Saving files")
        for line in tqdm(content):
            json.dump(line, output_file)
            output_file.write("\n")
