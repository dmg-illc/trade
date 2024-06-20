import json
from pathlib import Path


def open_json(path):
    with open(path) as file:
        cont = file.read()

    data_dict = json.loads(cont)
    return data_dict




def get_project_root() -> Path:
    return Path(__file__).parent.parent