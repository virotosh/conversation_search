from typing import Dict
from dataclasses import dataclass

import numpy as np


@dataclass
class Document:
    page_content: str
    metadata: Dict


class Generator:
    """https://stackoverflow.com/a/34073559"""
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.val = yield from self.gen
        return self.val


def cosine_similarity(X, Y) -> np.ndarray:
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


def dict_list_to_pretty_str(data: list[dict]) -> str:
    _str = ""
    if isinstance(data, dict):
        data = [data]
    if isinstance(data, list):
        for i, d in enumerate(data):
            _str += f"result {i+1}\n"
            _str += f"tile: {d['title']}\n"
            _str += f"{d['body']}\n"
            _str += f"url: {d['href']}\n"
        return ret_str
    else:
        raise ValueError("Input must be dict or list[dict]")