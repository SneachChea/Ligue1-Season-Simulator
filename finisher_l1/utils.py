import numpy as np


def sum_after_index(
    goal_probabilities: np.ndarray, goal_index: int
) -> np.ndarray:
    somme = sum(goal_probabilities[goal_index:])
    return np.concatenate([goal_probabilities[:goal_index], [somme]])


def get_probability(matrix: np.ndarray) -> tuple[float, float, float]:
    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))
    return home_win, draw, away_win
