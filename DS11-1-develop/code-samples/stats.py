import numpy as np


def z_stat(sample1, sample2):
    n_a = len(sample1)
    n_b = len(sample2)

    p_a = float(sum(sample1)) / n_a
    p_b = float(sum(sample2)) / n_b
    P = float(p_a * n_a + p_b * n_b) / (n_a + n_b)

    return (p_a - p_b) / np.sqrt(P * (1 - P) * (1. / n_a + 1. / n_b))
