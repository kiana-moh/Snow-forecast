import numpy as np

def baseline_persistence(y):
    # predict tomorrow score = today's score (shift by 24h is handled by dataset alignment)
    return np.roll(y, 1)

def baseline_mean_train(y_train, n):
    return np.full((n,), float(np.mean(y_train)), dtype=np.float32)
