SEED = 42
TZ = "America/Toronto"

WINDOW_HOURS = 72           # sliding window length
BATCH_SIZE = 256
EPOCHS = 25
LR = 1e-3
PATIENCE = 5                # early stopping patience

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15             # test is the remainder

DEVICE = "cuda"             # will fall back to cpu automatically if not available
