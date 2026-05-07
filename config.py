IMG_SIZE = 112          # smaller = faster, still accurate
SEQUENCE_LENGTH = 16
BATCH_SIZE = 8          # larger batch = faster epochs
EPOCHS = 20
LEARNING_RATE = 1e-3    # higher start for OneCycleLR
NUM_CLASSES = 2
NUM_WORKERS = 4         # parallel data loading