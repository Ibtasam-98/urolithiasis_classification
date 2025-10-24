import os

class Config:
    SEED = 123
    DATA_DIR = "dataset"
    IMG_SIZE = (64, 64)
    BATCH_SIZE = 32
    SAVE_FIGS_DIR = "figs"
    MODELS_DIR = "saved_models"

    def __init__(self):
        # Create directories
        os.makedirs(self.SAVE_FIGS_DIR, exist_ok=True)
        os.makedirs(self.MODELS_DIR, exist_ok=True)

config = Config()