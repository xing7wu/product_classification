from configuration import config

from datasets import load_from_disk


def get_dataset(type='train'):
    path = str(config.PROCESSED_DATA_DIR / type)
    return load_from_disk(path)
