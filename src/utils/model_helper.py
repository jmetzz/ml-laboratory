import pickle  # noqa: S403
from pathlib import Path


def load_pickle_model(filename):
    with Path(filename).open(mode="rb") as file_handler:
        model = pickle.load(file_handler)  # noqa: S301
    return model


def save_pickle_model(filename, model):
    with Path(filename).open(mode="wb") as file_handler:
        pickle.dump(model, file_handler, protocol=pickle.HIGHEST_PROTOCOL)
