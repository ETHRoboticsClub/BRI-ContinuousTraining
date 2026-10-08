import yaml

from src.dataset import build_dataset
from src.preprocessing import preprocess

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))
    build_dataset(params, preprocess)
