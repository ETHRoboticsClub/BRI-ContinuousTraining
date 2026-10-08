import yaml

from src.dataset import make_loaders
from src.model import Model
from src.trainer import set_seed, train

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))
    cfg = params["train"]
    set_seed(cfg["seed"])

    train_loader, val_loader, test_loader = make_loaders(params)
    info = train_loader.dataset.info
    model = Model(len(info["channel_names"]), len(info["target_names"]), **params["model"])

    train(
        model,
        train_loader,
        val_loader,
        cfg["out_dir"],
        test_loader,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
