# -*- coding: utf-8 -*-
import os
import shutil
import yaml
from pathlib import Path
import torch, random
import torch.optim as optim
import torchvision
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import logging
from model import ConvNext, UNet3D, Mednext
from dataset import get_loaders
from utils.validation import segmentation_validation, plot_segmentation
from utils import utilities
from utils.losses import DiceLoss

torchvision.disable_beta_transforms_warning()

logging.basicConfig(
    encoding="utf-8", level=logging.DEBUG, format="%(levelname)s - %(message)s"
)
parser = argparse.ArgumentParser(prog="Training")

parser.add_argument("--e", type=int, default=100, help="Number of epochs for training")
parser.add_argument(
    "--log", type=str, default="INFO", help="Define debug level. Defaults to INFO."
)
parser.add_argument(
    "--gpu",
    type=int,
    default=0,
    help="GPU used for training.",
)
parser.add_argument(
    "--config",
    type=str,
    help="Path to configuration file",
    default="train_skullstrip.yaml",
)


def set_seed(seed: int = 42) -> None:
    """Set seeds for the libraries numpy, random, torch and torch.cuda.

    Args:
        seed (int, optional): Seed to be used. Defaults to `42`.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.debug(f"Random seed set as {seed}")


class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""

    def __init__(
        self,
        patience=20,
        verbose=True,
        delta=0,
        monitor="val_loss",
        op_type="min",
        logger=None,
    ):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type
        self.logger = logger

        if self.op_type == "min":
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):
        score = -val_score if self.op_type == "min" else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0
        else:
            self.counter += 1
            logging.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def print_and_update(self, val_score):
        """print_message when validation score decrease."""
        if self.verbose:
            logging.info(
                f"{self.monitor} optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...",
            )
        self.val_score_min = val_score


class TrainNetwork:
    """Train a neural network based on PyTorch architecture.

    Args:
        args (dict): Dictionary containing user-specified settings.
        config (dict): Dictionary containing settings set in a yaml-config file.
    """

    def __init__(self, args: dict, config: dict) -> None:
        """
        Initializes the TrainNetwork class.

        Args:
            args (dict): A dictionary containing the command-line arguments.
            config (dict): A dictionary containing the configuration settings.

        Attributes:
            args (dict): A dictionary containing the command-line arguments.
            config (dict): A dictionary containing the configuration settings.
            train_path (str): The path to the training data.
            val_path (str): The path to the validation data.
            base_output (Path): The base output path.
            init_lr (float): The initial learning rate.
            epochs (int): The number of epochs.
            model_name (str): The name of the model.
            device (torch.device): The device to be used for training.

        Returns:
            None
        """
        self.scheduler = None
        self.optimizer = None
        self.early_stopping = None
        self.loss = None
        self.val_loader = None
        self.train_loader = None
        self.save_folder = None
        self.metric_list = None
        self.metric = None
        self.total_train_loss = None
        self.lr = None
        self.args: dict = args
        self.config: dict = config
        self.train_path: str = config["train_path"]
        self.val_path: str = config["val_path"]
        self.base_output: Path = config["base_output"]
        self.init_lr: float = config["lr"]
        self.epochs: int = args.e
        self.model_name: str = f"{config['model']}_{config['lr']}_{config['comment']}"
        self.device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        )
        TrainNetwork._init_network(self, self.config)

    def _init_network(self, configuration: dict) -> None:
        """
        Initializes the network based on the provided configuration.

        Args:
            configuration (dict): A dictionary containing the configuration parameters.

        Raises:
            ValueError: If an invalid model is selected.

        Returns:
            None
        """
        network_classes = {
            "convnext": ConvNext,
            "unet3d": UNet3D,
            "mednext": Mednext
        }
        network_class = network_classes.get(configuration["model"])
        if network_class is None:
            raise ValueError("Select valid model!")
        self.model = network_class().to(self.device)

    @utilities.timer
    def train_fn(self) -> None:
        """Train function.

        Calculates loss per batch, performs backpropagation and optimizer step.

        Args:
            self: self object of the class.

        Returns:
            None.
        """
        loop = tqdm(self.train_loader)
        self.total_train_loss = 0

        for batch_idx, data_dict in enumerate(loop):
            data = data_dict["image"].to(device=self.device, non_blocking=True)
            targets = data_dict["mask"].to(device=data.device, non_blocking=True)

            predictions = self.model(data.float())

            loss = self.loss(predictions, targets)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            self.total_train_loss += loss.item()

            if torch.isnan(loss):
                logging.warning("-- Loss nan --")
                break

            loop.set_postfix(loss=loss.item())

        self.scheduler.step()
        self.lr = self.scheduler.get_last_lr()[0]
        self.total_train_loss = self.total_train_loss / len(self.train_loader)

    @utilities.timer
    def validation(self) -> None:
        """Performs validation after each epoch.

        This method saves one batch of the validation set in the save-folder
        and calculates the dice score as well as the validation loss.
        The results are logged to Weights & Biases.

        Args:
            self: Instance of `TrainNetwork` class.

        Returns:
            None
        """
        self.model.eval()
        total_validation_loss = 0
        total_metrics = None
        loop = tqdm(self.val_loader)

        n = random.randint(0, len(self.val_loader) - 1)

        for batch_idx, data_dict in enumerate(loop):
            data = data_dict["image"].to(device=self.device, non_blocking=True)

            # forward
            with torch.no_grad():
                predictions = self.model(data.float())

            targets = data_dict["mask"].to(device=data.device, non_blocking=True)

            predictions_sum = predictions.float()
            loss = self.loss(predictions, targets)

            loop.set_postfix(loss=loss.item())

            val_metrics = segmentation_validation(predictions_sum, targets)
            if total_metrics is None:
                # initialize total_metrics with keys from val_metrics and all values set to 0
                total_metrics = {key: 0 for key in val_metrics.keys()}
            for key in total_metrics.keys():
                total_metrics[key] += val_metrics[key].item()

            total_validation_loss += loss.item()

            if batch_idx == n and self.epoch % 1 == 0:
                plot_segmentation(
                    data,
                    predictions_sum,
                    targets,
                    self.epoch,
                    self.save_folder,
                )

        total_validation_loss /= len(self.val_loader)
        val_metrics = {
            key: total / len(self.val_loader) for key, total in total_metrics.items()
        }
        logging.info(f"Val-loss: {total_validation_loss:.3f}")
        logging.info(f"DSC: {val_metrics['dsc']:.3f} | IoU: {val_metrics['iou']:.3f}")
        stop_metric = val_metrics["dsc"]

        self.early_stopping(stop_metric)

        if stop_metric > self.metric:
            self.metric = stop_metric
            torch.save(self.model, Path(self.save_folder) / self.model_name)

        self.model.train()

    @utilities.timer
    def main(self) -> None:
        """Performs all necessary training steps by initiating the epoch loop
        and saves the trained model at the end.

        Args:
            config (dict): Dictionary containing predefined settings used by several external functions.
        """

        train_paths = pd.read_csv(self.train_path)
        val_paths = pd.read_csv(self.val_path)

        self.metric_list = []
        self.save_folder = f"{self.base_output}/train_{self.model_name}"
        Path(self.save_folder).mkdir(exist_ok=True)

        logging.info(f"Device: {self.device}")
        table = utilities.count_parameters(self.model)
        logging.info(f"\n{table}")

        self.train_loader, self.val_loader = get_loaders(
            train_paths,
            val_paths,
            batch_size=2,
        )

        self.loss = DiceLoss().to(self.device)

        self.early_stopping = EarlyStopping(
            patience=20, verbose=True, monitor="dsc", op_type="max"
        )
        self.metric = 0.0
        self.lr = self.init_lr
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=0
        )

        # Copy config file to save folder
        shutil.copyfile(args.config, Path(self.save_folder, Path(args.config).name))
        # Log save folder information
        logging.info(f"Save folder: {str(self.save_folder)}")

        # Start epoch loop
        for self.epoch in range(self.epochs):
            logging.info(f"Now training epoch {self.epoch}!")
            TrainNetwork.train_fn(self)

            logging.info(f"Train-loss: {self.total_train_loss:.3f}")

            # Validate the model
            TrainNetwork.validation(self)
            if self.early_stopping.early_stop:
                logging.info("Early stopping ...")
                break


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(True)

    args = parser.parse_args()
    args.config = "configs/" + args.config

    with open(args.config, "r") as conf:
        config = yaml.safe_load(conf)

    torch.set_num_threads(5)
    try:
        set_seed(42)
        training = TrainNetwork(
            args=args,
            config=config,
        )
        logging.info(training.__repr__())
        training.main()
    except Exception as e:
        logging.exception(e)
