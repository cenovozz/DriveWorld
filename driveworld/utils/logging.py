"""Structured logging utility with TensorBoard integration."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Unified logger with console + TensorBoard output."""

    def __init__(self, log_dir: str, name: str = "driveworld"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.log_dir / f"{name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(run_dir))

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self.logger.addHandler(handler)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_value_dict: dict, step: int) -> None:
        self.writer.add_scalars(main_tag, tag_value_dict, step)

    def log_image(self, tag: str, img_tensor, step: int) -> None:
        self.writer.add_image(tag, img_tensor, step)

    def log_images(self, tag: str, img_tensor, step: int) -> None:
        self.writer.add_images(tag, img_tensor, step)

    def close(self) -> None:
        self.writer.close()


def setup_logger(log_dir: str, name: str = "driveworld") -> Logger:
    """Create a Logger instance."""
    return Logger(log_dir, name)
