"""
Lightweight training logger.

Writes per-episode metrics to a CSV file and (optionally) to
TensorBoard for live monitoring during training.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class TrainingLogger:
    """
    Logs scalar metrics per episode to CSV + stdout.

    Parameters
    ----------
    log_dir  : directory where the CSV (and optional TB events) are saved
    agent_name : label used in file names and stdout headers
    use_tensorboard : whether to also write TensorBoard events
    """

    def __init__(
        self,
        log_dir: str,
        agent_name: str = "agent",
        use_tensorboard: bool = False,
        seed: int = 0,
    ):
        self.log_dir    = Path(log_dir)
        self.agent_name = agent_name
        self.seed       = seed
        self.log_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self.log_dir / f"{agent_name}_seed{seed}.csv"
        self._csv_file = open(csv_path, "w", newline="")
        self._writer   = None          # lazy init on first log

        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(
                    log_dir=str(self.log_dir / f"tb_{agent_name}_seed{seed}")
                )
            except ImportError:
                pass

        self._start_time = time.time()

    # ------------------------------------------------------------------
    def log(self, episode: int, metrics: Dict[str, Any], verbose: bool = False) -> None:
        """Write one row of metrics."""
        if self._writer is None:
            fieldnames = ["episode", "elapsed_s"] + sorted(metrics.keys())
            self._writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._writer.writeheader()

        row = {
            "episode":   episode,
            "elapsed_s": round(time.time() - self._start_time, 2),
            **metrics,
        }
        self._writer.writerow(row)
        self._csv_file.flush()

        if self._tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, episode)

        if verbose:
            parts = [f"Ep {episode:5d}"] + [
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(metrics.items())
            ]
            print(" | ".join(parts))

    def close(self) -> None:
        self._csv_file.close()
        if self._tb_writer is not None:
            self._tb_writer.close()
