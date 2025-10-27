from __future__ import annotations

import json
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.model import GrepModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPERVISED_DATA = PROJECT_ROOT / "datasets" / "example_supervised.jsonl"
MODEL_PATH = PROJECT_ROOT / "model.pth"


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    device = device.lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no compatible GPU is available.")
        return torch.device("cuda")
    return torch.device("cpu")


def load_supervised_records(path: Path, limit: Optional[int] = None) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Expected supervised dataset at {path}")
    records: List[Dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError("Supervised dataset is empty; cannot train model.")
    if limit is not None and limit > 0 and len(records) > limit:
        random.Random(0).shuffle(records)
        records = records[:limit]
    return records


def build_vocab(records: List[Dict]) -> Dict[str, int]:
    vocab: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
    for record in records:
        for token in record["query"].lower().split():
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def build_label_map(records: List[Dict]) -> Dict[str, int]:
    label_to_idx: Dict[str, int] = {}
    for record in records:
        for target in record.get("ground_truth", []):
            path = target["path"]
            if path not in label_to_idx:
                label_to_idx[path] = len(label_to_idx)
    if not label_to_idx:
        raise ValueError("No ground-truth targets found to build label map.")
    return label_to_idx


def build_tool_map(records: List[Dict]) -> Dict[str, int]:
    tool_to_idx: Dict[str, int] = {}
    for record in records:
        for target in record.get("ground_truth", []):
            tool = target.get("tool")
            if tool is None:
                continue
            if tool not in tool_to_idx:
                tool_to_idx[tool] = len(tool_to_idx)
    if not tool_to_idx:
        raise ValueError("No tool labels found in supervised dataset.")
    return tool_to_idx


def main(
    num_epochs: int = 6,
    device: str | torch.device = "auto",
    data_path: Optional[Path] = None,
    max_records: Optional[int] = None,
) -> None:
    target_device = _resolve_device(device)
    dataset_path = data_path or SUPERVISED_DATA
    records = load_supervised_records(dataset_path, limit=max_records)
    vocab = build_vocab(records)
    label_to_idx = build_label_map(records)
    tool_to_idx = build_tool_map(records)

    training_samples: List[Tuple[str, int, int]] = []
    for record in records:
        query = record["query"]
        for target in record["ground_truth"]:
            path = target["path"]
            tool = target.get("tool")
            if tool is None:
                raise ValueError(
                    f"Missing tool annotation for record {record['query_id']} ({path})"
                )
            training_samples.append(
                (
                    query,
                    label_to_idx[path],
                    tool_to_idx[tool],
                )
            )

    model = GrepModel(vocab=vocab, label_to_idx=label_to_idx, tool_to_idx=tool_to_idx)
    model.to(target_device)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn_path = nn.NLLLoss()
    loss_fn_tool = nn.NLLLoss()

    for epoch in range(num_epochs):
        random.shuffle(training_samples)
        running_path_loss = 0.0
        running_tool_loss = 0.0
        for query, path_label, tool_label in training_samples:
            outputs = model(query)
            log_path = outputs["log_path"]
            log_tool = outputs["log_tool"]
            target_path = torch.tensor(
                [path_label], dtype=torch.long, device=log_path.device
            )
            target_tool = torch.tensor(
                [tool_label], dtype=torch.long, device=log_tool.device
            )

            loss_path = loss_fn_path(log_path, target_path)
            loss_tool = loss_fn_tool(log_tool, target_tool)
            loss = loss_path + loss_tool
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_path_loss += loss_path.item()
            running_tool_loss += loss_tool.item()

        avg_path_loss = running_path_loss / len(training_samples)
        avg_tool_loss = running_tool_loss / len(training_samples)
        print(
            f"epoch {epoch + 1} path-loss {avg_path_loss:.4f} "
            f"tool-loss {avg_tool_loss:.4f}"
        )

    checkpoint = {
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "label_to_idx": label_to_idx,
        "tool_to_idx": tool_to_idx,
    }
    torch.save(checkpoint, MODEL_PATH)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train the op-grep action selector model")
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Number of training epochs (default: 6)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device to use. 'auto' selects CUDA when available.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=SUPERVISED_DATA,
        help="Path to the supervised JSONL dataset (default: datasets/example_supervised.jsonl)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit the number of supervision records loaded from the dataset",
    )
    args = parser.parse_args()
    main(
        num_epochs=args.epochs,
        device=args.device,
        data_path=args.data,
        max_records=args.max_records,
    )
