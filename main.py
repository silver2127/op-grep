from argparse import ArgumentParser
from dataclasses import dataclass

import torch

from src.model import GrepModel
from src.runtime import load_model


@dataclass
class ActionPrediction:
    tool: str
    path: str
    tool_probs: torch.Tensor
    path_probs: torch.Tensor


def predict_action(model: GrepModel, query: str) -> ActionPrediction:
    with torch.no_grad():
        outputs = model(query)

    path_log = outputs["log_path"].squeeze(0)
    tool_log = outputs["log_tool"].squeeze(0)
    path_probs = path_log.exp()
    tool_probs = tool_log.exp()

    top_path_idx = torch.argmax(path_probs).item()
    top_tool_idx = torch.argmax(tool_probs).item()
    target_path = getattr(model, "idx_to_label", {}).get(top_path_idx, str(top_path_idx))
    target_tool = getattr(model, "idx_to_tool", {}).get(top_tool_idx, str(top_tool_idx))

    return ActionPrediction(
        tool=target_tool,
        path=target_path,
        tool_probs=tool_probs.detach().cpu(),
        path_probs=path_probs.detach().cpu(),
    )


def main():
    parser = ArgumentParser(description="Run op-grep inference")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device for inference. 'auto' selects CUDA when available.",
    )
    args = parser.parse_args()

    model = load_model(device=args.device)

    tests = [
        "Why does the retry loop skip logging when the proxy returns 429?",
        "How does the deploy script decide between blue/green targets?",
        "Where is the feature flag `modal_new_footer` evaluated before render?",
        "What triggers a full-text reindex when the schema changes?",
        "Where do we configure the backup retention policy?",
    ]

    idx_to_label = getattr(model, "idx_to_label", {})
    idx_to_tool = getattr(model, "idx_to_tool", {})

    def topk(format_map, probs, k=3):
        pairs = sorted(
            ((format_map.get(i, str(i)), float(p)) for i, p in enumerate(probs)),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
        return ", ".join(f"{name} ({prob:.2f})" for name, prob in pairs)

    for query in tests:
        prediction = predict_action(model, query)
        print(query)
        print(f"  predicted: {prediction.tool}:{prediction.path}")
        print(f"  top tools : {topk(idx_to_tool, prediction.tool_probs)}")
        print(f"  top paths : {topk(idx_to_label, prediction.path_probs)}")
        print()


if __name__ == "__main__":
    main()
