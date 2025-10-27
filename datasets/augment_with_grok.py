"""Augment JSONL supervision records with answers from Grok-4 Fast.

The script reads an existing dataset (for example the GitHub code harvest),
optionally injects relevant source snippets into the prompt, and calls the
xAI Grok chat completion API to obtain synthetic reference answers. The
resulting records keep all existing keys and add `reference_answer` plus a
`llm_metadata` block documenting the model invocation.

Example usage
-------------
    python augment_with_grok.py \
        --input github_examples.jsonl \
        --output github_examples_with_answers.jsonl \
        --model grok-4-fast \
        --api-base https://api.x.ai/v1 \
        --temperature 0.2

Provide the xAI token using --api-key, the XAI_API_KEY environment
variable, or by placing `xai_api_key.txt` under `../secrets/`. Records that already contain `reference_answer` are skipped by
default; pass --overwrite to regenerate.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib import error, request

DEFAULT_API_BASE = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-4-fast"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful senior engineer. Answer concisely and accurately. "
    "Use the supplied context to ground your response and explain the key "
    "file or function when relevant."
)

SECRET_FILE_NAME = "xai_api_key.txt"


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    records: List[Dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_context(record: Dict) -> str:
    parts: List[str] = []
    repo = record.get("repo")
    if repo:
        parts.append(f"Repository: {repo}")
    path = record.get("path")
    if path:
        parts.append(f"Path: {path}")
    if record.get("code"):
        parts.append("Snippet:\n" + record["code"])
    elif record.get("ground_truth"):
        targets = ", ".join(t.get("path", "unknown") for t in record["ground_truth"])
        parts.append(f"Relevant files: {targets}")
    return "\n\n".join(parts)


def locate_secret_file() -> Optional[Path]:
    workspace_root = Path(__file__).resolve().parents[3]
    candidate = workspace_root / "secrets" / SECRET_FILE_NAME
    return candidate if candidate.exists() else None


def resolve_api_key(cli_value: Optional[str]) -> Optional[str]:
    if cli_value:
        return cli_value.strip()
    env_value = os.environ.get("XAI_API_KEY")
    if env_value:
        return env_value.strip()
    secret_path = locate_secret_file()
    if secret_path:
        return secret_path.read_text(encoding="utf-8").strip()
    return None


def make_payload(
    query: str,
    context: str,
    model: str,
    temperature: float,
    system_prompt: str,
) -> Dict:
    user_content = f"Question:\n{query.strip()}"
    if context:
        user_content += "\n\nContext:\n" + context
    return {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }


def call_grok(api_base: str, api_key: str, payload: Dict) -> Dict:
    endpoint = api_base.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(endpoint, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with request.urlopen(req) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"Grok API error {exc.code}: {text}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to reach Grok API: {exc.reason}") from exc


def extract_answer(payload: Dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("Grok response missing choices array")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        raise ValueError("Grok response did not return content")
    return content.strip()


def augment_records(
    records: List[Dict],
    *,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    max_records: Optional[int],
    overwrite: bool,
    system_prompt: str,
    sleep_ms: int,
) -> List[Dict]:
    updated: List[Dict] = []
    remaining = max_records if max_records is not None and max_records > 0 else None
    for record in records:
        if remaining is not None and remaining <= 0:
            updated.append(record)
            continue
        already = record.get("reference_answer")
        if already and not overwrite:
            updated.append(record)
            continue
        query = record.get("query")
        if not query:
            updated.append(record)
            continue
        context = build_context(record)
        payload = make_payload(
            query=query,
            context=context,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
        )
        response = call_grok(api_base=api_base, api_key=api_key, payload=payload)
        answer = extract_answer(response)
        metadata = {
            "model": model,
            "temperature": temperature,
            "system_prompt": system_prompt,
            "api_base": api_base,
            "created": response.get("created"),
        }
        record["reference_answer"] = answer
        record["llm_metadata"] = metadata
        updated.append(record)
        if remaining is not None:
            remaining -= 1
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000)
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment dataset with Grok-4 Fast answers")
    parser.add_argument("--input", type=Path, required=True, help="Source JSONL dataset path")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL path")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="xAI model name (default: grok-4-fast)")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE, help="xAI API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="xAI API key (fallback: XAI_API_KEY env)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    parser.add_argument("--max-records", type=int, default=None, help="Cap the number of records to augment")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate answers even if present")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="Custom system prompt text")
    parser.add_argument("--sleep-ms", type=int, default=250, help="Delay between requests to respect rate limits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(args.api_key)
    if not api_key:
        raise SystemExit("Provide an xAI API key via --api-key or XAI_API_KEY env var")

    records = read_jsonl(args.input)
    enriched = augment_records(
        records,
        api_base=args.api_base,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        max_records=args.max_records,
        overwrite=args.overwrite,
        system_prompt=args.system_prompt,
        sleep_ms=args.sleep_ms,
    )
    write_jsonl(enriched, args.output)
    print(f"Wrote {len(enriched)} augmented records to {args.output}")


if __name__ == "__main__":
    main()
