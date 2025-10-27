"""Harvest multi-language code examples from GitHub into a JSONL dataset.

The script performs GitHub code searches across a configurable set of
languages, grabs the raw file contents, and emits JSONL records suitable
for downstream experimentation. Each record captures the language, the
repository metadata, and a truncated code sample so that tooling like
op-grep can exercise non-X4 sources. By default it only accepts
permissively licensed repositories (MIT, Apache-2.0, BSD, ISC, CC0,
Unlicense, Zlib), but the filter is configurable.

Basic usage
-----------
    python generate_github_synthetic.py \
        --output github_examples.jsonl \
        --languages python javascript go rust \
        --examples-per-language 25 \
    --allowed-license MIT --allowed-license Apache-2.0 \
        [--github-token $GITHUB_TOKEN]

Provide a GitHub personal access token via --github-token or the
GITHUB_TOKEN environment variable to avoid strict rate limits.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib import error, parse, request


DEFAULT_LANGUAGES = [
    "python",
    "javascript",
    "typescript",
    "go",
    "rust",
    "java",
    "csharp",
]

LANGUAGE_ALIASES: Dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "golang": "go",
    "c#": "C#",
    "csharp": "C#",
    "cplusplus": "C++",
    "cpp": "C++",
}

KEYWORD_HINTS: Dict[str, Sequence[str]] = {
    "python": ("def", "async", "class"),
    "javascript": ("function", "class", "async"),
    "typescript": ("interface", "class", "type"),
    "go": ("func", "package", "struct"),
    "rust": ("fn", "struct", "impl"),
    "java": ("class", "interface", "public"),
    "C#": ("class", "async", "Task"),
    "C++": ("template", "class", "namespace"),
}

MAX_CODE_CHARS = 8000
MAX_FILE_BYTES = 200_000
USER_AGENT = "op-grep-dataset-generator"
RATE_LIMIT_SAFETY = 5
DEFAULT_PERMISSIVE_LICENSES = [
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "CC0-1.0",
    "Unlicense",
    "Zlib",
]


@dataclass
class CodeRecord:
    query_id: str
    query: str
    language: str
    repo: str
    path: str
    sha: str
    github_url: str
    size: int
    license: Optional[str]
    code: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "query_id": self.query_id,
                "query": self.query,
                "language": self.language,
                "repo": self.repo,
                "path": self.path,
                "sha": self.sha,
                "github_url": self.github_url,
                "size_bytes": self.size,
                "license": self.license,
                "code": self.code,
            },
            ensure_ascii=False,
        )


class GitHubClient:
    def __init__(self, token: Optional[str], max_retries: int = 3, backoff: float = 2.0) -> None:
        self._token = token
        self._max_retries = max_retries
        self._backoff = backoff
        self._license_cache: Dict[str, Optional[str]] = {}

    def _build_request(self, url: str) -> request.Request:
        req = request.Request(url)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("User-Agent", USER_AGENT)
        if self._token:
            req.add_header("Authorization", f"Bearer {self._token}")
        return req

    def _rate_limit_wait(self, headers: Dict[str, str]) -> None:
        remaining = headers.get("X-RateLimit-Remaining")
        reset = headers.get("X-RateLimit-Reset")
        if remaining is not None and int(remaining) <= RATE_LIMIT_SAFETY and reset:
            reset_at = int(reset)
            sleep_for = max(0, reset_at - int(time.time()) + 1)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def get_json(self, endpoint: str, params: Optional[Dict[str, str]] = None) -> Tuple[Dict, Dict[str, str]]:
        url = endpoint
        if params:
            query = parse.urlencode(params)
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{query}"
        req = self._build_request(url)
        attempt = 0
        while True:
            try:
                with request.urlopen(req) as resp:
                    payload = resp.read().decode("utf-8")
                    headers = dict(resp.headers.items())
                    self._rate_limit_wait(headers)
                    return json.loads(payload), headers
            except error.HTTPError as exc:
                headers = dict(exc.headers.items()) if exc.headers else {}
                if exc.code == 403 and "X-RateLimit-Reset" in headers:
                    self._rate_limit_wait(headers)
                    continue
                attempt += 1
                if attempt > self._max_retries:
                    raise RuntimeError(f"GitHub request failed ({exc.code}): {exc.reason}") from exc
                time.sleep(self._backoff ** attempt)
            except error.URLError as exc:
                attempt += 1
                if attempt > self._max_retries:
                    raise RuntimeError(f"GitHub request failed: {exc.reason}") from exc
                time.sleep(self._backoff ** attempt)


    def get_repo_license(self, full_name: str, api_url: Optional[str] = None) -> Optional[str]:
        if full_name in self._license_cache:
            return self._license_cache[full_name]

        endpoint = api_url or f"https://api.github.com/repos/{full_name}"
        data, _ = self.get_json(endpoint)
        license_info = data.get("license") or {}
        spdx = license_info.get("spdx_id")
        if not isinstance(spdx, str) or spdx == "NOASSERTION":
            spdx = None
        else:
            spdx = spdx.upper()
        self._license_cache[full_name] = spdx
        return spdx


def resolve_language(raw: str) -> str:
    normalized = raw.lower()
    mapped = LANGUAGE_ALIASES.get(normalized)
    if mapped:
        return mapped
    return raw


def default_keywords(language: str) -> Sequence[str]:
    return KEYWORD_HINTS.get(language, ("class", "function"))


def trim_code(content: str) -> str:
    if len(content) <= MAX_CODE_CHARS:
        return content
    head = content[: MAX_CODE_CHARS // 2]
    tail = content[-MAX_CODE_CHARS // 2 :]
    return f"{head}\n\n... trimmed ...\n\n{tail}"


def build_query(keyword: str, language: str) -> str:
    safe_keyword = keyword.strip().replace(" ", "+")
    return f"{safe_keyword}+language:{language}"


def search_language_examples(
    client: GitHubClient,
    language: str,
    target: int,
    keywords: Sequence[str],
    sleep_ms: int,
    allowed_licenses: Optional[Set[str]],
    allow_unlicensed: bool,
) -> List[CodeRecord]:
    collected: List[CodeRecord] = []
    seen: set[Tuple[str, str]] = set()
    lang_label = resolve_language(language)
    queries = list(default_keywords(lang_label))
    if keywords:
        for keyword in keywords:
            sanitized = keyword.strip()
            if sanitized and sanitized not in queries:
                queries.append(sanitized)
    if not queries:
        queries = ["class"]

    for keyword in queries:
        if len(collected) >= target:
            break
        page = 1
        while len(collected) < target:
            search_url = "https://api.github.com/search/code"
            params = {
                "q": build_query(keyword, lang_label),
                "per_page": "30",
                "page": str(page),
            }
            data, _ = client.get_json(search_url, params=params)
            items = data.get("items", [])
            if not items:
                break
            for item in items:
                repo_info = item.get("repository", {})
                if not isinstance(repo_info, dict):
                    continue
                repo = repo_info.get("full_name")
                if not repo:
                    continue
                path = item["path"]
                key = (repo, path)
                if key in seen:
                    continue
                seen.add(key)
                license_id = client.get_repo_license(
                    repo,
                    api_url=repo_info.get("url"),
                )
                if license_id is None and not allow_unlicensed:
                    continue
                if allowed_licenses and license_id and license_id not in allowed_licenses:
                    continue
                metadata, _ = client.get_json(item["url"])
                if metadata.get("size", 0) > MAX_FILE_BYTES:
                    continue
                if metadata.get("encoding") != "base64" or "content" not in metadata:
                    continue
                content_bytes = base64.b64decode(metadata["content"].replace("\n", ""))
                try:
                    decoded = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    decoded = content_bytes.decode("latin-1", errors="replace")
                snippet = trim_code(decoded.strip())
                if not snippet:
                    continue
                record = CodeRecord(
                    query_id=f"GITHUB_{lang_label.upper()}_{len(collected) + 1:04d}",
                    query=f"Show the {lang_label} implementation found at `{path}` in `{repo}`.",
                    language=lang_label,
                    repo=repo,
                    path=path,
                    sha=item.get("sha", ""),
                    github_url=item.get("html_url", ""),
                    size=metadata.get("size", 0),
                    license=license_id,
                    code=snippet,
                )
                collected.append(record)
                if len(collected) >= target:
                    break
            if len(collected) >= target:
                break
            if len(items) < int(params["per_page"]):
                break
            page += 1
            time.sleep(sleep_ms / 1000)
    return collected


def write_jsonl(records: Iterable[CodeRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(record.to_json() + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-language GitHub code examples")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("github_examples.jsonl"),
        help="Path for the generated JSONL dataset (default: datasets/github_examples.jsonl)",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=DEFAULT_LANGUAGES,
        help="Languages to harvest (default: python javascript typescript go rust java csharp)",
    )
    parser.add_argument(
        "--examples-per-language",
        type=int,
        default=20,
        help="Target number of examples per language (default: 20)",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=None,
        help="GitHub token for higher rate limits (falls back to GITHUB_TOKEN env)",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="Additional search keyword applied across all languages (repeatable)",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=1000,
        help="Delay between paginated search requests in milliseconds (default: 1000)",
    )
    parser.add_argument(
        "--allowed-license",
        action="append",
        dest="allowed_licenses",
        default=None,
        help="SPDX license identifier to allow (repeatable). Defaults to a permissive set.",
    )
    parser.add_argument(
        "--allow-unlicensed",
        action="store_true",
        help="Include repositories without license metadata (default: excluded).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.github_token or os.environ.get("GITHUB_TOKEN")
    client = GitHubClient(token=token)
    if args.allowed_licenses is None:
        allowed = {lic.upper() for lic in DEFAULT_PERMISSIVE_LICENSES}
    else:
        allowed = {lic.upper() for lic in args.allowed_licenses if lic}
    allow_unlicensed = bool(args.allow_unlicensed)
    records: List[CodeRecord] = []

    for language in args.languages:
        lang_records = search_language_examples(
            client=client,
            language=language,
            target=args.examples_per_language,
            keywords=args.keyword,
            sleep_ms=args.sleep_ms,
            allowed_licenses=allowed,
            allow_unlicensed=allow_unlicensed,
        )
        records.extend(lang_records)
        print(f"Collected {len(lang_records)} {language} examples")

    if not records:
        raise SystemExit("No GitHub examples collected. Adjust languages or provide a token.")

    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
