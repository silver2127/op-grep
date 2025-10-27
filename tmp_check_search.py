from pathlib import Path
import json
import urllib.request
import urllib.parse

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    token = (root / "secrets" / "github_token.txt").read_text(encoding="utf-8").strip()
    params = urllib.parse.urlencode({"q": "def language:python", "per_page": 30, "page": 1})
    url = f"https://api.github.com/search/code?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"token {token}",
            "User-Agent": "op-grep-dataset-generator",
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            payload = resp.read().decode("utf-8")
    except Exception as exc:
        print(f"error: {exc}")
        return
    print(payload)


if __name__ == "__main__":
    main()
