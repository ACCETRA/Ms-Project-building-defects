"""Check local links in the project's first-party Markdown documents."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [ROOT / "README.md", ROOT / "PROJECT_READINESS.md", *sorted((ROOT / "docs").glob("*.md"))]
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def main() -> None:
    missing: list[str] = []
    checked = 0
    for document in DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        for raw_target in LINK.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            checked += 1
            resolved = (document.parent / unquote(target)).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")

    print(f"documents={len(DOCUMENTS)} local_links={checked} missing={len(missing)}")
    for item in missing:
        print(item)
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
