import re
from pathlib import Path
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^]]+]\(([^)]+)\)")


def public_markdown_files() -> list[Path]:
    return [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "THIRD_PARTY_NOTICES.md",
        *sorted((PROJECT_ROOT / "docs").glob("*.md")),
    ]


def test_public_markdown_relative_links_resolve():
    broken: list[str] = []
    for document in public_markdown_files():
        text = document.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative_target = unquote(target.split("#", 1)[0])
            if relative_target and not (document.parent / relative_target).exists():
                broken.append(f"{document.relative_to(PROJECT_ROOT)} -> {target}")
    assert broken == []


def test_public_docs_do_not_contain_job_or_company_source_identifiers():
    blocked = ("104.com", "919i1", "8k12f", "1a2x6bn8kq")
    findings = []
    for document in public_markdown_files():
        text = document.read_text(encoding="utf-8").lower()
        for token in blocked:
            if token.lower() in text:
                findings.append(f"{document.relative_to(PROJECT_ROOT)}: {token}")
    assert findings == []
