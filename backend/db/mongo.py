"""
Stockage local des articles en JSON (remplace MongoDB Atlas).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_STORE_PATH = Path(os.environ.get("JSON_STORE_PATH", "articles.json"))
_cache: list[dict[str, Any]] | None = None


def _load() -> list[dict[str, Any]]:
    global _cache
    if _cache is None:
        if _STORE_PATH.exists():
            with open(_STORE_PATH, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        else:
            _cache = []
    return _cache


def _save(articles: list[dict[str, Any]]) -> None:
    global _cache
    _cache = articles
    with open(_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)


def insert_articles(articles: list[dict[str, Any]]) -> int:
    if not articles:
        return 0
    try:
        existing = _load()
        existing.extend(articles)
        _save(existing)
        return len(articles)
    except Exception as exc:
        raise RuntimeError(f"Erreur JSON insertion : {exc}") from exc


def get_article(code_name: str, article_number: str | int) -> dict[str, Any]:
    try:
        for a in _load():
            if a.get("code_name") == code_name and str(a.get("article_number")) == str(article_number):
                return dict(a)
        return {}
    except Exception as exc:
        raise RuntimeError(f"Erreur JSON lecture : {exc}") from exc


def count_articles() -> int:
    try:
        return len(_load())
    except Exception as exc:
        raise RuntimeError(f"Erreur JSON comptage : {exc}") from exc


def delete_articles_by_source_files(source_files: list[str]) -> int:
    if not source_files:
        return 0
    try:
        articles = _load()
        before = len(articles)
        articles = [a for a in articles if a.get("source_file") not in source_files]
        _save(articles)
        return before - len(articles)
    except Exception as exc:
        raise RuntimeError(f"Erreur JSON suppression : {exc}") from exc