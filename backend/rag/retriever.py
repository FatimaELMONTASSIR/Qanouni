"""
Recherche vectorielle des articles pertinents (Qdrant Cloud).
"""

from __future__ import annotations

import os

from backend.db import qdrant_client
from backend.rag.embedder import embed


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """
    Retourne les articles les plus proches sémantiquement de la requête.

    Chaque élément contient : code_name, article_number, article_text, score.
    """
    if top_k is None:
        top_k = int(os.environ.get("TOP_K_RESULTS", "5"))
    q = (query or "").strip()
    if not q:
        return []
    try:
        query_vector = embed(q)
    except Exception as exc:
        raise RuntimeError(
            f"Impossible de calculer l'embedding de la requête : {exc}"
        ) from exc
    try:
        raw = qdrant_client.search(query_vector, top_k=top_k)
    except Exception as exc:
        raise RuntimeError(
            f"Impossible d'interroger Qdrant Cloud : {exc}"
        ) from exc
    results: list[dict] = []
    for row in raw:
        payload = row.get("payload") or {}
        results.append(
            {
                "code_name": str(payload.get("code_name", "")),
                "article_number": str(payload.get("article_number", "")),
                "article_text": str(payload.get("article_text", "")),
                "score": float(row.get("score", 0.0)),
            }
        )
    return results
