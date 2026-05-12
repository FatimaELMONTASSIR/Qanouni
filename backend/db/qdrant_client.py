"""
Client Qdrant Cloud : collection vectorielle des articles juridiques.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

VECTOR_SIZE = 384
COLLECTION_DEFAULT = "lexmaroc_articles"

_client: QdrantClient | None = None


def _collection_name() -> str:
    return os.environ.get("QDRANT_COLLECTION", COLLECTION_DEFAULT)


def _get_client() -> QdrantClient:
    global _client
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    if not url or not api_key:
        raise ValueError(
            "QDRANT_URL et QDRANT_API_KEY doivent être définis dans l'environnement."
        )
    if _client is None:
        _client = QdrantClient(url=url, api_key=api_key, timeout=60)
    return _client


def _ensure_collection(client: QdrantClient, name: str) -> None:
    """Crée la collection si elle n'existe pas (vecteur 384, distance cosinus)."""
    try:
        names = client.get_collections().collections
        existing = {c.name for c in names}
        if name in existing:
            return
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(
                size=VECTOR_SIZE,
                distance=qm.Distance.COSINE,
            ),
        )
    except UnexpectedResponse as exc:
        raise RuntimeError(
            f"Erreur Qdrant lors de la création ou la vérification de la collection : {exc}"
        ) from exc


def _stable_point_id(code_name: str, article_number: str, source_file: str) -> str:
    """Identifiant déterministe pour éviter les doublons lors des réingestions."""
    raw = f"{code_name}|{article_number}|{source_file}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def upsert_articles(articles_with_embeddings: list[dict[str, Any]]) -> int:
    """
    Insère ou met à jour des points dans Qdrant.

    Chaque élément attendu :
    - vector : liste de 384 flottants
    - payload : dict avec code_name, article_number, article_text, source_file
    """
    if not articles_with_embeddings:
        return 0
    try:
        client = _get_client()
        name = _collection_name()
        _ensure_collection(client, name)
        points: list[qm.PointStruct] = []
        for item in articles_with_embeddings:
            vec = item.get("vector")
            payload = dict(item.get("payload") or {})
            if vec is None or len(vec) != VECTOR_SIZE:
                raise ValueError("Chaque article doit avoir un vecteur de taille 384.")
            cid = payload.get("code_name", "")
            anum = str(payload.get("article_number", ""))
            src = payload.get("source_file", "")
            pid = _stable_point_id(cid, anum, src)
            points.append(
                qm.PointStruct(id=pid, vector=list(vec), payload=payload)
            )
        client.upsert(collection_name=name, points=points, wait=True)
        return len(points)
    except UnexpectedResponse as exc:
        raise RuntimeError(
            f"Erreur Qdrant lors de l'upsert des articles : {exc}"
        ) from exc


def search(query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    """
    Recherche par similarité cosinus.

    Retourne une liste de dicts : id, score, payload.
    """
    if len(query_vector) != VECTOR_SIZE:
        raise ValueError("Le vecteur de requête doit être de dimension 384.")
    try:
        client = _get_client()
        name = _collection_name()
        _ensure_collection(client, name)
        res = client.search(
            collection_name=name,
            query_vector=list(query_vector),
            limit=top_k,
            with_payload=True,
        )
        out: list[dict[str, Any]] = []
        for hit in res:
            out.append(
                {
                    "id": hit.id,
                    "score": float(hit.score),
                    "payload": dict(hit.payload or {}),
                }
            )
        return out
    except UnexpectedResponse as exc:
        raise RuntimeError(
            f"Erreur Qdrant lors de la recherche vectorielle : {exc}"
        ) from exc
