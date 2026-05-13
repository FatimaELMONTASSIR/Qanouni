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
BATCH_SIZE = 100  # upload par lots pour eviter les timeouts

_client: QdrantClient | None = None


def _collection_name() -> str:
    return os.environ.get("QDRANT_COLLECTION", COLLECTION_DEFAULT)


def _get_client() -> QdrantClient:
    global _client
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    if not url or not api_key:
        raise ValueError(
            "QDRANT_URL et QDRANT_API_KEY doivent etre definis dans l'environnement."
        )
    if _client is None:
        _client = QdrantClient(url=url, api_key=api_key, timeout=120)
    return _client


def _ensure_collection(client: QdrantClient, name: str) -> None:
    """Cree la collection si elle n'existe pas (vecteur 384, distance cosinus)."""
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
            f"Erreur Qdrant creation collection : {exc}"
        ) from exc


def _stable_point_id(code_name: str, article_number: str, source_file: str) -> str:
    """Identifiant deterministe pour eviter les doublons lors des reingestions."""
    raw = f"{code_name}|{article_number}|{source_file}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def upsert_articles(articles_with_embeddings: list[dict[str, Any]]) -> int:
    """
    Insere ou met a jour des points dans Qdrant par lots de 100.

    Chaque element attendu :
    - vector  : liste de 384 flottants
    - payload : dict avec code_name, article_number, article_text, source_file
    """
    if not articles_with_embeddings:
        return 0
    try:
        client = _get_client()
        name = _collection_name()
        _ensure_collection(client, name)

        # Construction de tous les points
        points: list[qm.PointStruct] = []
        for item in articles_with_embeddings:
            vec = item.get("vector")
            payload = dict(item.get("payload") or {})
            if vec is None or len(vec) != VECTOR_SIZE:
                raise ValueError("Chaque article doit avoir un vecteur de taille 384.")
            pid = _stable_point_id(
                payload.get("code_name", ""),
                str(payload.get("article_number", "")),
                payload.get("source_file", ""),
            )
            points.append(qm.PointStruct(id=pid, vector=list(vec), payload=payload))

        # Upload par lots de BATCH_SIZE
        total = 0
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i : i + BATCH_SIZE]
            client.upsert(collection_name=name, points=batch, wait=True)
            total += len(batch)
            print(f"    Qdrant : {total}/{len(points)} points uploades...", end="\r")

        print(f"    Qdrant : {total}/{len(points)} points uploades.   ")
        return total

    except UnexpectedResponse as exc:
        raise RuntimeError(
            f"Erreur Qdrant upsert : {exc}"
        ) from exc


def search(query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    """
    Recherche par similarite cosinus.
    Retourne une liste de dicts : id, score, payload.
    """
    if len(query_vector) != VECTOR_SIZE:
        raise ValueError("Le vecteur de requete doit etre de dimension 384.")
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
        return [
            {"id": hit.id, "score": float(hit.score), "payload": dict(hit.payload or {})}
            for hit in res
        ]
    except UnexpectedResponse as exc:
        raise RuntimeError(
            f"Erreur Qdrant recherche : {exc}"
        ) from exc