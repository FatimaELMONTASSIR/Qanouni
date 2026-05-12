"""
Génération d'embeddings avec sentence-transformers (chargement unique).
"""

from __future__ import annotations

import os
from typing import Any

_model: Any = None


def _get_model() -> Any:
    """Charge le modèle d'embeddings une seule fois (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        model_name = os.environ.get(
            "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        _model = SentenceTransformer(model_name)
    return _model


def embed(text: str) -> list[float]:
    """Encode un texte en vecteur dense de dimension 384."""
    if not text or not str(text).strip():
        raise ValueError("Le texte à encoder ne peut pas être vide.")
    model = _get_model()
    vector = model.encode(str(text), convert_to_numpy=True)
    return [float(x) for x in vector.tolist()]


def embed_many(texts: list[str]) -> list[list[float]]:
    """
    Encode plusieurs textes en un seul appel au modèle (ingestion).

    Retourne une liste de vecteurs dans le même ordre que l'entrée.
    """
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [[float(x) for x in row.tolist()] for row in vectors]
