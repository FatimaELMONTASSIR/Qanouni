"""
Connexion MongoDB Atlas et opérations sur la collection des articles.
"""

from __future__ import annotations

import os
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

_client: MongoClient | None = None


def _get_collection() -> Collection[Any]:
    """Retourne la collection `articles`, en initialisant le client si nécessaire."""
    global _client
    uri = os.environ.get("MONGO_URI")
    if not uri:
        raise ValueError(
            "La variable d'environnement MONGO_URI est manquante ou vide."
        )
    db_name = os.environ.get("MONGO_DB_NAME", "lexmaroc")
    if _client is None:
        _client = MongoClient(uri, serverSelectionTimeoutMS=15000)
    return _client[db_name]["articles"]


def insert_articles(articles: list[dict[str, Any]]) -> int:
    """
    Insère une liste d'articles dans MongoDB.

    Chaque élément doit contenir au minimum :
    code_name, article_number, article_text, source_file.
    """
    if not articles:
        return 0
    try:
        coll = _get_collection()
        coll.create_index([("code_name", 1), ("article_number", 1)])
        result = coll.insert_many(articles, ordered=False)
        return len(result.inserted_ids)
    except PyMongoError as exc:
        raise RuntimeError(
            f"Erreur MongoDB lors de l'insertion des articles : {exc}"
        ) from exc


def get_article(code_name: str, article_number: str | int) -> dict[str, Any]:
    """Récupère un article par code juridique et numéro d'article."""
    try:
        coll = _get_collection()
        doc = coll.find_one(
            {"code_name": code_name, "article_number": str(article_number)}
        )
        if doc is None:
            doc = coll.find_one(
                {"code_name": code_name, "article_number": article_number}
            )
        if doc is None:
            return {}
        doc.pop("_id", None)
        return dict(doc)
    except PyMongoError as exc:
        raise RuntimeError(
            f"Erreur MongoDB lors de la lecture d'un article : {exc}"
        ) from exc


def count_articles() -> int:
    """Retourne le nombre total de documents dans la collection articles."""
    try:
        coll = _get_collection()
        return int(coll.count_documents({}))
    except PyMongoError as exc:
        raise RuntimeError(
            f"Erreur MongoDB lors du comptage des articles : {exc}"
        ) from exc


def delete_articles_by_source_files(source_files: list[str]) -> int:
    """
    Supprime les articles correspondant aux fichiers sources indiqués.

    Utilisé avant une réingestion pour éviter les doublons dans MongoDB.
    """
    if not source_files:
        return 0
    try:
        coll = _get_collection()
        res = coll.delete_many({"source_file": {"$in": source_files}})
        return int(res.deleted_count)
    except PyMongoError as exc:
        raise RuntimeError(
            f"Erreur MongoDB lors de la suppression d'articles : {exc}"
        ) from exc
