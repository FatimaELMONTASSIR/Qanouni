"""
Point d'entrée du pipeline d'ingestion : PDF -> MongoDB + Qdrant.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Répertoire racine du projet (dossier contenant `backend` et, par défaut, `data`)
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")
load_dotenv()

from backend.db import mongo, qdrant_client
from backend.ingestion.chunker import chunk_from_pdf_map
from backend.ingestion.pdf_extractor import extract_all_pdfs, get_data_directory
from backend.rag.embedder import embed_many


def main() -> None:
    print("Étape 1/5 : extraction des textes PDF...")
    pdf_map = extract_all_pdfs()
    if not pdf_map:
        dossier = get_data_directory()
        print(
            f"Aucun fichier PDF trouvé dans le dossier des données ({dossier}). "
            "Arrêt."
        )
        return

    print("Étape 2/5 : découpage en articles...")
    articles = chunk_from_pdf_map(pdf_map)
    if not articles:
        print(
            "Aucun article détecté. Vérifiez que les PDF contiennent des marqueurs "
            "« Article », « Art. » ou « ARTICLE »."
        )
        return

    print("Étape 3/5 : génération des embeddings...")
    texts = [a["article_text"] for a in articles]
    vectors = embed_many(texts)

    print("Étape 4/5 : écriture dans MongoDB Atlas...")
    source_files = list(pdf_map.keys())
    deleted = mongo.delete_articles_by_source_files(source_files)
    if deleted:
        print(f"  ({deleted} ancien(s) document(s) supprimé(s) pour ces fichiers.)")
    inserted = mongo.insert_articles(articles)
    print(f"  {inserted} article(s) inséré(s) dans MongoDB.")

    print("Étape 5/5 : indexation dans Qdrant Cloud...")
    batch: list[dict] = []
    for art, vec in zip(articles, vectors, strict=True):
        payload = {
            "code_name": art["code_name"],
            "article_number": str(art["article_number"]),
            "article_text": art["article_text"],
            "source_file": art["source_file"],
        }
        batch.append({"vector": vec, "payload": payload})
    n_qdrant = qdrant_client.upsert_articles(batch)
    print(f"  {n_qdrant} point(s) vectoriel(s) upsert dans Qdrant.")

    total_mongo = mongo.count_articles()
    print("\n--- Résumé ---")
    print(f"Fichiers PDF traités : {len(pdf_map)}")
    print(f"Articles indexés (lot courant) : {len(articles)}")
    print(f"Documents total dans MongoDB (collection articles) : {total_mongo}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Échec de l'ingestion : {exc}", file=sys.stderr)
        sys.exit(1)
