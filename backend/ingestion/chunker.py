"""
Découpage des textes juridiques en articles via expressions régulières.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Motif de début d'article : Article 12, Art. 12, ARTICLE 12, Article premier, etc.
_ARTICLE_START = re.compile(
    r"(?msi)^[ \t]*"
    r"(?:"
    r"ARTICLE\s+(?P<num_romain>\d+)"
    r"|Article\s+(?P<premier>premier|PREMIER|Première|PREMIÈRE)"
    r"|Article\s+(?P<num_word>\d+)(?:\s*(?:er|ère|e|re))?"
    r"|Art\.\s*(?P<num_art>\d+)"
    r")"
    r"\b"
)


def _normalize_article_number(match: re.Match[str]) -> str:
    if match.group("premier"):
        return "1"
    for key in ("num_romain", "num_word", "num_art"):
        g = match.group(key)
        if g:
            return str(int(g))
    return "?"


def _code_name_from_filename(filename: str) -> str:
    """
    Déduit le nom affiché du code à partir du nom de fichier PDF.
    """
    stem = Path(filename).stem.lower().replace(" ", "_")
    mapping = {
        "code_travail": "Code du Travail",
        "travail": "Code du Travail",
        "code_penal": "Code Pénal",
        "penal": "Code Pénal",
        "pénal": "Code Pénal",
        "code_famille": "Code de la Famille",
        "famille": "Code de la Famille",
        "code_commerce": "Code de Commerce",
        "commerce": "Code de Commerce",
        "coc": "Code de Commerce",
        "code_obligations": "Code des Obligations et Contrats",
        "obligations": "Code des Obligations et Contrats",
        "cob": "Code des Obligations et Contrats",
        "code_procedure_civile": "Code de Procédure Civile",
        "procedure_civile": "Code de Procédure Civile",
        "cpc": "Code de Procédure Civile",
    }
    for key, label in mapping.items():
        if key in stem:
            return label
    return stem.replace("_", " ").title()


def chunk_document(
    code_name: str, full_text: str, source_file: str
) -> list[dict[str, Any]]:
    """
    Découpe le texte en articles individuels.

    Retourne une liste de dicts :
    code_name, article_number, article_text, source_file.
    """
    if not full_text.strip():
        return []
    matches = list(_ARTICLE_START.finditer(full_text))
    if not matches:
        return []
    chunks: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        block = full_text[start:end].strip()
        article_number = _normalize_article_number(m)
        if not block:
            continue
        chunks.append(
            {
                "code_name": code_name,
                "article_number": article_number,
                "article_text": block,
                "source_file": source_file,
            }
        )
    return chunks


def chunk_from_pdf_map(pdf_map: dict[str, str]) -> list[dict[str, Any]]:
    """Applique le découpage à tous les fichiers extraits."""
    all_chunks: list[dict[str, Any]] = []
    for filename, text in pdf_map.items():
        code = _code_name_from_filename(filename)
        all_chunks.extend(chunk_document(code, text, filename))
    return all_chunks
