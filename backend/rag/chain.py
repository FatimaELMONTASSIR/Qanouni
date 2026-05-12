"""
Chaîne RAG : récupération d'articles, construction du contexte, appel Anthropic.
"""

from __future__ import annotations

import os
from typing import Any

import anthropic
from anthropic import Anthropic

from backend.rag.prompt import SYSTEM_PROMPT
from backend.rag.retriever import retrieve


def _format_context(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "(Aucun article pertinent n'a été trouvé dans le corpus.)"
    parts: list[str] = []
    for s in sources:
        header = f"[{s.get('code_name', '')} - Art. {s.get('article_number', '')}]"
        body = s.get("article_text", "")
        parts.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(parts)


def ask_lexmaroc(question: str, chat_history: list[dict[str, str]]) -> dict[str, Any]:
    """
    Pose une question au modèle en s'appuyant sur la recherche vectorielle.

    Retourne : {"answer": str, "sources": list[dict]}.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "La clé API Anthropic (ANTHROPIC_API_KEY) est absente de l'environnement."
        )
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    max_tokens = int(os.environ.get("MAX_TOKENS", "1000"))

    try:
        sources = retrieve(question)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Impossible de récupérer les articles pertinents : {exc}"
        ) from exc
    context = _format_context(sources)

    user_block = (
        "Voici les articles du corpus pouvant être pertinents :\n\n"
        f"{context}\n\n"
        "---\n\n"
        f"Question de l'utilisateur : {question.strip()}"
    )

    messages: list[dict[str, str]] = []
    for turn in chat_history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_block})

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
    except anthropic.APIError as exc:
        raise RuntimeError(
            f"Erreur de l'API Anthropic lors de la génération de la réponse : {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Erreur inattendue lors de l'appel au modèle Claude : {exc}"
        ) from exc

    parts: list[str] = []
    for block in response.content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        elif getattr(block, "type", None) == "text":
            parts.append(str(getattr(block, "text", "")))
    answer = "".join(parts).strip()

    return {
        "answer": answer,
        "sources": sources,
    }
