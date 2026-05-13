"""
Chaîne RAG : récupération d'articles, construction du contexte, appel Gemini (Google).
"""

from __future__ import annotations

import os
import re
from typing import Any

import google.generativeai as genai

from backend.rag.prompt import PHRASE_REFUS, SYSTEM_PROMPT
from backend.rag.retriever import retrieve


def texte_assistant_pour_historique(
    answer_short: str, answer_detail: str | None
) -> str:
    """Message assistant envoyé au modèle au tour suivant (résumé + détail si présent)."""
    court = (answer_short or "").strip()
    if answer_detail and answer_detail.strip():
        return (
            f"### Résumé\n{court}\n\n### Détail\n{answer_detail.strip()}"
        )
    return court


def _parser_resume_detail(brut: str) -> tuple[str, str | None]:
    """
    Extrait résumé et détail si le modèle a respecté les sections ### Résumé / ### Détail.
    Sinon retourne le texte entier comme résumé et pas de détail.
    """
    brut = (brut or "").strip()
    if not brut:
        return "", None
    if brut.strip() == PHRASE_REFUS:
        return brut, None

    m = re.search(r"(?ms)^#{1,3}\s*D[eé]tail\s*$", brut)
    if not m:
        return brut, None
    avant = brut[: m.start()].strip()
    apres = brut[m.end() :].strip()
    if not apres:
        return brut, None

    resume = re.sub(
        r"(?is)^#{1,3}\s*R[eé]sum[eé]\s*\n?",
        "",
        avant,
        count=1,
    ).strip()
    if not resume:
        resume = avant
    return resume, apres


def _format_context(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "(Aucun article pertinent n'a été trouvé dans le corpus.)"
    parts: list[str] = []
    for s in sources:
        header = f"[{s.get('code_name', '')} - Art. {s.get('article_number', '')}]"
        body = s.get("article_text", "")
        parts.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(parts)


def _cle_gemini() -> str | None:
    """Clé API : GEMINI_API_KEY, puis GOOGLE_API_KEY (défaut Google AI Studio)."""
    return (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
        or None
    )


def _nom_modele_court(nom_api: str) -> str:
    """Transforme ``models/gemini-2.0-flash`` en ``gemini-2.0-flash``."""
    if nom_api.startswith("models/"):
        return nom_api[len("models/") :]
    return nom_api


def _selectionner_modele_gemini(cle: str, modele_env: str | None) -> str:
    """
    Choisit un identifiant de modèle valide pour ``generateContent``.

    Les noms courts (ex. ``gemini-1.5-flash``) changent selon les régions et
    versions d'API ; on interroge la liste des modèles exposés pour la clé.
    """
    genai.configure(api_key=cle)
    try:
        listed = list(genai.list_models())
    except Exception as exc:
        raise RuntimeError(
            "Impossible de lister les modèles Gemini (vérifiez la clé et le réseau) : "
            f"{exc}"
        ) from exc

    avec_generation: dict[str, str] = {}
    for entree in listed:
        methodes = getattr(entree, "supported_generation_methods", None) or []
        if "generateContent" not in methodes:
            continue
        court = _nom_modele_court(entree.name)
        avec_generation[court] = entree.name

    if not avec_generation:
        raise RuntimeError(
            "Aucun modèle Gemini compatible avec generateContent n'est disponible "
            "pour cette clé API."
        )

    pref = (modele_env or "").strip()
    if pref and pref in avec_generation:
        return pref

    ordre_fallback = (
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-002",
        "gemini-1.5-flash-001",
        "gemini-flash-latest",
    )
    for candidat in ordre_fallback:
        if candidat in avec_generation:
            return candidat

    for cle_nom in sorted(avec_generation.keys()):
        if "flash" in cle_nom.lower():
            return cle_nom

    return sorted(avec_generation.keys())[0]


def _historique_gemini(chat_history: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Convertit l'historique Streamlit au format attendu par Gemini (user / model)."""
    historique: list[dict[str, Any]] = []
    for tour in chat_history:
        role = tour.get("role", "")
        texte = (tour.get("content") or "").strip()
        if not texte:
            continue
        if role == "user":
            historique.append({"role": "user", "parts": [texte]})
        elif role == "assistant":
            historique.append({"role": "model", "parts": [texte]})
    return historique


def ask_lexmaroc(question: str, chat_history: list[dict[str, str]]) -> dict[str, Any]:
    """
    Pose une question au modèle en s'appuyant sur la recherche vectorielle.

    Retourne : answer / answer_short (résumé), answer_detail (optionnel), sources.
    """
    cle = _cle_gemini()
    if not cle:
        raise ValueError(
            "Aucune clé API Gemini : définissez GEMINI_API_KEY ou GOOGLE_API_KEY "
            "(ex. clé créée dans Google AI Studio)."
        )
    modele_demande = os.environ.get("GEMINI_MODEL", "").strip() or None
    max_tokens = int(os.environ.get("MAX_TOKENS", "2048"))

    try:
        sources = retrieve(question)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Impossible de récupérer les articles pertinents : {exc}"
        ) from exc

    context = _format_context(sources)
    consigne_retrieval = ""
    if sources:
        consigne_retrieval = (
            "\n\n[Consigne] Des extraits non vides ont été fournis ci-dessus. "
            "Tu dois répondre à partir de leur contenu (même partiel) avec citations. "
            f"N'utilise la phrase de refus « {PHRASE_REFUS} » uniquement si aucun extrait "
            "ne permet le moindre lien avec la question."
        )
    consigne_format = (
        "\n\n[Format] Réponds avec les sections ### Résumé puis ### Détail "
        "(voir instructions système). Pour la seule phrase de refus, aucune section."
    )
    bloc_utilisateur = (
        "Voici les articles du corpus pouvant être pertinents :\n\n"
        f"{context}\n\n"
        "---\n\n"
        f"Question de l'utilisateur : {question.strip()}"
        f"{consigne_retrieval}"
        f"{consigne_format}"
    )

    genai.configure(api_key=cle)
    modele = _selectionner_modele_gemini(cle, modele_demande)

    generation_config = {"max_output_tokens": max_tokens}
    modele_ia = genai.GenerativeModel(
        model_name=modele,
        system_instruction=SYSTEM_PROMPT,
        generation_config=generation_config,
    )

    historique = _historique_gemini(chat_history)
    try:
        chat = modele_ia.start_chat(history=historique)
        reponse = chat.send_message(bloc_utilisateur)
    except Exception as exc:
        raise RuntimeError(
            f"Erreur de l'API Gemini lors de la génération de la réponse : {exc}"
        ) from exc

    try:
        texte = (reponse.text or "").strip()
    except ValueError:
        texte = (
            "La réponse du modèle est vide ou a été filtrée par les règles de "
            "sécurité. Reformulez votre question ou vérifiez le corpus."
        )

    texte = _corriger_refus_abusif(chat, texte, sources)

    resume, detail = _parser_resume_detail(texte)
    # Compatibilité : « answer » = résumé court affiché par défaut
    return {
        "answer": resume,
        "answer_short": resume,
        "answer_detail": detail,
        "sources": sources,
    }


def _extraits_substantiels(sources: list[dict[str, Any]]) -> bool:
    """True si au moins un article contient un minimum de texte exploitable."""
    for s in sources:
        if len((s.get("article_text") or "").strip()) >= 40:
            return True
    return False


def _reponse_est_refus(texte: str) -> bool:
    t = texte.strip()
    return PHRASE_REFUS in t or t == PHRASE_REFUS


def _corriger_refus_abusif(
    chat: Any,
    texte: str,
    sources: list[dict[str, Any]],
) -> str:
    """
    Si le modèle renvoie la phrase de refus alors que des extraits substantiels
    ont été fournis, une seconde consigne réduit les refus erronés.
    """
    if not sources or not _extraits_substantiels(sources):
        return texte
    if not _reponse_est_refus(texte):
        return texte
    relance = (
        "Les extraits juridiques ci-dessus ont été fournis dans ton message précédent "
        "et contiennent du texte. Réponds maintenant en t'appuyant exclusivement sur ces "
        "extraits : résume ce qu'ils établissent concernant la question, avec citations "
        "(nom du code et numéro d'article). Utilise les sections ### Résumé et ### Détail. "
        f"N'écris pas la phrase : « {PHRASE_REFUS} »."
    )
    try:
        reponse2 = chat.send_message(relance)
        t2 = (reponse2.text or "").strip()
        if t2:
            return t2
    except Exception:
        pass
    return texte
