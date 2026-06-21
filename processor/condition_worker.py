"""Condition grading — Ollama.

Reads a listing's free-text description (with the seller-declared ``spec_stan`` as
a hint) and infers the bike's *actual* condition, because sellers routinely list a
used bike as "Nowy" and a damaged one as "Używany". Two outputs:

  * a 1–5 quality grade  → ``offers.condition``  (5 = like-new, 1 = broken)
  * a categorical state  → used to enrich ``offers.spec_stan`` (fill a NULL, or
    mark ``Uszkodzony`` when the description shows damage)

Local Ollama only (qwen2.5:3b); no cloud, no API cost. Replaces the old HerBERT
sentiment feature, which was removed.
"""
import json
import logging

from scraper.ollama_common import append_debug_csv as _append_debug_csv, ollama_available, ollama_generate

log = logging.getLogger(__name__)

# LLM categorical state → Allegro's spec_stan label.
_STAN_LABELS = {
    "nowy": "Nowy",
    "używany": "Używany",
    "uzywany": "Używany",
    "powystawowy": "Powystawowy",
    "uszkodzony": "Uszkodzony",
}


def build_condition_prompt(title: str, description: str, spec_stan: str | None) -> str:
    """The exact prompt sent to Ollama (exposed so the debug CSV can record it)."""
    declared = spec_stan or "nieznany"
    return (
        "You assess the ACTUAL physical condition of a second-hand road bike from "
        "its Polish listing. Sellers often overstate condition (a used bike listed "
        "as 'Nowy', a damaged one as 'Używany'), so judge from what the text "
        "actually describes — wear, damage, mileage, 'na części', 'uszkodzony', "
        "'jak nowy', 'nieużywany', etc. — not from the declared state.\n\n"
        "Return ONLY a JSON object with exactly these keys:\n"
        '- "condition_grade": integer 1-5 — 5 = like new/unused, 4 = very good '
        "(minimal wear), 3 = good (normal used wear), 2 = poor (heavy wear / needs "
        "work), 1 = broken / for parts / not working.\n"
        '- "broken": true if the bike is damaged, incomplete, for parts or not '
        "working, else false.\n"
        '- "state": one of "nowy", "używany", "powystawowy", "uszkodzony".\n\n'
        f"Seller-declared state (a hint, may be wrong): {declared}\n"
        f"Title: {title or ''}\n"
        f"Description:\n{(description or '')[:2000]}\n"
    )


def classify_condition(
    title: str | None, description: str | None, spec_stan: str | None,
    debug_csv: str | None = None,
) -> dict | None:
    """Grade the bike's condition from its description.

    Returns ``{"grade": 1-5, "broken": bool, "stan": <Allegro label>|None}`` or
    None if Ollama is unreachable / the response can't be parsed into a valid
    grade. ``stan`` is the Allegro spec_stan label inferred by the model.
    """
    if not description:
        return None
    prompt = build_condition_prompt(title or "", description, spec_stan)
    raw = ollama_generate(prompt)
    result: dict = {}
    try:
        parsed = json.loads(raw)
        grade = int(parsed["condition_grade"])
        if grade < 1 or grade > 5:
            raise ValueError(f"grade out of range: {grade}")
        broken = bool(parsed.get("broken", False))
        stan = _STAN_LABELS.get(str(parsed.get("state", "")).strip().lower())
        result = {"grade": grade, "broken": broken, "stan": stan}
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        log.debug("condition parse failed: %s", exc)
        result = {}
    if debug_csv:
        _append_debug_csv(debug_csv, description, prompt, raw, result)
    return result or None
