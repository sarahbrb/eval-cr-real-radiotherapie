"""
Minimal punctuation of radiotherapy reports before linguistic evaluation

The extracted reports often chain sentences and sections without any final
punctuation ("... le01/01/1950 Indication: radiothérapie ..."). This script
ONLY inserts:
  - a period at the end of an unpunctuated sentence, right before the next one
  - a space between "le/du/au" and a glued date ("le01/01/1950" -> "le 01/01/1950")
Nothing is removed or modified, no line break is added: removing the inserted
characters gives back the original text (checked for every file).

A period is inserted (if the previous character is not already . ! ? : ; ,)
before:
  - a section heading ("Indication:", "Préparation au traitement", ...)
    or an administrative block ("Ref :", "Adressé à :", "Signé électroniquement")
  - a "Label :" item ("Sein gauche :", "OAR cœur :", "Poumon droit:")
  - a word that usually starts a sentence in the corpus ("Pas", "Poursuite" ...)
unless the sentence being closed would only hold one word, a bare heading, or
a sub-heading right after a colon ("Volumes : Cibles CTV sein droit: ...").

Usage:
    python scripts/punctuate_reports.py --input_dir "2. Corps Extraits" \\
        --output_dir "3. CRs ponctues"
"""

import argparse
import re
import sys
from pathlib import Path

L = "A-Za-zÀ-ÖØ-öø-ÿœŒ­"     # letters (soft hyphen kept inside words)
UPPER = "A-ZÉÈÀÂÎÔŒ"
PUNCT_BEFORE = ".!?:;,…"         # no period added after these

# ── Section headings seen in the corpus (with or without a colon) ─────────────
HEADINGS = [
    "Indications?", "Préparation (?:au |à la |du )?(?:traitement|radiothérapie)",
    "Réalisation du traitement", "Modalités (?:de|du) traitement", "Technique",
    "Dosimétrie", "Volumes cibles et organes à risque", "Volumes cibles",
    "Volumes", "Organes? à risque", "Tolérance(?: au traitement| à l['’]irradiation)?",
    "Thérapeutique ultérieure et surveillance", "Suite de la prise en charge",
    "Décision en fin de traitement", "En fin de traitement", "En fin traitement",
    "Prévisions?", "Décision (?:de )?R ?C ?P", "Histologie",
    "Traitement post ?op", "Antécédents", "Histoire de la maladie",
    # administrative blocks
    "Adressé à", "Copie à", "R[ée]f\\.? ?:", "COMPTE[\\s-]*RENDU", "Compte[\\s-]*rendu",
    "Signé électroniquement", "Vous pouvez joindre",
    "Vous êtes médecin", "Pour recevoir les courriers",
]
HEADING_ALT = "|".join(HEADINGS)
HEADING_RE = re.compile(rf"(?<=\S)(\s+)(?=(?:{HEADING_ALT})(?:\s?:|\s[{UPPER}0-9]|\s))")
BARE_HEADING_RE = re.compile(rf"(?:{HEADING_ALT})\s?:?")

# ── Generic "Label :" item ────────────────────────────────────────────────────
# The label starts at the rightmost Title-case word (or known acronym) among
# the 4 words before the colon, with no punctuation in between.
LABEL_COLON_RE = re.compile(r"\s?:(?!\d)")
LABEL_ACRONYMS = {"OAR", "CT", "CTV", "PTV", "GTV"}
LABEL_STOP = {"mGy", "cGy", "Gy", "MeV", "MV", "Cm",          # units
              "Toshiba", "Aquilon", "Aquilion", "Varian", "Siemens"}  # devices
TITLE_WORD_RE = re.compile(rf"[{UPPER}][a-zà-öø-ÿœ'’\-­]+")

# ── Words that start a sentence in the corpus ─────────────────────────────────
# (capitalised after ". " in ≥ 15 reports and used as a common word elsewhere;
# proper nouns and "Docteur" excluded)
STARTERS = """
Préparation La Dosimétrie Tolérance Le Volumes Sein Il Mise Double Pas Je Un
Thérapeutique Technique Décision Les Scanner Réalisation Bilan Étalement Vous
Radiothérapie Suite Consultation Chimiothérapie Modalités Lit
Épidermite Densitométrie Elle Prescription Schéma Hormonothérapie Dossier
Nécessité Aire Instauration Surveillances Cette Patiente Paroi Absence Poursuite
Une Reprise Mammographie Contrôle Imagerie Lymphographie Reverra Biopsie Cœur
Échographie Poumon Doses Aires Ostéodensitométrie Rendez Nous Surimpression
Biologie Présence Surdosage Sera Va Puis Carcinome Imageries Séquence Masse
Surveillance Antécédents Quelques Diagnostic Prochaine Cet Introduction Ce
Dépistage Histoire Bonne Aura Suivi Mastectomie Après Repositionnement Volume
Arrêt Palpation Lors Prurit Site Chirurgie Tumorectomie Douleurs Bon Canal Poids
Organe Antériorité Chaîne Devant Boost Test Œdème Histologie Complément Prise
Érythème Sectoriectomie Exérèse Première Biopsies Lésion Poumons Dose Marge
Coeur Explications Avis Proposition Stade Apparition Deuxième Microbiopsies
Séances Persistance Cependant Prochain Diminution Discussion Dysphagie Remise
Amélioration Verra Curage Premier Suites Irradiation Rougeur Réadaptation
Substitution Dermite Découverte Refus Analyse Préconisation Scintigraphie
Kinésithérapie Prévisions Contrôles Echographie
""".split()
STARTER_RE = re.compile(rf"(?<=[{L}0-9%)])(\s+)(?=(?:{'|'.join(STARTERS)})\b)")
# No period after these words ("le Docteur", "du Sein", "par Radiothérapie" ...)
NO_BREAK_AFTER = {
    "le", "la", "les", "l", "du", "de", "des", "d", "un", "une", "au", "aux",
    "à", "par", "pour", "sur", "avec", "en", "et", "ou", "chez", "dans", "sans",
    "selon", "entre", "dr", "pr", "docteur", "professeur", "mme", "m",
}
# Existing sentence ends (already punctuated): they only reset the sentence
SENT_END_RE = re.compile(r"(?<=[.!?])(\s+)(?=\S)")
# Date glued to the previous word: "le01/01/1950" -> "le 01/01/1950"
# (only after le/du/au: "pN0/2" is a TNM stage, not a date)
GLUED_DATE_RE = re.compile(r"\b(?:le|du|au)(?=\d{1,2}/\d{1,2}(?:/\d{2,4})?\b)")


def _after_no_break_word(text: str, pos: int) -> bool:
    prev = re.findall(rf"[{L}]+", text[max(0, pos - 30):pos])
    return bool(prev) and prev[-1].lower() in NO_BREAK_AFTER


def _label_cuts(text: str) -> set:
    cuts = set()
    for m in LABEL_COLON_RE.finditer(text):
        lo = max(0, m.start() - 80)
        words = list(re.finditer(r"\S+", text[lo:m.start()]))[-5:]
        if words and words[-1].group().startswith("("):
            continue                             # "(CTDI:" is not a label
        start = None
        for i, wm in enumerate(words):
            w = wm.group()
            if i == 0:
                continue                         # words[0] = word before label
            prev = words[i - 1].group()
            if re.search(r"[,;:(/+=.!?]$", prev):
                start = None                     # punctuation inside the window
                continue
            if w not in LABEL_STOP and (
                    TITLE_WORD_RE.fullmatch(w) or w in LABEL_ACRONYMS):
                start = lo + wm.start()
        if start is not None:
            ws = re.search(r"\s+$", text[:start])
            if ws and not _after_no_break_word(text, start):
                cuts.add(ws.start())
    return cuts


def punctuate(text: str):
    """Return (new_text, [(position in `text`, inserted string), ...])."""
    cuts = {m.start(1) for m in HEADING_RE.finditer(text)}
    cuts |= _label_cuts(text)
    cuts |= {m.start(1) for m in STARTER_RE.finditer(text)
             if not _after_no_break_word(text, m.start(1))}
    resets = {m.start(1) for m in SENT_END_RE.finditer(text)}

    inserted, sent_start = [(m.end(), " ") for m in GLUED_DATE_RE.finditer(text)], 0
    for pos in sorted(cuts | resets):
        sentence = text[sent_start:pos]
        if pos in cuts and pos not in resets:
            after_colon = sentence.rsplit(":", 1)[-1].split() if ":" in sentence else None
            if (text[pos - 1] in PUNCT_BEFORE
                    or len(sentence.split()) <= 1
                    or BARE_HEADING_RE.fullmatch(sentence.strip())
                    # sub-heading right after a colon ("Volumes : Cibles CTV ...")
                    or (after_colon is not None and len(after_colon) == 1
                        and TITLE_WORD_RE.fullmatch(after_colon[0]))):
                continue
            inserted.append((pos, "."))
        sent_start = pos

    inserted.sort()
    out, last = [], 0
    for pos, s in inserted:
        out += [text[last:pos], s]
        last = pos
    out.append(text[last:])
    return "".join(out), inserted


def remove_insertions(text: str, inserted: list) -> str:
    """Inverse of punctuate(): used to check that nothing else changed."""
    out, last, shift = [], 0, 0
    for pos, s in inserted:
        p = pos + shift                          # shift by insertions already made
        assert text[p:p + len(s)] == s
        out.append(text[last:p])
        last = p + len(s)
        shift += len(s)
    return "".join(out + [text[last:]])


def main():
    parser = argparse.ArgumentParser(description="Insert missing sentence periods.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    src, dst = Path(args.input_dir), Path(args.output_dir)
    files = sorted(src.glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files found in: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    n_periods = n_spaces = 0
    altered = []
    for path in files:
        raw = path.read_text(encoding="utf-8")
        out, inserted = punctuate(raw)
        if remove_insertions(out, inserted) != raw:
            altered.append(path.name)
        n_periods += sum(s == "." for _, s in inserted)
        n_spaces  += sum(s == " " for _, s in inserted)
        (dst / path.name).write_text(out, encoding="utf-8")

    print(f"{len(files)} report(s) written to {dst}")
    print(f"  periods inserted: {n_periods} (mean {n_periods / len(files):.1f} per report)")
    print(f"  spaces inserted before glued dates: {n_spaces}")
    print(f"  text altered other than insertions: {len(altered)} file(s) {altered[:5]}")


if __name__ == "__main__":
    main()
