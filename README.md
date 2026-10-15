# Évaluation linguistique des CRs de radiothérapie réels

Métriques linguistiques calculées sur les comptes rendus (CRs) réels de la
clinique, pour les comparer aux CRs synthétiques générés par LLM (pipeline CAG).

> **Aucune donnée de la clinique n'est versionnée.** Les CRs réels, les
> résultats par CR (dont les noms de fichiers contiennent des identifiants)
> et les journaux sont exclus par le `.gitignore` (liste blanche).

📖 **Toutes les explications** (chronologie des décisions, prétraitement règle
par règle, formules des métriques, échantillonnage, résultats complets,
interprétation, limites, questions fréquentes) :
[`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md).

## Organisation

```
docs/
  DOCUMENTATION.md          explications complètes de tout le travail
scripts/
  punctuate_reports.py      prétraitement : ajout des points et espaces manquants
  evaluate_real_reports.py  métriques par CR + métriques de corpus (1 graine)
  corpus_metrics_seeds.py   métriques de corpus sur plusieurs graines
results/
  eval_bruts_graine42_summary.csv                métriques par CR (moyenne ± écart-type sur
                                                 les 3 687 CRs) + corpus (graine 42), CRs bruts
  eval_ponctues_graine42_summary.csv             idem, CRs ponctués
  corpus_independants_graines42-46.csv           métriques de corpus, 1 ligne par tirage
                                                 indépendant (graines 42 à 46) × 2 versions
  corpus_independants_graines42-46_summary.csv   moyenne ± écart-type sur ces 5 tirages
  corpus_disjoints_melange42_5blocs.csv          métriques de corpus, 1 ligne par bloc
                                                 disjoint (mélange graine 42) × 2 versions
  corpus_disjoints_melange42_5blocs_summary.csv  moyenne ± écart-type sur ces 5 blocs
```

Convention de nommage : `bruts` = CRs tels qu'extraits, `ponctues` = après
prétraitement ; `_summary` = agrégat (moyenne ± écart-type).

Non versionnés (à placer localement) : `2. Corps Extraits/` (3 687 CRs réels,
un `.txt` UTF-8 par CR) et `3. CRs ponctues/` (sortie du prétraitement).

## Prétraitement des CRs réels

À l'extraction, la mise en forme des CRs a été perdue : chaque CR tient sur une
ligne et beaucoup de phrases s'enchaînent sans ponctuation. Le découpage en
phrases conditionne la perplexité et la lisibilité (`textstat` compte les
phrases avec `. ! ?`).

`punctuate_reports.py` **n'ajoute que** :
1. un point à la fin d'une phrase non ponctuée, avant un intitulé de section,
   un élément « Libellé : valeur » ou un mot qui débute habituellement une
   phrase dans le corpus ;
2. une espace entre `le/du/au` et une date collée (`le01/01/1950`), sans toucher
   aux stades TNM (`pN0/2`).

Rien n'est supprimé ni modifié (en-têtes et signatures conservés). Pour chaque
CR, le script vérifie qu'en retirant les caractères ajoutés on retrouve le texte
d'origine. Résultat : 33 096 points et 1 410 espaces ajoutés ; longueur médiane
des phrases 21,5 → 14,3 mots.

Les CRs synthétiques ne sont pas prétraités.

## Métriques

Calculées avec le même code que `evaluate_cag.py`, le script d'évaluation des
CRs synthétiques (vérifié : valeurs reproduites à l'identique sur 252 CRs
synthétiques) :

- **par CR** (3 687 CRs) : n_mots, MSTTR-100, Flesch RE et Gunning Fog
  (`textstat`, langue `fr`), perplexité (`asi/gpt-fr-cased-base`, 1 024 tokens) ;
- **par corpus** : Self-BLEU-4, Distinct-1/2, cosinus TF-IDF moyen, sur des
  tirages de **700 CRs** (taille des groupes synthétiques, car ces métriques
  dépendent de N), selon deux schémas :
  - *indépendants* : 5 tirages sans remise avec les graines 42 à 46 ; deux
    tirages peuvent partager des CRs (≈ 134 en commun ; 2 399 CRs distincts
    utilisés) ;
  - *disjoints* : les CRs sont mélangés une fois (graine 42, la même pour tous
    les blocs) puis découpés en 5 blocs de 700 (`fold` 1 à 5) sans CR commun
    (3 500 CRs utilisés sur 3 687).

  Une même graine sélectionne les mêmes CRs dans les versions brute et ponctuée.

## Reproduire

```bash
pip install -r requirements.txt

python scripts/punctuate_reports.py --input_dir "2. Corps Extraits" --output_dir "3. CRs ponctues"

python scripts/evaluate_real_reports.py --input_dir "2. Corps Extraits" \
    --output results/eval_bruts_graine42.csv --device cuda --sample_n 700 --seed 42
python scripts/evaluate_real_reports.py --input_dir "3. CRs ponctues" \
    --output results/eval_ponctues_graine42.csv --device cuda --sample_n 700 --seed 42

python scripts/corpus_metrics_seeds.py --input_dirs "2. Corps Extraits" "3. CRs ponctues" \
    --seeds 42 43 44 45 46 --sample_n 700 --output results/corpus_independants_graines42-46.csv
python scripts/corpus_metrics_seeds.py --mode disjoint --input_dirs "2. Corps Extraits" "3. CRs ponctues" \
    --split_seed 42 --n_folds 5 --sample_n 700 --output results/corpus_disjoints_melange42_5blocs.csv
```

Sous Windows, définir `PYTHONIOENCODING=utf-8` si la sortie est redirigée.
