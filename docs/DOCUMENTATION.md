# Documentation complète du travail

Ce document explique **tout ce qui a été fait** pour évaluer les CRs réels de la
clinique, **pourquoi** chaque choix a été fait, et **comment le refaire**.
Il est écrit pour être relu plus tard, sans accès au serveur de la clinique.

> Aucune donnée de patient n'apparaît ici. Les exemples de texte sont
> fictifs (noms, dates) ; les chiffres sont des agrégats.

## Sommaire

1. [Objectif](#1-objectif)
2. [Les données](#2-les-données)
3. [Chronologie des décisions](#3-chronologie-des-décisions)
4. [Le prétraitement en détail](#4-le-prétraitement-en-détail)
5. [Les métriques](#5-les-métriques)
6. [L'échantillonnage des 700 CRs](#6-léchantillonnage-des-700-crs)
7. [Résultats](#7-résultats)
8. [Interprétation](#8-interprétation)
9. [Limites et points de vigilance](#9-limites-et-points-de-vigilance)
10. [Environnement technique](#10-environnement-technique)
11. [Reproduire](#11-reproduire)
12. [Fichiers de résultats : contenu des colonnes](#12-fichiers-de-résultats--contenu-des-colonnes)
13. [Dépôt Git et protection des données](#13-dépôt-git-et-protection-des-données)
14. [Questions fréquentes](#14-questions-fréquentes)

---

## 1. Objectif

Projet : génération automatique de comptes rendus (CRs) de radiothérapie par
LLM (pipeline CAG). Les CRs synthétiques ont été évalués avec des métriques
linguistiques (script `evaluate_cag.py`). Pour savoir si ces valeurs sont
« bonnes », il faut une **référence** : les mêmes métriques, calculées **de la
même façon**, sur les **vrais CRs de la clinique**.

Ce dépôt contient tout ce qui sert à produire cette référence.

---

## 2. Les données

### 2.1 CRs réels (non versionnés)

- **3 687 CRs** de radiothérapie (sein), dossier `2. Corps Extraits/` sur le
  serveur de la clinique, un fichier `.txt` par CR, encodage UTF-8.
- Les noms de fichiers contiennent des identifiants numériques, le nom du
  médecin et les dates de traitement → **données identifiantes**.
- Chaque CR contient : un **en-tête administratif** (destinataires, « Ref : »,
  identité de la patiente, date de naissance), le **corps clinique**
  (Indication, Préparation au traitement, Technique/Réalisation, Volumes,
  Tolérance, Thérapeutique ultérieure…) et un **pied de page** (signature,
  numéro de téléphone de la coordination, mention de messagerie sécurisée).

**Problème constaté à l'extraction** : la mise en forme d'origine (sauts de
ligne, listes) a été perdue.

- **100 %** des CRs tiennent sur **une seule ligne**.
- Beaucoup de phrases et de sections s'enchaînent **sans ponctuation**.
  Exemple (fictif) :
  `… né(e) MARTIN le01/01/1950 Indication: radiothérapie adjuvante …`
- Longueur des « phrases » (texte entre deux `. ! ?`) : médiane 21,5 mots,
  mais jusqu'à **255 mots** pour certains CRs.
- Quelques particularités de caractères : tirets invisibles (U+00AD, « trait
  d'union conditionnel ») dans 1 245 CRs (`sus­claviculaire`), apostrophes
  typographiques (`’`), dates collées au mot précédent (`le01/01/1950`,
  1 410 occurrences).
- Les accents sont corrects dans les fichiers (s'ils apparaissent comme `�`
  dans un terminal Windows, c'est un problème d'affichage de la console, pas
  des fichiers).

### 2.2 CRs synthétiques (hors de ce dépôt)

- Générés au format JSONL (une ligne JSON par CR), champs : `idx`, `dataset`
  (metabric / seer), `prompt_type` (0shot / 5shot), `model`, `temperature`,
  `top_p`, `repetition_penalty`, `context`, `cr_text`, `duration_s`.
- Modèle évalué : `Henrychur/MMed-Llama-3-8B-EnIns`, **12 groupes**
  (2 datasets × 2 types de prompt × 3 configurations d'échantillonnage) de
  **700 CRs** chacun.
- Le texte évalué est `cr_text`, **tel quel**, sans aucun prétraitement.
- Les CRs synthétiques contiennent généralement de la ponctuation et quelques
  sauts de ligne (médiane : 5 par CR). Certains sont des **générations ratées**
  (phrase répétée en boucle, morceaux de dialogue « assistant… »), ce qui
  explique des valeurs extrêmes dans leurs résultats (Flesch à −654, etc.).
- Ces fichiers sont conservés ailleurs (pas sur le serveur de la clinique).
  Les copies déposées un temps sur le serveur étaient **tronquées**
  (1 310 720 octets exactement, coupées au milieu d'une ligne) : ne pas s'en
  servir.

---

## 3. Chronologie des décisions

Cette section raconte **dans l'ordre** ce qui a été essayé, ce qui a été
gardé, et pourquoi. Elle sert à justifier la méthode.

### Étape 1 — Environnement

- Dépendances vérifiées ; `sacrebleu` manquait → installé.
- Le serveur a un GPU (NVIDIA RTX 4000 Ada, 20 Go) mais PyTorch était la
  version **CPU** → remplacé par `torch 2.6.0+cu124` (même version, build CUDA).
- Le modèle de perplexité `asi/gpt-fr-cased-base` a été téléchargé depuis
  HuggingFace et testé.

### Étape 2 — Premier nettoyage : **rejeté**

Un premier prétraitement retirait l'en-tête et le pied de page, normalisait
des caractères (`50Gy` → `50 Gy`, apostrophes…) et mettait une phrase par
ligne.

**Rejeté** : règle fixée pour tout le projet → **le texte des CRs réels doit
rester exactement tel qu'il est écrit**. On ne supprime rien (ni en-tête, ni
signature) et on ne modifie aucun caractère.

### Étape 3 — Sauts de ligne uniquement : **abandonné**

Deuxième version : seulement remplacer certains espaces par des sauts de ligne
(≈ 27 lignes par CR), texte vérifié identique.

**Abandonné** : les sauts de ligne n'ont presque aucun effet sur les
métriques (`textstat` ne les compte pas comme fins de phrase), et ce format
(une phrase par ligne) ne correspond pas non plus à celui des synthétiques
(≈ 5 sauts de ligne par CR).

### Étape 4 — Alignement sur le script synthétique

Le script `evaluate_cag.py` a été comparé ligne à ligne au script des CRs
réels. **Différence majeure trouvée** : Flesch et Gunning Fog étaient calculés
avec la bibliothèque `textstat` côté synthétique, mais avec une formule
« maison » côté réel → valeurs non comparables. Le script réel a été aligné
(voir §5.3).

**Vérification** : les métriques ont été recalculées sur 252 CRs synthétiques
avec le code aligné, et comparées aux valeurs d'origine du fichier
`medllama8b_eval_per_cr.csv` :

| Métrique | Résultat |
|---|---|
| n_mots, MSTTR-100, Flesch, Gunning Fog | identiques sur les 252 CRs |
| Perplexité | écart maximal 0,0005 (arrondi GPU) |

→ Le calcul est **exactement** celui utilisé pour les synthétiques.

### Étape 5 — Ponctuation minimale : **retenue**

Constat : une phrase de 100 mots sans point « n'a pas de sens » et fausse la
lisibilité et la perplexité. Décision : **ajouter** la ponctuation manquante
(et un espace dans les dates collées), **sans rien retirer ni modifier**.
Détails au §4.

### Étape 6 — Évaluer les deux versions

Pour mesurer l'effet de cette ponctuation ajoutée, les métriques ont été
calculées sur **les deux versions** : CRs **bruts** (`2. Corps Extraits`) et
CRs **ponctués** (`3. CRs ponctues`), sur exactement les mêmes CRs.

### Étape 7 — Robustesse de l'échantillonnage

Les métriques de corpus sont calculées sur 700 CRs (§6). Pour vérifier que le
résultat ne dépend pas des CRs tirés, le tirage a été répété **5 fois**, de
deux façons (tirages indépendants, puis blocs disjoints). Résultat : les deux
façons donnent la même chose.

### Étape 8 — Dépôt GitHub privé

Code, documentation et résultats agrégés poussés sur un dépôt privé ;
**aucune donnée de la clinique** (§13).

---

## 4. Le prétraitement en détail

Script : `scripts/punctuate_reports.py`.
Entrée : `2. Corps Extraits/` → sortie : `3. CRs ponctues/` (mêmes noms de
fichiers).

### 4.1 Principe

**On n'ajoute que deux choses, on ne retire et ne modifie rien.**

1. Un **point** à la fin d'une phrase non ponctuée, juste avant le début de la
   phrase suivante. Le point est collé au dernier mot ; l'espace d'origine est
   conservé.
2. Un **espace** entre `le`, `du` ou `au` et une date collée :
   `le01/01/1950` → `le 01/01/1950`.

Pas de saut de ligne ajouté : chaque CR reste sur une ligne.

### 4.2 Où ajoute-t-on un point ?

Un point est ajouté avant le début d'une nouvelle phrase, détectée de trois
façons :

1. **Intitulé de section** : `Indication`, `Préparation au traitement` /
   `à la radiothérapie`, `Réalisation du traitement`, `Modalités de traitement`,
   `Technique`, `Dosimétrie`, `Volumes (cibles)`, `Organes à risque`,
   `Tolérance (au traitement / à l'irradiation)`,
   `Thérapeutique ultérieure et surveillance`, `Suite de la prise en charge`,
   `Décision en fin de traitement`, `En fin (de) traitement`, `Prévision(s)`,
   `Décision RCP`, `Histologie`, `Traitement post op`, `Antécédents`,
   `Histoire de la maladie` ;
   et blocs administratifs : `Adressé à`, `Copie à`, `Ref :` / `Réf. :`,
   `COMPTE RENDU` / `Compte-rendu`, `Signé électroniquement`,
   `Vous pouvez joindre…`, `Vous êtes médecin…`, `Pour recevoir les courriers…`.
   (`CONCERNANT` n'est **pas** un intitulé : « COMPTE RENDU DE RADIOTHERAPIE
   CONCERNANT Mme X » est une seule phrase.)
2. **Élément « Libellé : valeur »** : `Sein gauche :`, `OAR cœur :`,
   `Poumon droit:`, `CTV sein droit :`… Le libellé commence au dernier mot en
   majuscule (ou acronyme OAR, CT, CTV, PTV, GTV) parmi les 4 mots avant les
   deux-points. Unités (`Gy`, `MV`…) et noms d'appareils (`Toshiba`, `Varian`…)
   ne peuvent pas débuter un libellé ; `(CTDI :` non plus.
3. **Mot qui commence habituellement une phrase** : `Pas`, `Poursuite`,
   `Double`, `Dosimétrie`, `Mise`, `La`, `Le`, `Il`, `Elle`, `Je`… (≈ 200 mots).
   La liste a été **construite à partir du corpus lui-même** : mots qui suivent
   un point dans au moins 15 CRs et qui existent aussi en minuscule ailleurs
   (pour exclure les noms propres), puis relue à la main (`Docteur` retiré).

**Exceptions — pas de point si** :

- le caractère précédent est déjà `. ! ? : ; , …` ;
- le mot précédent est un article ou une préposition (`le Docteur`,
  `du Sein`, `par Radiothérapie`…) ;
- la phrase fermée ne contiendrait qu'**un seul mot** ou **un intitulé seul**
  (`Préparation au traitement Scanner dosimétrique…` reste intact) ;
- c'est un **sous-intitulé après deux-points** (`Volumes : Cibles CTV…`).

Les phrases qui ont déjà leur point ne sont pas touchées.

### 4.3 Exemples (fictifs)

| Avant | Après |
|---|---|
| `… né(e) MARTIN le01/01/1950 Indication: radiothérapie …` | `… né(e) MARTIN le 01/01/1950. Indication: radiothérapie …` |
| `… avec et sans filtre Pas de boost du fait …` | `… avec et sans filtre. Pas de boost du fait …` |
| `… dose moyenne 51 Gy OAR cœur : dose moyenne 2 Gy …` | `… dose moyenne 51 Gy. OAR cœur : dose moyenne 2 Gy …` |
| `… le 01/02/2020 COMPTE RENDU DE RADIOTHERAPIE …` | `… le 01/02/2020. COMPTE RENDU DE RADIOTHERAPIE …` |
| `… RH +, Cerb2 - Décision RCP : …` | `… RH +, Cerb2 -. Décision RCP : …` |

### 4.4 Ce qui n'est **pas** touché

- Les stades TNM qui ressemblent à des dates (`pN0/2`, `ypN1/3` : ganglions
  atteints / examinés) : 774 occurrences intactes.
- Les autres collages (`50Gy`, `3semaines`), les artefacts d'extraction
  (`CT V` pour CTV, `p T1 b`), les tirets invisibles, les apostrophes.
- L'en-tête, la signature, le pied de page.

### 4.5 Contrôle automatique

Pour **chaque** CR, le script retire les caractères qu'il a ajoutés et vérifie
qu'on retrouve **exactement** le texte d'origine. Résultat : 0 CR modifié
autrement, sur 3 687.

Contrôle manuel : 40 points ajoutés tirés au hasard dans tout le corpus ont été
relus → tous corrects (un cas après `…` a été corrigé en ajoutant `…` aux
ponctuations finales).

### 4.6 Bilan chiffré

| | Valeur |
|---|---|
| Points ajoutés | 33 096 (≈ 9 par CR) |
| Espaces ajoutés (dates) | 1 410 |
| Mots par phrase, médiane | 21,5 → 14,3 |
| Mots par phrase, 90ᵉ centile | 40,7 → 17,6 |
| Mots par phrase, maximum (moyenne d'un CR) | 255 → 30 |

### 4.7 Ce qu'il faut dire dans un article

> Les CRs réels ayant perdu leur mise en forme à l'extraction, un point a été
> ajouté à la fin des phrases non ponctuées (avant un intitulé de section, un
> élément « libellé : valeur » ou un mot débutant habituellement une phrase),
> ainsi qu'un espace dans les dates collées. Aucun caractère n'a été supprimé
> ni modifié (vérifié automatiquement). Les CRs synthétiques n'ont pas été
> prétraités. Les résultats sont rapportés pour les deux versions des CRs réels.

---

## 5. Les métriques

### 5.1 Vue d'ensemble

| Métrique | Niveau | Ce qu'elle mesure | Sens |
|---|---|---|---|
| `n_mots` | par CR | longueur | — |
| `msttr_100` | par CR | richesse lexicale | ↑ = vocabulaire plus varié |
| `flesch_re_fr` | par CR | lisibilité | ↑ = plus facile à lire |
| `gunning_fog_fr` | par CR | complexité | ↑ = plus complexe |
| `perplexite` | par CR | fluidité (naturel du texte pour un modèle de langue) | ↓ = plus naturel |
| `self_bleu_4` | corpus | similarité entre CRs | ↑ = CRs plus semblables (moins divers) |
| `distinct_1`, `distinct_2` | corpus | proportion de mots / paires de mots uniques | ↑ = plus divers |
| `mean_pairwise_cosine` | corpus | similarité TF-IDF moyenne entre paires de CRs | ↑ = moins divers |

### 5.2 Mots

Un « mot » = une suite de lettres (accents compris), éventuellement avec des
traits d'union, mise en minuscules (expression régulière
`[A-Za-zÀ-ÖØ-öø-ÿ]+(?:-[A-Za-zÀ-ÖØ-öø-ÿ]+)*`). Les nombres et la
ponctuation ne sont pas des mots. **Conséquence** : ajouter des points ou des
espaces ne change ni `n_mots`, ni MSTTR, ni Distinct.

### 5.3 Formules

- **MSTTR-100** : le texte est découpé en fenêtres de 100 mots consécutives ;
  pour chaque fenêtre, TTR = mots différents / 100 ; MSTTR = moyenne des TTR.
  La dernière fenêtre incomplète est ignorée ; un CR de moins de 100 mots a son
  TTR simple.
- **Flesch (français, Kandel & Moles)**, via `textstat` 0.7.13, langue `fr` :
  `207 − 1,015 × (mots / phrase) − 73,6 × (syllabes / mot)`.
  Syllabes comptées avec le dictionnaire de césure `pyphen` (français).
- **Gunning Fog**, via `textstat` :
  `0,4 × (mots / phrase + 100 × mots difficiles / mots)`,
  « mot difficile » = 3 syllabes ou plus (textstat n'a pas de liste de mots
  faciles en français).
- **Comptage des phrases dans textstat** : segments terminés par `. ! ?` ;
  les segments de 2 mots ou moins ne comptent pas. Les sauts de ligne ne
  comptent pas. C'est pourquoi la ponctuation ajoutée change Flesch et Fog.
- **Perplexité** : modèle `asi/gpt-fr-cased-base` (GPT-2 français). Le CR
  entier est passé au modèle (tronqué à **1 024 tokens**) ; perplexité =
  exp(perte moyenne par token).
- **Self-BLEU-4** (`sacrebleu`, BLEU-4 phrase avec `effective_order`) : chaque
  CR est comparé à tous les autres CRs du corpus comme références ; on fait la
  moyenne. Échelle 0–100.
- **Distinct-n** : nombre de n-grammes différents / nombre total de n-grammes,
  sur tout le corpus.
- **Cosinus TF-IDF moyen** : vecteurs TF-IDF (`scikit-learn`, paramètres par
  défaut), similarité cosinus pour toutes les paires de CRs, moyenne.

### 5.4 Ce qui est identique au script synthétique

`scripts/evaluate_real_reports.py` reprend **le même code** que
`evaluate_cag.py` pour : tokenisation des mots, MSTTR, Flesch, Fog (textstat),
perplexité (même modèle, même troncature), Self-BLEU-4, Distinct, cosinus.
Le texte est passé tel quel (pas de `.strip()`), sans arrondi.

Différences **voulues** :

- pas de BERTScore (il sert à détecter une copie des exemples du prompt
  5-shot : sans objet pour des CRs réels) ;
- pas de groupes (un seul corpus réel) ;
- sous-échantillonnage à 700 pour les métriques de corpus (§6).

### 5.5 Métriques par CR vs métriques de corpus

- **Par CR** : calculées sur **les 3 687 CRs**, puis moyenne ± écart-type.
  Elles ne dépendent pas de la taille du corpus.
- **De corpus** : elles **dépendent de la taille N du corpus**, il faut donc le
  même N que les synthétiques (700) :
  - Distinct **baisse** quand N augmente (le vocabulaire nouveau se raréfie,
    loi de Heaps) ;
  - Self-BLEU **augmente** quand N augmente (plus de références → plus de
    chances de retrouver chaque n-gramme) ;
  - le cosinus moyen dépend peu de N (moyenne de paires), un peu via l'IDF.

  Calculer sur 3 687 CRs n'aurait pas été « faux », mais **non comparable** aux
  groupes de 700, et Self-BLEU coûte N² (≈ 28 fois plus long).

---

## 6. L'échantillonnage des 700 CRs

### 6.1 Mécanisme commun

1. Les fichiers `.txt` sont triés **par ordre alphabétique de nom** →
   chaque CR a un numéro (indice) de 0 à 3 686. Tri de texte, pas
   chronologique ; seul compte le fait qu'il soit toujours le même.
2. `random.seed(graine)` met le générateur aléatoire de Python dans un état
   précis : **même graine = même tirage**, sur n'importe quelle machine.
3. Les CRs sont choisis au hasard (vraiment mélangés, pas des indices
   consécutifs : vérifié, répartition uniforme sur toute la liste).

Les dossiers brut et ponctué ont **exactement les mêmes noms de fichiers**
→ une graine donnée choisit **les mêmes CRs dans les deux versions**. La
comparaison brut / ponctué porte donc sur les mêmes textes.

⚠ Ajouter ou retirer un fichier du dossier décale les indices : pour
reproduire les résultats, le dossier doit rester identique.

### 6.2 Tirages indépendants (graines 42, 43, 44, 45, 46)

- **5 graines différentes**, un tirage de 700 CRs par graine
  (`random.sample`, **sans remise** : pas de doublon dans un même corpus).
- Les tirages sont indépendants → **deux corpus peuvent partager des CRs**
  (≈ 134 en commun en moyenne, ce qu'attend le hasard : 700 × 700 / 3 687).
- Couverture : **2 399 CRs distincts** (65 %) ; 1 288 jamais tirés.

| Nombre de graines ayant tiré le CR | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Nombre de CRs | 1 288 | 1 517 | 687 | 172 | 22 | 1 |

- La graine 42 est celle de l'évaluation principale
  (`evaluate_real_reports.py --seed 42`) : mêmes valeurs.

### 6.3 Blocs disjoints (mélange graine 42, blocs 1 à 5)

- **Une seule graine (42)**, utilisée **une seule fois** pour mélanger toute la
  liste des 3 687 CRs (comme battre un jeu de cartes).
- La liste mélangée est coupée en **5 blocs consécutifs de 700** (comme
  distribuer 5 paquets). Ce qui distingue les 5 corpus est donc le **numéro de
  bloc** (`fold`), pas la graine (`split_seed` vaut toujours 42).

| Bloc (`fold`) | Position dans la liste mélangée |
|---|---|
| 1 | 1 à 700 |
| 2 | 701 à 1 400 |
| 3 | 1 401 à 2 100 |
| 4 | 2 101 à 2 800 |
| 5 | 2 801 à 3 500 |

- **Aucun CR commun entre deux blocs** (vérifié) ; 3 500 CRs utilisés, les
  187 derniers de la liste mélangée sont laissés de côté.
- Particularité : le **bloc 1 est identique au tirage indépendant de la
  graine 42** (même graine + même algorithme de mélange de Python : les 700
  premiers du mélange complet sont exactement les 700 que tire
  `random.sample(…, 700)`). Ce n'est pas un problème ; cela sert de contrôle.

### 6.4 Pourquoi deux schémas ?

Les tirages indépendants sont la méthode standard (sous-échantillonnage). Leur
recouvrement rend les 5 valeurs un peu dépendantes, ce qui peut sous-estimer
l'écart-type. Les blocs disjoints répondent à l'objection « certains CRs
comptent plusieurs fois ». **Résultat : les deux donnent la même chose** (§7.3),
on peut présenter l'un ou l'autre.

### 6.5 Formulation possible pour les encadrants / un article

> Les métriques de diversité dépendant de la taille du corpus, elles ont été
> calculées sur des sous-échantillons de 700 CRs réels (taille des groupes
> synthétiques). Nous rapportons la moyenne ± écart-type sur 5
> sous-échantillons : soit 5 tirages aléatoires indépendants sans remise
> (graines 42 à 46), soit 5 blocs disjoints obtenus en découpant une
> permutation aléatoire du corpus (aucun CR commun). Les deux schémas donnent
> des valeurs identiques à moins d'un écart-type. Les métriques par CR sont
> calculées sur les 3 687 CRs.

---

## 7. Résultats

### 7.1 Métriques par CR (3 687 CRs) — fichiers `eval_*_graine42_summary.csv`

| Métrique | CRs bruts | CRs ponctués | Écart moyen (médian) | CRs concernés |
|---|---|---|---|---|
| n_mots | 367,8 ± 143,8 | 367,8 ± 143,8 | 0 | aucun |
| MSTTR-100 | 0,728 ± 0,043 | 0,728 ± 0,043 | 0 | aucun |
| Flesch RE | 59,5 ± 10,4 | 67,7 ± 5,3 | +8,2 (+4,8) | 99,9 % en hausse |
| Gunning Fog | 16,6 ± 3,8 | 13,3 ± 1,3 | −3,2 (−1,9) | 99,9 % en baisse |
| Perplexité | 44,8 ± 16,6 | 41,7 ± 14,0 | −3,1 (−1,9), soit −7 % | 99,4 % en baisse |

Test de Wilcoxon apparié (même CR, brut vs ponctué) : p ≈ 0 pour Flesch, Fog et
perplexité.

### 7.2 Métriques de corpus, graine 42 seule (N = 700)

| Métrique | Bruts | Ponctués |
|---|---|---|
| Self-BLEU-4 | 78,91 | 79,78 |
| Distinct-1 | 0,0299 | 0,0299 |
| Distinct-2 | 0,1735 | 0,1735 |
| Cosinus TF-IDF | 0,2840 | 0,2862 |

### 7.3 Métriques de corpus, moyenne ± écart-type sur 5 sous-échantillons (N = 700)

| Métrique | Bruts — indépendants | Bruts — disjoints | Ponctués — indépendants | Ponctués — disjoints |
|---|---|---|---|---|
| Self-BLEU-4 | 78,59 ± 0,26 | 78,61 ± 0,28 | 79,47 ± 0,24 | 79,50 ± 0,28 |
| Distinct-1 | 0,0306 ± 0,0009 | 0,0305 ± 0,0007 | 0,0306 ± 0,0009 | 0,0305 ± 0,0007 |
| Distinct-2 | 0,1762 ± 0,0029 | 0,1756 ± 0,0017 | 0,1762 ± 0,0029 | 0,1756 ± 0,0017 |
| Cosinus TF-IDF | 0,2814 ± 0,0025 | 0,2803 ± 0,0033 | 0,2835 ± 0,0025 | 0,2825 ± 0,0033 |

Écart Self-BLEU ponctués − bruts, sous-échantillon par sous-échantillon :
+0,86 à +0,92 (indépendants), +0,87 à +0,91 (disjoints).

### 7.4 Rappel : CRs synthétiques (MMed-Llama-3-8B, 700 CRs par groupe)

Valeurs copiées de `medllama8b_eval_summary.csv` (moyennes ; T = température,
p = top_p, rp = repetition_penalty).

| Groupe | n_mots | MSTTR | Flesch | Fog | Perplexité | Self-BLEU-4 | Distinct-1 | Distinct-2 | Cosinus |
|---|---|---|---|---|---|---|---|---|---|
| metabric 0shot T0.3 p0.85 rp1.1 | 417,3 | 0,676 | 51,3 | 17,4 | 23,6 | 74,6 | 0,0250 | 0,119 | 0,197 |
| metabric 0shot T0.5 p0.9 rp1.1 | 417,5 | 0,695 | 29,9 | 19,3 | 30,9 | 67,6 | 0,0332 | 0,174 | 0,191 |
| metabric 0shot T0.8 p0.95 rp1.15 | 475,2 | 0,777 | 17,9 | 20,8 | 46,4 | 47,5 | 0,0678 | 0,393 | 0,185 |
| metabric 5shot T0.3 p0.85 rp1.1 | 573,4 | 0,567 | 31,6 | 20,8 | 53,4 | 75,5 | 0,0180 | 0,089 | 0,147 |
| metabric 5shot T0.5 p0.9 rp1.1 | 544,6 | 0,603 | −653,9 | 23,2 | 61,5 | 64,7 | 0,0236 | 0,123 | 0,125 |
| metabric 5shot T0.8 p0.95 rp1.15 | 467,3 | 0,769 | −63,5 | 21,0 | 80,1 | 43,1 | 0,0774 | 0,419 | 0,118 |
| seer 0shot T0.3 p0.85 rp1.1 | 436,2 | 0,686 | 43,9 | 17,7 | 24,9 | 76,2 | 0,0240 | 0,123 | 0,211 |
| seer 0shot T0.5 p0.9 rp1.1 | 446,3 | 0,686 | 40,1 | 17,8 | 25,9 | 68,0 | 0,0318 | 0,168 | 0,188 |
| seer 0shot T0.8 p0.95 rp1.15 | 474,7 | 0,765 | 39,5 | 20,7 | 45,0 | 48,9 | 0,0672 | 0,381 | 0,194 |
| seer 5shot T0.3 p0.85 rp1.1 | 574,5 | 0,594 | 39,3 | 16,7 | 29,4 | 75,7 | 0,0195 | 0,093 | 0,164 |
| seer 5shot T0.5 p0.9 rp1.1 | 570,0 | 0,624 | −216,9 | 18,1 | 32,6 | 68,8 | 0,0228 | 0,120 | 0,156 |
| seer 5shot T0.8 p0.95 rp1.15 | 535,8 | 0,762 | −137,9 | 19,9 | 62,5 | 44,7 | 0,0663 | 0,384 | 0,154 |
| **CRs réels, bruts** | 367,8 | 0,728 | 59,5 | 16,6 | 44,8 | 78,6 | 0,0306 | 0,176 | 0,281 |
| **CRs réels, ponctués** | 367,8 | 0,728 | 67,7 | 13,3 | 41,7 | 79,5 | 0,0306 | 0,176 | 0,284 |

(Réels : métriques de corpus = moyenne des 5 tirages indépendants.)

---

## 8. Interprétation

### 8.1 Effet de la ponctuation ajoutée

- **Aucun effet** sur n_mots, MSTTR, Distinct : attendu, les mots ne changent pas
  (et cela confirme qu'aucun mot n'a été modifié).
- **Lisibilité** : effet fort et systématique. Les phrases étant plus courtes,
  Flesch +8 points et Fog −3. L'écart-type est divisé par deux : les quelques CRs
  sans ponctuation (« phrases » de 100 à 255 mots) donnaient des valeurs
  extrêmes.
- **Perplexité** : baisse modérée (−7 %) mais quasi systématique (99 % des CRs) :
  le modèle de langue trouve le texte plus naturel quand les phrases sont
  délimitées.
- **Self-BLEU / cosinus** : très légère hausse (+0,9 / +0,002), systématique.
  Les points ajoutés créent des n-grammes répétés d'un CR à l'autre
  (ex. `2018 . COMPTE RENDU`). Effet négligeable.

**Conclusion** : le choix brut / ponctué compte surtout pour la **lisibilité**.
Pour la perplexité et la diversité, la conclusion de la comparaison avec les
synthétiques est la même quelle que soit la version.

### 8.2 Stabilité de l'échantillonnage

La variabilité entre sous-échantillons est très faible (Self-BLEU ± 0,26 ;
cosinus ± 0,003) et les deux schémas (indépendants / disjoints) concordent :
**le résultat ne dépend pas des CRs tirés.**

### 8.3 Réels vs synthétiques — pistes de lecture

- Les CRs réels sont **plus courts** (368 mots contre 417–575).
- Les CRs réels sont **moins divers** que tous les groupes synthétiques
  (Self-BLEU 78,6 contre 43–76 ; cosinus 0,28 contre 0,12–0,21). Explication
  probable : CRs très standardisés (même plan, phrases de dosimétrie
  récurrentes) **et** blocs administratifs identiques conservés dans le texte
  (en-tête, « Vous pouvez joindre l'équipe… », mention de messagerie). Les
  synthétiques n'ont pas forcément ces blocs.
- Les CRs réels sont **plus lisibles** selon Flesch que la plupart des groupes
  synthétiques ; les moyennes synthétiques sont toutefois tirées vers le bas par
  des générations ratées (valeurs à −654).
- La perplexité des réels (42–45) est dans la fourchette des synthétiques
  (24–80) : les configurations à basse température sont **plus** « fluides »
  que les vrais CRs, celles à haute température moins.

---

## 9. Limites et points de vigilance

1. **Ponctuation heuristique** : les règles couvrent la grande majorité des cas
   (contrôle de 40 exemples : tous corrects) mais certaines frontières de phrase
   ne sont pas détectées, et quelques points peuvent être mal placés.
2. **Asymétrie de prétraitement** : seuls les CRs réels sont ponctués (les
   synthétiques le sont déjà, puisque le modèle écrit la ponctuation). D'où la
   présentation des deux versions.
3. **Blocs administratifs conservés** dans les CRs réels : ils gonflent
   Self-BLEU et cosinus, et pèsent sur la perplexité. Choix assumé (règle :
   texte intact) ; à mentionner.
4. **Gunning Fog** est une formule conçue pour l'anglais (seuil de 3 syllabes,
   pas de liste de mots faciles en français) : à interpréter avec prudence.
   Flesch peut devenir très négatif sur des textes dégénérés.
5. **Troncature à 1 024 tokens** pour la perplexité : 302 CRs réels bruts
   (8,2 %) et 314 ponctués (8,5 %) dépassent cette limite ; seule leur partie
   initiale est évaluée (comme pour les synthétiques). Médiane : ≈ 680 tokens.
6. **Générations ratées** dans les synthétiques : elles faussent leurs moyennes
   (Flesch notamment) ; une comparaison sur les CRs synthétiques valides serait
   plus juste.
7. **Reproductibilité** : dépend des versions (`textstat` 0.7.13 en
   particulier, dont les formules changent selon les versions) et du contenu
   exact du dossier de CRs (§6.1). La perplexité sur GPU peut varier à la
   4ᵉ décimale.

---

## 10. Environnement technique

### 10.1 Machine utilisée

- Serveur Windows Server 2022, Python 3.9.13 (Anaconda).
- GPU NVIDIA RTX 4000 Ada (20 Go), pilote compatible CUDA 12.4.
- 12 cœurs / 24 threads, 64 Go de RAM.

### 10.2 Installation

```bash
pip install -r requirements.txt
# GPU : version CUDA de PyTorch (la version pip par défaut sous Windows est CPU)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Le modèle `asi/gpt-fr-cased-base` est téléchargé automatiquement depuis
HuggingFace au premier lancement (≈ 500 Mo, mis en cache dans
`~/.cache/huggingface`). Sans internet, il faut copier ce cache.

Avertissement sans conséquence sous Windows : « cache-system uses symlinks… »
(désactivable avec `HF_HUB_DISABLE_SYMLINKS_WARNING=1`).

### 10.3 Pièges rencontrés

- **Accents / symboles** : les scripts affichent `⚠` et `→`. Si la sortie est
  redirigée vers un fichier sous Windows, définir `PYTHONIOENCODING=utf-8`,
  sinon erreur d'encodage.
- **Sortie qui n'apparaît pas dans le journal** : Python met la sortie en
  mémoire tampon quand elle est redirigée → lancer avec `python -u`.
- **`git` non reconnu dans VS Code** après installation de Git : fermer et
  rouvrir complètement VS Code (le terminal garde l'ancien PATH), ou utiliser
  le chemin complet de `git.exe`.

### 10.4 Durées observées

| Étape | Durée |
|---|---|
| Prétraitement (3 687 CRs) | quelques secondes |
| Perplexité, 3 687 CRs sur GPU | ≈ 15–20 min (≈ 3,4 CRs/s, deux évaluations en parallèle) |
| Self-BLEU-4 sur 700 CRs (CPU, 1 cœur) | ≈ 12–15 min |
| 10 sous-échantillons en parallèle (10 cœurs) | ≈ 13 min |

---

## 11. Reproduire

Depuis la racine du dépôt, avec les CRs placés dans `2. Corps Extraits/` :

```bash
# 1. Prétraitement (crée "3. CRs ponctues/")
python scripts/punctuate_reports.py --input_dir "2. Corps Extraits" --output_dir "3. CRs ponctues"

# 2. Métriques par CR + corpus (graine 42), pour chaque version
python -u scripts/evaluate_real_reports.py --input_dir "2. Corps Extraits" \
    --output results/eval_bruts_graine42.csv --device cuda --sample_n 700 --seed 42
python -u scripts/evaluate_real_reports.py --input_dir "3. CRs ponctues" \
    --output results/eval_ponctues_graine42.csv --device cuda --sample_n 700 --seed 42

# 3. Métriques de corpus sur 5 tirages indépendants
python -u scripts/corpus_metrics_seeds.py --input_dirs "2. Corps Extraits" "3. CRs ponctues" \
    --seeds 42 43 44 45 46 --sample_n 700 --output results/corpus_independants_graines42-46.csv

# 4. Métriques de corpus sur 5 blocs disjoints
python -u scripts/corpus_metrics_seeds.py --mode disjoint --input_dirs "2. Corps Extraits" "3. CRs ponctues" \
    --split_seed 42 --n_folds 5 --sample_n 700 --output results/corpus_disjoints_melange42_5blocs.csv
```

Les étapes 2 peuvent tourner en parallèle (le GPU a assez de mémoire).
Chaque script écrit aussi un fichier `…_summary.csv` à côté de `--output`.

### Scripts

| Script | Rôle | Options principales |
|---|---|---|
| `punctuate_reports.py` | ajoute points et espaces, vérifie que rien d'autre ne change | `--input_dir`, `--output_dir` |
| `evaluate_real_reports.py` | métriques par CR (tous les CRs) + corpus (1 tirage) | `--input_dir`, `--output`, `--lm_model`, `--device`, `--sample_n`, `--seed` |
| `corpus_metrics_seeds.py` | métriques de corpus sur plusieurs sous-échantillons | `--mode independent/disjoint`, `--seeds`, `--split_seed`, `--n_folds`, `--sample_n`, `--workers` |

---

## 12. Fichiers de résultats : contenu des colonnes

Convention de nommage : `bruts` = CRs tels qu'extraits ; `ponctues` = après
prétraitement ; `_summary` = agrégat (moyenne ± écart-type).

| Fichier | Contenu |
|---|---|
| `eval_bruts_graine42_summary.csv`, `eval_ponctues_graine42_summary.csv` | lignes = métriques ; colonnes `mean`, `std`. Les 5 premières lignes : métriques par CR (sur 3 687). Les 4 dernières (`… (N_corpus=700)`) : métriques de corpus de la graine 42 (pas d'écart-type). |
| `corpus_independants_graines42-46.csv` | 1 ligne par (version, graine) : `input_dir`, `seed`, `n_corpus`, 4 métriques |
| `corpus_independants_graines42-46_summary.csv` | 1 ligne par version : `n_seeds`, `<métrique>_mean`, `<métrique>_std` |
| `corpus_disjoints_melange42_5blocs.csv` | 1 ligne par (version, bloc) : `input_dir`, `split_seed` (toujours 42), `fold` (1–5), `n_corpus`, 4 métriques |
| `corpus_disjoints_melange42_5blocs_summary.csv` | 1 ligne par version : `n_folds`, `<métrique>_mean`, `<métrique>_std` |

`input_dir` vaut `2. Corps Extraits` (bruts) ou `3. CRs ponctues` (ponctués).

Non versionnés (sur le serveur uniquement) : `eval_bruts_graine42.csv` et
`eval_ponctues_graine42.csv` (1 ligne par CR, avec le nom de fichier du CR).

---

## 13. Dépôt Git et protection des données

- Dépôt **privé** : `github.com/sarahbrb/eval-cr-real-radiotherapie`.
- Le `.gitignore` est une **liste blanche** : tout est ignoré par défaut, seuls
  les fichiers explicitement listés sont versionnés. Un nouveau fichier de
  données ne peut donc pas partir par erreur. En plus, tous les `.txt` (sauf
  `requirements.txt`), `.jsonl` et `.log` sont bloqués partout.
- **Jamais versionnés** : les dossiers de CRs, les résultats par CR (noms de
  fichiers identifiants), les journaux d'exécution (`logs/`, qui listent les
  noms de fichiers), les fichiers synthétiques (le JSONL peut contenir des
  passages copiés des CRs réels des prompts 5-shot).
- Les exemples dans le code et la documentation utilisent des dates et noms
  **fictifs**.

Pour ajouter un nouveau fichier au dépôt : l'ajouter à la liste blanche du
`.gitignore`, puis toujours vérifier avec `git status` avant de commiter.

```bash
git pull
git add .
git status            # vérifier : aucun fichier de CR
git commit -m "description"
git push
```

---

## 14. Questions fréquentes

**Pourquoi 700 CRs pour les métriques de corpus ?**
Parce que chaque groupe synthétique contient 700 CRs et que Self-BLEU,
Distinct et (un peu) le cosinus dépendent de la taille du corpus (§5.5).

**Pourquoi ne pas tout calculer sur les 3 687 CRs ?**
Les métriques par CR le sont. Pour celles de corpus, ce serait non comparable
aux groupes de 700 (Distinct plus bas, Self-BLEU plus haut artificiellement),
et Self-BLEU serait ≈ 28 fois plus long.

**Les 5 graines couvrent-elles tous les CRs ?**
Non : 2 399 CRs sur 3 687 (65 %). Ce n'est pas nécessaire : chaque tirage
estime la valeur pour « un corpus de 700 CRs réels au hasard ». Les blocs
disjoints couvrent 3 500 CRs et donnent les mêmes valeurs.

**Un même CR peut-il être dans deux corpus ?**
Tirages indépendants : oui (≈ 134 CRs communs entre deux graines), mais jamais
deux fois dans le même corpus. Blocs disjoints : non, jamais.

**Pourquoi la graine est-elle toujours 42 en mode disjoint ?**
Parce qu'il n'y a qu'un seul mélange ; ce qui change d'un corpus à l'autre est
le bloc (`fold`). Avec plusieurs graines, il y aurait plusieurs mélanges et des
CRs communs.

**Les CRs sont-ils pris à la suite dans la liste ?**
Non, ils sont vraiment mélangés (dans les deux modes). En mode disjoint, les
blocs sont consécutifs dans la liste **déjà mélangée**.

**Pourquoi le bloc 1 donne-t-il les mêmes valeurs que la graine 42 ?**
Même graine et même algorithme de mélange : les 700 premiers CRs du mélange
complet sont exactement ceux du tirage de 700 avec la graine 42 (§6.3).

**Pourquoi les CRs synthétiques ne sont-ils pas ponctués aussi ?**
Ils ont déjà leur ponctuation (écrite par le modèle) ; le problème de
ponctuation des CRs réels vient de leur extraction du dossier patient.

**Pourquoi utiliser textstat plutôt qu'une formule maison ?**
Parce que c'est ce qu'utilise le script des synthétiques : la comparaison
n'a de sens qu'avec exactement le même calcul.
