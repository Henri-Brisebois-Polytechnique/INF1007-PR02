# INF1007-PR02

## Description du projet
Ce projet vise à la prédiction de lettres et de chiffres faits en langage des signes américain (ASL) en temps réel grâce à un modèle d'apprentissage automatique.

La webcam capte la main, MediaPipe en extrait 21 points de repère (x, y, z), ces points sont normalisés puis envoyés à un classificateur scikit-learn qui prédit l'une des 36 classes (0-9, A-Z). L'interface Gradio affiche la prédiction, les deux autres possibilités les plus probables et un graphique des probabilités.

## Source et licence du jeu de données
Source du jeu de données:

Pranto, Md. Famidul Islam ; Islam, Md. Rifatul; Akbor, Md. Ali; Ghosh, Nabonita; Alam, Md. Rahatun; Chaki, Sudipto; Islam, Md Masudul (2025), “ASL-HG: American Sign Language Hand Gesture Image Dataset”, Mendeley Data, V1, doi: 10.17632/j4y5w2c8w9.1

Le jeu de données est disponible à ce lien:

https://data.mendeley.com/datasets/j4y5w2c8w9/1

Licence du jeu de données:

CC BY 4.0

Les images brutes ne sont pas incluses dans le dépôt (trop volumineuses). Seul le CSV des points de repère extraits (`donnees/asl_landmarks.csv`) est fourni. Pour relancer l'extraction (`src/extraction.py`) ou l'évaluation (`src/reussite.py`), télécharger le jeu de données et placer les images à la racine du projet sous `asl_processed/train/<classe>/` et `asl_processed/test/<classe>/`.

## Procédure d'installation

Installer les packages nécessaires:

```bash
pip install -r requirements.txt
```

## Procédure de lancement

À partir de la racine du projet (le dossier qui contient `modeles/` et `donnees/`), lancer:

```bash
python src/main.py
```

Ouvrir ensuite le lien local affiché dans le terminal (par défaut http://127.0.0.1:7860).

Autres scripts (toujours depuis la racine):

```bash
python src/extraction.py   # images -> donnees/asl_landmarks.csv (nécessite asl_processed/)
python src/modele.py       # entraîne et sauvegarde les trois modèles dans modeles/
python src/reussite.py     # évalue les modèles sur P9 et P10 et génère les graphiques reussite_*.png
```

## Structure du code

```
2026A-PR02/
├── README.md
├── DECISIONS.md                 # fiche de décisions
├── JOURNAL_IA.md                # journal d'utilisation de l'IA
├── requirements.txt
├── pytest.ini
├── .github/workflows/tests.yml  # lance pytest + coverage à chaque push
├── assets/asl.png               # alphabet ASL affiché dans l'interface
├── donnees/asl_landmarks.csv    # points de repère extraits (une ligne par image)
├── modeles/                     # modèles entraînés (.joblib) + métadonnées (.json)
│   ├── foret/
│   ├── hgboost/
│   └── regressionlogistique/
├── src/
│   ├── main.py                  # point d'entrée : lance l'interface
│   ├── config.py                # chemins et constantes
│   ├── exceptions.py            # exceptions personnalisées
│   ├── tools.py                 # détection MediaPipe, normalisation, chargement du modèle, décorateur
│   ├── extraction.py            # images -> CSV de points de repère
│   ├── donnees.py               # lecture du CSV, classe JeuDonnees
│   ├── modele.py                # entraînement et sauvegarde des modèles
│   ├── reussite.py              # évaluation des modèles sur le jeu de test
│   ├── interface.py             # interface graphique Gradio
│   └── hand_landmarker.task     # modèle MediaPipe de détection de main
└── tests/
    ├── test_donnees.py
    └── test_modele.py
```

## Précision des modèles

Métrique : taux de bonnes prédictions (exactitude) sur les images de test des personnes P9 et P10, jamais vues à l'entraînement (P1 à P8). Une main non détectée compte comme une erreur.

| Modèle | Exactitude |
| --- | --- |
| Forêt aléatoire (`RandomForestClassifier`) | 72 % |
| Histogram gradient boosting (`HistGradientBoostingClassifier`) | 69 % |
| Régression logistique (`LogisticRegression`) | 84 % |

Référence : avec 36 classes équilibrées, une prédiction au hasard obtiendrait environ 1/36 ≈ 2,8 %.

Le détail par classe est dans les graphiques `reussite_*.png`.

## Couverture de tests

Mesurée avec `pytest-cov` (basé sur `coverage`), depuis la racine du projet:

```bash
pytest --cov=src --cov-report=term-missing
```

Pour un rapport HTML détaillé (ligne par ligne), ouvrir `htmlcov/index.html` après:

```bash
pytest --cov=src --cov-report=html
```

Résultat : **16 %** (19 tests réussis).

| Fichier | Couverture |
| --- | --- |
| `config.py`, `donnees.py`, `exceptions.py` | 100 % |
| `modele.py`, `tools.py` | 0 % |
| `interface.py`, `extraction.py`, `reussite.py`, `main.py` | 0 % |

## Références à la documentation

- pandas; `read_csv` : https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
- pandas; `Series.isin` : https://pandas.pydata.org/docs/reference/api/pandas.Series.isin.html
- pandas; `DataFrame.drop` : https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
- NumPy; `array` : https://numpy.org/doc/stable/reference/generated/numpy.array.html
- NumPy; `round` : https://numpy.org/doc/stable/reference/generated/numpy.round.html
- NumPy; `pad` : https://numpy.org/doc/stable/reference/generated/numpy.pad.html
- NumPy; `ascontiguousarray` : https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html
- NumPy; `copy` : https://numpy.org/doc/stable/reference/generated/numpy.copy.html
- NumPy; `max` : https://numpy.org/doc/stable/reference/generated/numpy.max.html
- NumPy; `linalg.norm` : https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
- NumPy; `ndarray.ravel` : https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ravel.html
- NumPy; `allclose` : https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
- NumPy; `isfinite` : https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- scikit-learn; `RandomForestClassifier` : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- scikit-learn; `HistGradientBoostingClassifier` : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
- scikit-learn; `LogisticRegression` : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- scikit-learn; persistance des modèles (joblib) : https://scikit-learn.org/stable/model_persistence.html
- Gradio; `Blocks` : https://www.gradio.app/docs/gradio/blocks
- MediaPipe; Hand Landmarker (Python) : https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- pytest-cov : https://pytest-cov.readthedocs.io/en/latest/

## Outils du chapitre 7 utilisés

- **Décorateur** : `@chronometrer` (`src/tools.py`) affiche le temps d'exécution d'une fonction ; il chronomètre l'entraînement des modèles (`Modele.entrainer`), l'extraction et l'évaluation.
- **Générateur** : `iterer_images` (`src/extraction.py`, `src/reussite.py`) produit les couples (chemin, classe) un à la fois avec `yield`, sans charger la liste de milliers d'images en mémoire.
- **Lambda** : `sorted(probas, key=lambda p: p[1], reverse=True)` (`src/interface.py`) trie les classes par probabilité décroissante pour afficher les trois plus probables.
- **Modules** : le code est réparti en modules (`config`, `exceptions`, `tools`, `donnees`, `modele`, `interface`, ...) qui s'importent les uns les autres, avec `main.py` comme point d'entrée.
