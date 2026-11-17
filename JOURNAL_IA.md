
|Outil|Prompt|Réponse reçue|Vérification faite|Décision|
| --- | --- | --- | --- | --- |
| Claude Code | Dans extraction.py, MediaPipe ne détecte aucune main sur environ 8 % des images du dataset (souvent quand la main est collée au bord ou coupée par le cadre). Comment améliorer le taux de détection sans fausser les coordonnées ? | Voir réponse 1 | Testé l'extraction à nouveau, observé un rendement de 99 % (vs 92 %). Puisque le dataset d'images a déjà été traité, on peut être sûrs que chaque image contient une main; un pourcentage plus élevé est donc normal. | Gardé |
| Claude | I am a university student studying in computer engineering. I have attached the project I am currently working on, which aims to have us train a machine learning model in Python. We're doing our project on sign language detection. We've split our project in three parts: analysing the photos, training our model on the data, and producing the graphical interface. I'm currently working on the second part. I have at my disposition a CSV file containing all the data from the images we've analysed using pylance. Suggest some paths i can take to accomplish my task to train our first model off our CSV, including what modules I could use, what algorithm would fit our project best, and break down my part of the project in steps. I know nothing about machine learning, so explain as simply and completely as possible. | Voir réponse 2 | Lecture de la documentation dans sklearn, joblib et pandas. | Conserver la structure générale du plan, mais avec certaines modifications détaillées dans le prochain prompt |
| Claude | Consider I've only given you the training data. I already have test data saved for later. The images have also already been preprocessed by subtracting the wrist point from every other point, and dividing by its norm. I don't want to evaluate my model just yet. Give me a new step-by-step plan of this section, and suggest some resources I can consult to use for the training | Voir réponse 3 | Lecture de la documentation dans sklearn, joblib et pandas. Écouter des vidéos Youtube sur différents modèles disponibles. Demander une question follow-up à Claude. | Conserver le plan pour le modèle |
| Claude | What do you mean by scaling? If I've already normalized my data, is that still a problem? | Voir réponse 4 | Lecture de la réponse de Claude. Lecture de la documentation de sklearn. Recherches en ligne sur différents modèles disponibles. | Utiliser un modèle random forest classifier |
| Claude | I have decided to use the random forest algorithm. I have added a column to the CSV called "personne" which contains data labeled "p1" "p2"... "p10". I only want to train my model on half the entries (from p1 to p5). I'm currently on step 1, loading the file with Pandas. What steps should I take now properly load my file? | Voir réponse 5 | Lecture de la documentation de sklearn | Conserver la structure du code suggéré par Claude, avec modifications pour aider lisibilité et fonctionnalité désirée |
| Claude | Moving on to the next step, I want to train my model on my data. There are no empty lines, and the amount of lines are equally distributed amongst each other. | Voir réponse 6 | Lecture de la documentation de sklearn, functools, typing and joblib. | Changement de la structure du code suggéré par Claude. Décidé de ne pas ignorer le vecteur d'origine, puisqu'il représente quand même de l'information qui peut être pertinante. Il pourra être enlevé lors de l'entraînement d'un futur modèle. |
| Claude | Are the functiools, typing and joblib librairies necessary. i want to use the least amount of unknown libraries as possible. explain their use. 
I don't want to get rid of x0, x1 and x2, even if they are always 0. will this greatly impact the result? | Voir réponse 7 | Lecture de la réponse de Claude et validation avec documentation. | Enlever librairie functools. Remplacer librairie typing par collections.abc. Conservation de Joblib. |
| Claude | edit the code you suggested to me while removing functools and changing "from typing import Callable" to "`from collections.abc import Callable`". do not change anything about joblib. in addition, make the edits so that we no longer remove x0 x1 and x2 from the analysed dataset. | Voir réponse 8 | Lecture et révision du code présenté par Claude | Conserver la structure proposée par Claude, mais avec certaines modifications. Enlever l'appel d'erreurs (le cours n'avait pas encore été passé. ajout de l'appel d'erreurs plus tard). |
| Claude | I have updated the CSV file to the one I'm currently using. We removed the "main" column. Adjust the code to match this. Explain to me what GRAINE is for. Explain what from exceptions import DonneeInvalideError is used for, and whether or not i can remove it. I'm not writing tests for my code yet. | Voir réponse 9 | Lecture des réponses offertes par Claude pour mieux comprendre le code. Test du code dans VScode. | Conservation de la structure de code proposée par Claude, avec modifications pour augmenter la compréhension du code basée sur mes lectures. |
| Claude | Suggest something to go in "metrique." Additionally, add lines at the end to show how we're going to call the functions (and suggest a chemin modele and chemin meta). | Voir réponse 10 | Lecture et test du code généré par Claude | Conservation de la structure proposée par Claude, avec plusieurs modifications. Écriture de certain morceaux de code manuellement pour correspondre avec les objectifs du projet. |

# RÉPONSES COMPLÈTES

## Réponse 1
Enjeu : MediaPipe ne détecte aucune main sur environ 8 % des images du dataset. Ces images sont perdues pour l'entraînement.

Cause probable : dans ces images, la main est en gros plan, souvent coupée par le cadre ou collée au bord. Le HandLandmarker cherche d'abord une paume entière avec un détecteur de paume. Quand la paume touche le bord, ce détecteur échoue, et les 21 points de repère ne sont jamais calculés.

Solution : ajouter une marge autour de l'image avant la détection. La main se retrouve alors plus loin des bords et paraît un peu plus petite, ce qui correspond mieux aux images sur lesquelles le détecteur a été entraîné.

MARGE = 0.25  # fraction du plus grand côté ajoutée de chaque côté

def charger_image(chemin: str) -> mp.Image:
    """Charge une image et l'entoure d'une bordure (répétition des bords).

    :param chemin: chemin de l'image.
    :return: image MediaPipe.
    """
    pixels = mp.Image.create_from_file(chemin).numpy_view()[..., :3]
    p = int(max(pixels.shape[:2]) * MARGE)
    pixels = np.pad(pixels, ((p, p), (p, p), (0, 0)), mode="edge")
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(pixels))

Ensuite, dans la boucle, on appelle detecteur.detect(charger_image(chemin)) au lieu de mp.Image.create_from_file(chemin).

Choix techniques :
- mode="edge" plutôt que du noir : une bordure noire crée une arête nette qui peut tromper le détecteur. Répéter les pixels du bord prolonge le fond et le poignet de façon naturelle.
- Marge proportionnelle : la même valeur fonctionne quelle que soit la résolution des images.
- [..., :3] : enlève le canal alpha des PNG, car le format SRGB attend 3 canaux.
- ascontiguousarray : mp.Image refuse un tableau qui n'est pas contigu en découpage des canaux peuvent produire.

Pas d'impact sur les données : les coordonnées sont calculées par rapportnormaliser() centre les points sur le poignet et divise par une seulevaleur d'échelle, le décalage et le zoom disparaissent. La seule limite concerne les images non carrées : le rapport entre x et y change un peu. Ce n'est pas grave si toutes les
images ont le même format.

Comment choisir la valeur de la marge : il faut la mesurer plutôt que la traction sur un échantillon d'environ 500 images avec MARGE à 0, 0.1, 0.25et 0.5, puis on compare nb_ok / total. Trop peu de marge ne règle pas le problème. Trop de marge rend la main trop petite, et la détection recommence à baisser.

Si ça ne suffit pas, d'autres pistes :
- baisser min_hand_detection_confidence (par exemple à 0.3), en acceptant ;
- refaire une tentative avec l'image retournée horizontalement (np.fliplr) quand la première détection échoue ;
- redimensionner les images trop grandes ou trop petites avant de les pas

## Réponse 2

# Training your first model on the ASL landmarks CSV

## First, a quick read of your data

Your CSV has 28,681 rows and 65 columns:
- **`x0…z20`**: 21 hand points × 3 coordinates = 63 numbers. This looks like MediaPipe hand landmarks (you wrote "pylance", which is a VS Code tool, so you probably mean MediaPipe). Each row is one image of a hand.
- **`main`**: text, probably "Left"/"Right" hand. This is your **categorical** feature.
- **`classe`**: the letter or sign. This is your **target**, the thing the model must guess.

This fits the project criteria: 500 to 50,000 rows, at least 5 inputs, and one categorical plus several numeric variables. The type of problem is **classification**, because you predict a category (a letter), not a number.

## ML in plain words

Training a model is like showing flashcards to a student.
1. You give the model many examples with the answer attached (63 coordinates → letter "A").
2. It finds patterns, for example "when the fingertips are curled near the palm, it's usually an A".
3. You then test it on examples it has **never seen** to check that it really learned and didn't just memorize.

That is why you split the data into a **training set** (about 80%) and a **test set** (about 20%).

## Which algorithm?

| Algorithm | How it works | Pros | Cons |
|---|---|---|---|
| **Random Forest** (my recommendation) | Many small decision trees vote on the answer | Accurate with little tuning, handles 63 features, gives probabilities, gives feature importance (useful for your interface graph) | Slightly slower, larger saved file |
| **k-Nearest Neighbors** | Finds the k most similar hands in the training data and copies their letter | Very easy to explain | Sensitive to hand position and scale unless you normalize; slow at prediction |
| **Logistic Regression** | Draws mathematical boundaries between classes | Fast and simple | May struggle with complex hand shapes |
| Decision Tree | One tree of yes/no questions | Easy to understand | Usually less accurate, overfits easily |

**Recommendation:** use a Random Forest as your main model. If you have time, compare it to KNN or Logistic Regression for the "bonus" part of the assignment.

## Modules you'll use

- **pandas** to load and inspect the CSV (`pd.read_csv`)
- **numpy** for your own vectorized metric, which the project requires and a test should check against scikit-learn
- **scikit-learn** for everything ML:
  - `train_test_split`
  - `DummyClassifier` (the baseline)
  - `RandomForestClassifier`
  - `accuracy_score`, `f1_score`, `confusion_matrix`
  - `OneHotEncoder` (to turn "Left"/"Right" into numbers)
- **matplotlib** for exploration and result graphs
- **joblib** to save and load the model
- **json** for the metadata file
- **pytest** for tests

## Your part, step by step

**Step 1: Explore and clean**
- Load with `pd.read_csv`. Check `.info()`, `.describe()`, `.isna().sum()`, and `df["classe"].value_counts()`.
- Questions to answer: Any missing values? Are all letters represented about equally (class balance)? Any weird values, for example landmarks at 0,0,0 from failed detections?
- Make at least one saved graph (`savefig`), such as a bar chart of images per letter.

**Step 2: Preprocess (this matters a lot here)**
- Raw coordinates depend on *where* the hand is in the picture and how big it is. The same "A" at the left of the image has very different numbers from an "A" at the right.
- A standard fix is to subtract the wrist point (x0, y0, z0) from every point, then divide by the hand size, for example the wrist-to-middle-finger-base distance. Your interface must apply the **exact same** transformation to new images, so put it in one reusable function.
- Convert `main` into numbers with one-hot encoding (two columns: is_left, is_right).

**Step 3: Define the problem (Validation C with your TA)**
- Target: `classe`. Type: classification.
- Metric: **accuracy** (percent correct) plus **macro F1**, which treats every letter equally and catches the model failing on rare letters. Justify your choice.
- Model: Random Forest, with your reasons from the table above.

**Step 4: Split**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```
`random_state=42` makes results reproducible, and `stratify=y` keeps the letter proportions the same in both sets.

**Step 5: Baseline**
```python
from sklearn.dummy import DummyClassifier
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
```
It always guesses the most common letter. If your model barely beats it, something is wrong. (With about 24 letters, it should score around 4 to 5%.)

**Step 6: Train and evaluate**
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
Compute accuracy and F1 on the **test set only**, compare to the baseline, and plot a **confusion matrix**. It shows which letters get mixed up. Some pairs probably will be (for example M/N, or U/V), and the report should discuss this honestly as a limitation.

**Step 7: Save**
```python
joblib.dump(model, "modeles/modele.joblib")
```
Also write a **JSON metadata file** with the column names, the possible values of `main`, the min/max of the numeric columns, the list of letters, and the final scores. Your interface will read this.

**Step 8: Code structure and tests (for the project requirements)**
- Put this in `donnees.py` (loading, cleaning, variable classes) and `modele.py` (train, evaluate, predict), with a custom exception such as `DonneeInvalideError`.
- A decorator that times the training is an easy Chapter 7 requirement.
- Write at least 8 tests, for example: loading a missing file raises an error, the split sizes are right, your NumPy accuracy equals `accuracy_score` (use `pytest.approx`), and an invalid input raises your custom exception.

## Two things to raise with your team and TA

1. **The interface problem:** the project says the GUI needs one input field per feature. With 63 coordinates, nobody can type those by hand. You'll probably want the interface to take an *image* or webcam frame, run MediaPipe, and feed the landmarks to your model. Settle this at Validation D, and make sure the metadata JSON supports it.
2. **Data leakage:** if many rows come from near-identical frames of the same person, your test score may look better than real life. Mention it as a limitation, and if the CSV has any person or video ID, split by that instead of randomly.

If you want, I can write a starter `modele.py` with the classes, docstrings and type hints, or a first set of pytest tests.

## Réponse 3

# Revised plan: training only, test set untouched

Since your test data is already set aside and the landmarks are already normalized, your part gets simpler. The golden rule is that **the test set stays locked away until your final model is chosen**. Looking at it early, even once, and then adjusting your model to score better on it makes the final score dishonest. That is the main reason to keep it separate.

## Two things to check about your already-preprocessed data

- **x0, y0, z0 should all be 0.** You subtracted the wrist (point 0) from every point, so the wrist becomes the origin. A column that is always 0 gives the model no information. You can drop these 3 columns (leaving 60 numeric features), but then your interface must drop them too. Record this in the metadata JSON.
- **Confirm the scale.** After dividing by the norm, values should mostly fall between about -1 and 1. Check with `df.describe()`. If you see huge values or NaNs, some rows were probably divided by a norm close to 0 (a failed detection).

Your interface will need to apply this same normalization to live data, so keep that function in one place (for example `donnees.py`).

## Step-by-step plan

**Step 1: Load and inspect**
- `pd.read_csv`, then `.shape`, `.info()`, `.isna().sum()`, `.describe()`.
- Check that `main` has only the expected values (e.g. Left/Right) and see how `classe` is distributed with `value_counts()`.
- Look for duplicate rows (`df.duplicated().sum()`).

**Step 2: Clean**
- Remove or fix rows with missing values, all-zero landmarks, or extreme outliers.
- Write down in your DECISIONS.md what you removed and why (the project asks for this).

**Step 3: Make the exploration graph**
- A bar chart of images per letter, saved with `plt.savefig`. This also tells you if some letters are underrepresented, which matters for your metric choice later.

**Step 4: Separate inputs and target**
- `X` = all columns except `classe`; `y` = `classe`.
- Convert `main` to numbers with `OneHotEncoder`. Put the encoder and the model together in a scikit-learn `Pipeline`, so you save **one object** that handles everything. This avoids the classic bug where the interface encodes the input differently from training.

**Step 5: Pick the model and set it up**
- Start with `RandomForestClassifier(n_estimators=200, random_state=42)`. Random forests need no feature scaling, which suits your data. If you add KNN or logistic regression later, they are more sensitive to scaling.
- Validation C with your TA happens here, before training.

**Step 6: (Optional but recommended) Internal validation without touching the test set**
You don't have to evaluate yet, but if you later want to compare settings (for instance 100 vs 300 trees), do it inside your training data, not on the test set. Two ways:
- Hold out a **validation set** (e.g. 20% of the training data) with `train_test_split`.
- Use **cross-validation** (`cross_val_score`), which splits the training data several times automatically.

Watch for **leakage**: if your CSV came from video frames, consecutive frames look nearly identical. A random split puts near-copies in both parts, making scores look better than they are. If you have any person or video ID, use `GroupKFold`.

**Step 7: Train**
- `pipeline.fit(X_train, y_train)`, with a decorator that times it (this covers your Chapter 7 requirement).
- As a sanity check, only confirm it runs and that `predict_proba` gives one probability per letter for a few sample rows. This isn't evaluation, it just shows the pipeline works.

**Step 8: Save**
- `joblib.dump(pipeline, "modeles/modele.joblib")`
- Write a JSON metadata file: feature column names, dropped columns, possible values of `main`, min/max of each numeric column, the list of letters (`model.classes_`), and the random seed. Leave the metric field empty for now and fill it after final evaluation.

**Step 9: Structure and tests**
- `donnees.py` (load, clean, variable classes), `modele.py` (train, save, load, predict), `exceptions.py`.
- Tests you can already write: loading a missing file raises an error, cleaning removes bad rows, the output of `predict` is one of the known letters, the saved model reloads and gives the same predictions, the metadata JSON contains the expected keys.

## Resources

**Beginner-friendly explanations (start here)**
- *Kaggle Learn: "Intro to Machine Learning"* and *"Pandas"* are free, short, and interactive.
- *StatQuest with Josh Starmer* on YouTube: "Decision Trees" and "Random Forests" explain the idea very clearly with no heavy math.
- *Google Machine Learning Crash Course*: its sections on training/test sets and overfitting are especially relevant.

**Official documentation (the project asks you to cite at least 3 pages)**
- scikit-learn, "Getting started" and "Choosing the right estimator" (both recommended in your assignment).
- scikit-learn User Guide, "Ensembles: Gradient boosting, random forests" and the `RandomForestClassifier` API page.
- scikit-learn User Guide, "Cross-validation: evaluating estimator performance" (for Step 6).
- scikit-learn User Guide, "Pipelines and composite estimators" and `OneHotEncoder`.
- scikit-learn User Guide, "Model persistence" (explains `joblib` and its security and version caveats).
- pandas, "10 minutes to pandas".

**For understanding your data's origin**
- MediaPipe Hand Landmarker documentation (Google AI Edge) describes the 21 landmarks, so you know what x0 to x20 represent.

If you'd like, I can write a starter `modele.py` for steps 4 to 8 with docstrings, type hints and the timing decorator, so you have something concrete to run on your CSV.

## Réponse 4

# Scaling vs. what you've already done

## What "scaling" means

Scaling puts every feature (column) on a comparable range, so no column dominates just because its numbers are bigger.

Imagine predicting something from two columns: a person's age (0 to 100) and their salary (0 to 200,000). Some algorithms compare distances between rows, and in that comparison a salary difference of 5,000 completely drowns out an age difference of 30, even though both might matter equally. Scaling fixes this. The two common methods are:
- **Standardization** (`StandardScaler`): each column is shifted and divided so it has mean 0 and standard deviation 1.
- **Min-max scaling** (`MinMaxScaler`): each column is squeezed into 0 to 1.

## Is your normalization the same thing?

No, but it overlaps. They solve different problems:

| | What you did | Scaling |
|---|---|---|
| Works on | Each **row** (one hand) | Each **column** (one feature across all hands) |
| Goal | Remove the hand's position in the image and its size | Make features comparable in range |
| Effect | The same sign looks the same wherever the hand is | No single feature dominates by magnitude |

Your normalization is geometric and was essential. Scaling is statistical and is a separate, optional step.

## Is it still a problem for you?

**Mostly no.** All 60 of your coordinates are in the same unit (fractions of hand size), and they should all fall roughly between -1 and 1. That is already much closer to comparable than age vs. salary.

It isn't perfectly uniform, though. For example, the fingertips move a lot between signs, so their columns have a wide spread, while points near the wrist barely move. The z coordinates from MediaPipe also tend to have a different spread than x and y. For distance-based algorithms, the high-variance columns would have more influence. You can see this yourself with `df.describe()` and by comparing the `std` row across columns.

## What it means per algorithm

- **Random Forest (your main model): no scaling needed.** Each tree asks questions like "is y8 < 0.3?" on one column at a time, so the range of the column doesn't matter. You can skip it.
- **KNN: scaling matters.** It measures distances across all columns at once, so the spread of each column affects who counts as a "neighbor". With your data it may matter little, but testing it is cheap.
- **Logistic Regression: scaling helps.** It usually converges faster and more reliably when features have similar spreads (and you'll avoid convergence warnings).

## If you try KNN or Logistic Regression later

Put the scaler inside your `Pipeline`, so it's saved with the model and the interface applies it automatically:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
pipe.fit(X_train, y_train)
```

One rule: the scaler must learn its mean and standard deviation from the **training data only**. A Pipeline does this correctly for you, including inside cross-validation. Fitting a scaler on all the data before splitting is a common form of leakage.

One detail on the `main` column: it's already 0/1 after one-hot encoding, so you'd only scale the 60 coordinate columns. With a `ColumnTransformer` you can apply the scaler to those and the encoder to `main`. The scikit-learn docs page "Preprocessing data" explains `StandardScaler`, and "Column Transformer with Mixed Types" shows exactly this setup.

## Réponse 5

# Properly loading your CSV (Step 1)

## The plan

1. **Build the file path safely** with `os`, and check the file exists.
2. **Load** with `pd.read_csv`, inside a `with` block (the assignment requires `with` for every file opening).
3. **Check the structure**: expected columns, types, missing values.
4. **Check the `personne` column** before filtering. Look for typos like `"P1"` or `"p1 "` that would silently drop rows.
5. **Keep only p1 to p5.**
6. **Verify the result**: row count, letters per person, and that `personne` is **not** used as an input feature.

## Why keep `personne` after filtering?

Your model must not use `personne` as an input. If it did, it could learn "p3 usually signs these letters" instead of learning hand shapes, and it would fail on new people. But **keep the column in the DataFrame** and separate it from `X` later, because it is useful for cross-validation. With `GroupKFold`, each fold holds out entire people (for example, train on p1 to p4 and validate on p5). This is the honest way to estimate how the model does on a person it has never seen, and it solves the leakage problem I mentioned earlier.

## Starter code for `donnees.py`

```python
import os
import pandas as pd

from exceptions import DonneeInvalideError

COLONNES_ATTENDUES_MINIMUM = ["main", "personne", "classe"]
PERSONNES_ENTRAINEMENT = ["p1", "p2", "p3", "p4", "p5"]


def charger_csv(chemin: str) -> pd.DataFrame:
    """Charge le fichier CSV des points de repère.

    Reçoit : le chemin du fichier CSV.
    Renvoie : un DataFrame pandas contenant toutes les lignes du fichier.
    Échoue : FileNotFoundError si le fichier n'existe pas,
             DonneeInvalideError si des colonnes essentielles manquent.
    """
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    with open(chemin, "r", encoding="utf-8") as fichier:
        df = pd.read_csv(fichier)

    manquantes = [c for c in COLONNES_ATTENDUES_MINIMUM if c not in df.columns]
    if manquantes:
        raise DonneeInvalideError(f"Colonnes manquantes : {manquantes}")

    return df


def filtrer_personnes(df: pd.DataFrame, personnes: list[str]) -> pd.DataFrame:
    """Garde uniquement les lignes appartenant aux personnes données.

    Reçoit : le DataFrame complet et la liste des personnes à garder.
    Renvoie : une copie du DataFrame filtré.
    Échoue : DonneeInvalideError si une personne demandée est absente
             ou si le filtre ne laisse aucune ligne.
    """
    presentes = set(df["personne"].unique())
    absentes = [p for p in personnes if p not in presentes]
    if absentes:
        raise DonneeInvalideError(f"Personnes absentes du fichier : {absentes}")

    return df[df["personne"].isin(personnes)].copy()
```

And a small script (or notebook cell) to **look at** your data. Run it and read the output carefully:

```python
df = charger_csv("donnees/asl_landmarks.csv")

# Aperçu général
print(df.shape)                       # (lignes, colonnes)
print(df.head())                      # 5 premières lignes
df.info()                             # types et valeurs non nulles

# Vérifications avant le filtrage
print(df["personne"].value_counts())  # combien de lignes par personne ?
print(df["personne"].unique())        # repérer les fautes de frappe
print(df["main"].unique())            # valeurs de la main

# Filtrage p1 à p5
df_train = filtrer_personnes(df, PERSONNES_ENTRAINEMENT)

# Vérifications après le filtrage
print(df_train.shape)                 # environ la moitié des lignes ?
print(df_train["personne"].unique())  # seulement p1 à p5 ?
print(df_train.isna().sum().sum())    # total de valeurs manquantes
print(df_train.duplicated().sum())    # lignes dupliquées
print(df_train["classe"].value_counts())
print(pd.crosstab(df_train["personne"], df_train["classe"]))
```

## What to look for in the output

- **`df.shape`**: you should now have 66 columns (the original 65 plus `personne`).
- **`personne` values**: exactly `p1` to `p10`, with no variants. If you see `P1` or `"p1 "`, clean them with `df["personne"] = df["personne"].str.strip().str.lower()` before filtering.
- **Row count after filtering**: "half the entries" is only true if each person contributed a similar number of rows. `value_counts()` will tell you.
- **The crosstab** (rows are people, columns are letters): every person should have every letter. If p4 is missing the letter "G", the model never sees that letter in some folds, which affects later validation.
- **Missing values**: `isna().sum().sum()` should be 0. If it isn't, find which columns with `df.isna().sum()` and decide to drop or fix the rows (note it in DECISIONS.md).
- **Wrist columns**: check that `x0`, `y0` and `z0` are all 0 with `(df_train[["x0", "y0", "z0"]] == 0).all()`. If so, they carry no information and can be dropped in Step 2.

## Two small notes

- Since `charger_csv` and `filtrer_personnes` return their results and don't print anything, they are easy to test. Two tests you can write now: a missing file raises `FileNotFoundError`, and filtering p1 to p5 leaves only those people (use a tiny DataFrame you build by hand in the test).
- Read the full file first and filter afterward, as above, rather than filtering while reading. That way you can look at the whole dataset and confirm that nothing is wrong with `personne` before throwing half of it away.

When you've run this, paste the output of `value_counts()` and the crosstab if you'd like help interpreting it, then we can move on to cleaning (Step 2).

## Réponse 6

I'll write a `modele.py` that covers training and saving. It leaves the test set alone and does no evaluation yet.

# Training your model (Steps 4 to 8)

With no missing rows and balanced data, you can go straight to training. The `modele.py` file above does it in a few functions.

## What the code does

1. **`preparer_entrees`** splits your DataFrame into **X** (the inputs) and **y** (the answer to guess).
   - X is the 60 coordinates (x1 to z20) plus `main`.
   - It removes `classe` (that is y), `personne` (the model must not learn "who" is signing), and `x0, y0, z0`.
   - It also checks that the wrist columns are really 0, so it would catch a normalization mistake.
2. **`construire_pipeline`** builds one object that does two things in order:
   - It turns `main` into numbers (one-hot encoding), because a Random Forest only accepts numbers.
   - It runs the Random Forest, which is 200 decision trees that vote on the letter.
3. **`entrainer`** calls `.fit()`, which is the actual learning. The model looks at every training row with its correct letter and builds its trees. The `@chronometrer` decorator prints the duration, which covers your Chapter 7 requirement.
4. **`sauvegarder`** writes the trained pipeline to a `.joblib` file and the metadata to JSON. The metric is left as `None` until the final evaluation.

## How to run it

```python
from donnees import charger_csv, filtrer_personnes, PERSONNES_ENTRAINEMENT
from modele import preparer_entrees, construire_pipeline, entrainer, sauvegarder

df = charger_csv("donnees/asl_landmarks.csv")
df_train = filtrer_personnes(df, PERSONNES_ENTRAINEMENT)

X, y = preparer_entrees(df_train)
pipeline = construire_pipeline()
entrainer(pipeline, X, y)

sauvegarder(pipeline, X, "modeles/modele.joblib", "modeles/metadonnees.json")
```

## Why there is no `train_test_split`

Your test data is already stored separately, so all of `df_train` (p1 to p5) is used for training. Nothing else needs splitting.

## A quick check that it worked

This is not an evaluation, only a check that the pipeline runs:

```python
echantillon = X.head(3)
print(pipeline.predict(echantillon))
print(pipeline.predict_proba(echantillon).shape)  # (3, nombre de lettres)
```

Don't be surprised if the predictions on training rows are almost all correct. A Random Forest can nearly memorize its training data, so this says nothing about how it performs on new hands. That is exactly why the test set exists.

## Optional: choosing settings without using the test set

If you later want to compare settings (e.g. 100 vs 300 trees), use cross-validation by person on your training data:

```python
from sklearn.model_selection import GroupKFold, cross_val_score

scores = cross_val_score(
    construire_pipeline(), X, y,
    groups=df_train["personne"], cv=GroupKFold(n_splits=5),
)
print(scores.mean(), scores.std())
```

Each round trains on 4 people and checks on the 5th, so it measures how the model does on someone it has never seen. It also tells you early whether your final test score is likely to be good. Note this is still training-side work, so it respects your rule of not touching the test set.

## Tests you can write now

- `preparer_entrees` raises `DonneeInvalideError` if `classe` is missing.
- `preparer_entrees` raises `DonneeInvalideError` if the wrist columns are non-zero.
- `personne`, `classe`, `x0`, `y0` and `z0` are not in X.
- `entrainer` raises `DonneeInvalideError` when X and y have different lengths.
- Save, reload with `charger_modele`, and check that the predictions are identical.
- The metadata JSON contains the expected keys, and `classes` matches the letters in your CSV.

## Things to watch

- `joblib` files can be large for a forest trained on many rows. If the file exceeds 25 MB, check the assignment's rules before committing it to the public repo, or reduce `n_estimators`.
- A saved model only reloads reliably with the same scikit-learn version. Pin it in `requirements.txt`.
- Record in DECISIONS.md that you trained on p1 to p5 only, why, and that the wrist columns were dropped.

Let me know if you'd like the pytest file for these, or help interpreting the cross-validation results.

## Réponse 7

# The three libraries

## `functools`: optional
I used one tool from it, `functools.wraps`, inside the timing decorator. A decorator replaces your function with a wrapper function, and without `wraps` the wrapper hides the original's name and docstring. For instance, `entrainer.__name__` would say `"enveloppe"`, and `help(entrainer)` would show nothing. This is purely cosmetic, and your decorator works without it. If you remove it, delete the `@functools.wraps(fonction)` line and the import. The assignment asks for docstrings on every function, so keeping it is nicer, but it isn't required.

## `typing`: not necessary
I only used it for `Callable`, which is a type annotation meaning "a function". Annotations don't change how the code runs. You have two options:
- Import it from the standard library instead: `from collections.abc import Callable`.
- Remove it and write `def chronometrer(fonction):` with no annotations. The assignment asks for annotations on your functions, so I'd keep the `collections.abc` version.

Your other annotations (`list[str]`, `tuple[...]`, `dict`) are built into Python 3.9+ and need no import.

## `joblib`: used for saving the model
`joblib.dump` writes your trained pipeline to a file, and `joblib.load` reads it back, so you don't retrain every time the interface launches. It is not really an unknown library: it installs automatically with scikit-learn, and your assignment names it explicitly for this step. The only alternative is `pickle` from the standard library, which does the same job (`pickle.dump(obj, fichier)` with the file opened in `"wb"` mode). It works, but joblib is better for scikit-learn models and is what the documentation recommends, so I'd keep it.

# Keeping the wrist columns

Small correction first: the always-zero columns are **x0, y0 and z0** (the wrist point). x1 and x2 are real landmarks (thumb), and they are not constant, so they should stay anyway.

**Keeping x0, y0, z0 will not noticeably affect your result.** A Random Forest splits using questions like "is y8 < 0.3?". A column that is always 0 can't separate any rows into two groups, so no tree ever picks it. It is effectively ignored. The only cost is 3 extra columns of zeros, which is negligible in memory and speed. It would matter more for KNN or logistic regression, but you chose Random Forest.

Keeping them also has a small advantage: your interface can send all 21 landmarks without special handling.

To keep them, change these parts of `modele.py`:

```python
# 1. Dans preparer_entrees : on garde la vérification, mais on ne retire plus les colonnes
    a_retirer = [COLONNE_CIBLE, COLONNE_PERSONNE]
    X = df.drop(columns=[c for c in a_retirer if c in df.columns])
```

```python
# 2. Dans creer_metadonnees : plus aucune colonne retirée
        "colonnes_retirees": [],
```

You can keep the `np.allclose` check on the wrist columns. It no longer removes anything, but it still catches a normalization mistake, which is useful. If you'd rather not check at all, delete that block along with the `numpy` import (note that `numpy` is also what your assignment requires for your own metric later, so you'll use it elsewhere).

Your tests change slightly: "x0, y0, z0 are not in X" becomes "x0, y0, z0 **are** in X", and `personne` and `classe` are still absent. Update your DECISIONS.md entry too: you kept the wrist columns, because a Random Forest ignores constant columns.

## Réponse 8

I've finished the edits to `modele.py` and I'm now presenting the updated file.

`modele.py` is updated with the changes you asked for.

## What changed

- **`functools` removed.** I deleted the import and the `@functools.wraps(fonction)` line. The `chronometrer` decorator works the same, but `entrainer.__name__` will now show `"enveloppe"` instead of `"entrainer"`.
- **Import changed** to `from collections.abc import Callable`.
- **`joblib` untouched**, as you asked.
- **Wrist columns kept.** `preparer_entrees` now removes only `classe` and `personne`, so `x0, y0, z0` stay in X. The check that they are all 0 is still there, so it will raise `DonneeInvalideError` if your normalization went wrong. The metadata field `colonnes_retirees` is now an empty list.

A note on column names: you wrote x0, x1 and x2, but the always-zero columns are **x0, y0 and z0** (the wrist point). x1 and x2 are real thumb landmarks and were never removed, so they are in X either way.

## Test updates

- "x0, y0, z0 are not in X" becomes "x0, y0, z0 **are** in X".
- `personne` and `classe` must still be absent from X.
- Keep the test that non-zero wrist values raise `DonneeInvalideError`.

Record in DECISIONS.md that you kept the wrist columns because a Random Forest ignores constant columns.

## Réponse 9

# Changes for the missing `main` column

`modele.py` is updated. Since every remaining input is a number, there is nothing to encode, which simplifies things:

- **Removed:** `OneHotEncoder`, `ColumnTransformer`, `Pipeline`, and the `main` constant and checks. A Pipeline with a single step adds nothing, so the code now uses the `RandomForestClassifier` directly.
- **Renamed:** `construire_pipeline` is now `construire_modele`, and the other functions take `modele` instead of `pipeline`.
- **Metadata:** `variables_numeriques` now covers every input column. The categorical entry and `colonnes_retirees` are gone.

Your run script changes accordingly:

```python
X, y = preparer_entrees(df_train)
modele = construire_modele()
entrainer(modele, X, y)
sauvegarder(modele, X, "modeles/modele.joblib", "modeles/metadonnees.json")
```

You also need to edit `donnees.py`. Remove `"main"` from `COLONNES_ATTENDUES_MINIMUM`, otherwise `charger_csv` will raise an error about a missing column. Likewise, delete the `print(df["main"].unique())` line from your inspection script. The cross-validation snippet only needs `construire_modele()` in place of `construire_pipeline()`.

# What `GRAINE` is for

A Random Forest uses randomness when it is built. Each of the 200 trees is trained on a random sample of your rows and looks at random subsets of the columns at each decision. Without a fixed seed, every run produces a slightly different forest and slightly different results.

`random_state=GRAINE` (seed 42) fixes that randomness, so the same data gives exactly the same model every time. That makes your results reproducible, which the assignment asks for, and it lets you tell whether a change in score came from something you did or just from luck. The number 42 itself is arbitrary, and any integer works. Storing it in a named constant means it is defined in one place and gets saved in the metadata JSON.

# What `from exceptions import DonneeInvalideError` does

It imports the custom exception class you define in `exceptions.py`, which looks like this:

```python
class DonneeInvalideError(Exception):
    """Levée quand les données reçues sont invalides."""
```

`modele.py` uses it in `preparer_entrees` and `entrainer` with `raise DonneeInvalideError(...)`. That gives you clear, specific errors such as "missing column" or "X and y have different sizes", instead of a generic crash from deep inside pandas or scikit-learn.

**Can you remove it?** Technically yes, but only if you also remove every `raise DonneeInvalideError(...)` line. Otherwise Python raises a `NameError` when those lines run. I'd keep it, because the assignment requires at least one custom exception raised with `raise`, and it is graded under code quality. Not writing tests yet doesn't change that. You will just need to make sure `exceptions.py` exists in the same folder as `modele.py`.

## Réponse 10

`modele.py` now has both changes: a structured `metrique` field and an example `if __name__ == "__main__":` block at the end that calls the functions.

## The `metrique` field

It is now a small dictionary with empty slots, instead of a single `None`:

```python
"metrique": {
    "nom": "exactitude et F1 macro",
    "exactitude": None,
    "f1_macro": None,
    "reference_exactitude": None,
},
```

- **`exactitude`** (accuracy) is the percentage of correct predictions.
- **`f1_macro`** is the F1 score averaged over all letters, so each letter counts equally. It catches the model doing badly on a few letters while the overall accuracy still looks fine.
- **`reference_exactitude`** is the score of the trivial baseline model (`DummyClassifier`). The assignment wants the metric shown next to this reference, otherwise you can't tell whether your score is good.

All of these stay `None` until the final evaluation on your test set, since you aren't evaluating yet. When the time comes, you can fill them in by reading the JSON, updating the values and writing it back.

If you run cross-validation on p1 to p5 before then, you could add a separate key such as `"validation_croisee"` for that score. Keep it apart from the test-set scores so the two are never confused.

## Suggested paths

```python
chemin_csv    = os.path.join("donnees", "asl_landmarks.csv")
chemin_modele = os.path.join("modeles", "modele.joblib")
chemin_meta   = os.path.join("modeles", "metadonnees.json")
```

These match the repository layout in your assignment (`donnees/` and `modeles/`). I used `os.path.join` because the assignment asks for paths built with the `os` module. `sauvegarder` creates the `modeles/` folder if it doesn't exist.

## Running it

The paths are relative, so run the script from the **root of your repository**, not from inside `src/`:

```
python src/modele.py
```

This assumes `donnees.py` and `exceptions.py` are in the same `src/` folder, and that `donnees.py` no longer requires the `main` column (see my previous message). When you build `main.py`, you can move this block there as the entry point. Until then, running it confirms that loading, training and saving work end to end.