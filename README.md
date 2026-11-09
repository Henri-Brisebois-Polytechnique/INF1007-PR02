# INF1007-PR02


## Procédure d'instalation:

Créer un environnement virtuel dans le fichier projet:

```bash
cd INF1007-PR02
python -m venv .venv
```

Activer l'environnement:

```bash
##Windows
.venv\Scripts\Activate.ps1

##Linux ou MacOS
source .venv/bin/activate
```

Selectionner le bon interpreteur avec CTRL+SHIFT+P -> “Python: Select Interpreter” -> choisir celui à l'intérieur du .venv

Installer les packages nécessaires:

```bash
pip install -r requirements.txt
```