# Imagerie cérébrale · compte rendu ATRIM

Ouvrir [le notebook du TP5](TP5-background-substraction-sv.ipynb) dans JupyterLab. Le compte rendu et ses figures suivent l’énoncé.
Le code et les données permettent de reproduire les masques des deux outils et leur validation.

Depuis ce dossier, avec Python 3.10 ou plus récent :

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python main.py
./.venv/bin/python comparer_filtres.py
./.venv/bin/python test_aire_composantes.py
```

Les masques binaires sont dans `results/fusion_outils/`. Les scores et les superpositions
sont dans `results/validation_par_outil/`.
