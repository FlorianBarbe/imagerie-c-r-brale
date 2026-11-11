# Imagerie cérébrale · TP ATRIM

Le [compte rendu fusionné](compte_rendu_atrim_fusionne.ipynb) présente la méthode et les résultats. Le pipeline segmente d’abord un masque commun avec la référence 201, les composantes connexes d’aire minimale 300 pixels et une fermeture 3×3. Sur une image de 512 lignes, les pixels du masque au-dessus de la ligne 348 sont affichés en rouge (guidewire), et ceux de la ligne 348 ou en dessous en vert (microcathéter).

Avant la séparation, trois zones parasites sont exclues du masque : le bord supérieur gauche (x < 45 et y < 240), le bord inférieur horizontal (290 ≤ x < 438 et y ≥ 485) et le bord inférieur droit (x ≥ 456 et y ≥ 348), en coordonnées 512 × 512. Ces zones ne recoupent aucun pixel des annotations dilatées sur la séquence.

La coupure a été choisie en examinant les TIFF de cette séquence. L’attribution à chaque outil utilise la coupure fixe. Les annotations TIFF servent aussi à l’évaluation des masques. Le suivi temporel cumule les nouvelles détections du guidewire dans la zone haute.

Depuis la racine du projet, avec les dépendances installées dans `.venv` :

```bash
./.venv/bin/python main.py
./.venv/bin/python comparer_filtres.py
./.venv/bin/python test_aire_composantes.py
./.venv/bin/python validation_par_outil.py
./.venv/bin/python test_cumul_guidewire.py
```

Les superpositions par outil sont dans `results/catheter_vert/` et `results/guidewire_rouge/`. La vue combinée est dans `results/fusion_outils/`, la comparaison aux TIFF dans `results/validation_par_outil/` et le suivi temporel dans `results/suivi_pointe_guidewire/` et `results/cumul_pointe_guidewire/`.
