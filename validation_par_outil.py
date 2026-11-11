"""Évalue séparément le cathéter, le guidewire et leur union."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import dilation, disk

from segmentation import lire_image
from separation_outils import ligne_separation
from validation import compter_erreurs, exporter_scores, mesurer
from visualisation import ecrire_png, exporter_gif, planche, superposer, tracer_scores


def lire_masque(chemin):
    image = lire_image(chemin)
    if not np.all((image == 0) | (image == 255)):
        raise ValueError(f"Masque non binaire : {chemin}")
    return image == 0


def exporter_erreurs(dossier, erreurs):
    with (dossier / "erreurs.csv").open("w", newline="", encoding="utf-8") as fichier:
        writer = csv.writer(fichier)
        writer.writerow(("Image", "VP", "FP", "FN", "Rappel", "Précision"))
        writer.writerows(erreurs)


def valider(images, resultats, sortie):
    dossiers = {outil: sortie / outil for outil in ("catheter", "guidewire", "union")}
    for dossier in dossiers.values():
        dossier.mkdir(parents=True, exist_ok=True)

    scores = {outil: [] for outil in dossiers}
    erreurs = {outil: [] for outil in dossiers}
    vues_union = {}
    animation_union = []
    contraste = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    disque = disk(2)

    for numero in range(211, 292, 10):
        image = lire_image(images / f"frame_{numero}.png")
        catheter = lire_masque(resultats / "fusion_outils" / f"masque_catheter_{numero}.png")
        guidewire = lire_masque(resultats / "fusion_outils" / f"masque_guidewire_{numero}.png")

        ref_catheter = dilation(lire_image(images / f"{numero}_MicroCath.tiff") == 0, disque)
        ref_guidewire = dilation(lire_image(images / f"{numero}_GuideWire.tiff") == 0, disque)
        fond = contraste.apply(image)
        vue_union = superposer(fond, catheter, guidewire)
        vues_union[str(numero)] = cv2.cvtColor(vue_union, cv2.COLOR_BGR2RGB)
        animation_union.append(vue_union)

        mesures = {
            "catheter": (catheter, ref_catheter, superposer(fond, catheter),
                         "Cathéter détecté", "TIFF MicroCath dilaté"),
            "guidewire": (guidewire, ref_guidewire, superposer(fond, repere=guidewire),
                          "Guidewire détecté", "TIFF GuideWire dilaté"),
            "union": (catheter | guidewire, ref_catheter | ref_guidewire, vue_union,
                      "Union des détections", "Union des deux TIFF dilatés"),
        }
        if numero == 291:
            annotations = superposer(image, ref_catheter, ref_guidewire)
            y = ligne_separation(image.shape[0])
            cv2.line(annotations, (0, y), (image.shape[1] - 1, y), (255, 180, 0), 1)
            planche(sortie / "annotations_291.png", {
                "291_MicroCath.tiff · dilaté": ~ref_catheter,
                "291_GuideWire.tiff · dilaté": ~ref_guidewire,
                f"Annotations · coupe y = {y}": cv2.cvtColor(annotations, cv2.COLOR_BGR2RGB),
            }, colonnes=3)

        for outil, (estimation, reference, vue, titre, titre_reference) in mesures.items():
            dossier = dossiers[outil]
            scores[outil].append((numero, *mesurer(estimation, reference)))
            erreurs[outil].append((numero, *compter_erreurs(estimation, reference)))
            if outil == "union":
                ecrire_png(dossier / f"superposition_{numero}.png", vue)
            elif numero in (211, 291):
                planche(dossier / f"comparaison_{numero}.png", {
                    titre: ~estimation,
                    titre_reference: ~reference,
                    "Superposition": cv2.cvtColor(vue, cv2.COLOR_BGR2RGB),
                }, colonnes=3)

    for outil, dossier in dossiers.items():
        print(f"\n{outil}")
        exporter_scores(dossier, scores[outil])
        exporter_erreurs(dossier, erreurs[outil])
        tracer_scores(dossier / "scores.png", scores[outil])
    planche(dossiers["union"] / "detections_completes.png", vues_union, colonnes=3)
    exporter_gif(dossiers["union"] / "animation.gif", animation_union)


def main():
    projet = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=projet / "catheter")
    parser.add_argument("--resultats", type=Path, default=projet / "results")
    parser.add_argument("--sortie", type=Path)
    args = parser.parse_args()
    valider(args.images, args.resultats, args.sortie or args.resultats / "validation_par_outil")


if __name__ == "__main__":
    main()
