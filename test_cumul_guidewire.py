"""Cumule les détections de la pointe du fil-guide et les compare aux TIFF."""

import argparse
from pathlib import Path

import numpy as np
from skimage.morphology import dilation, disk

from segmentation import lire_image
from validation import exporter_scores, mesurer
from visualisation import ecrire_png, exporter_gif, planche, superposer, tracer_scores


def main():
    projet = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=projet / "catheter")
    parser.add_argument("--resultats", type=Path, default=projet / "results")
    args = parser.parse_args()

    source = args.resultats / "suivi_pointe_guidewire"
    sortie = args.resultats / "cumul_pointe_guidewire"
    comparaison = sortie / "reference_guidewire_cumulee"
    comparaison.mkdir(parents=True, exist_ok=True)

    cumul = None
    cumul_reference = None
    scores_courants = []
    scores_cumules = []
    noyau_annotation = disk(2)
    masques_cumules = {}
    vues_cumulees = []

    for numero in range(211, 292, 10):
        chemin = source / f"masque_{numero}.png"
        masque = lire_image(chemin)
        if not np.all((masque == 0) | (masque == 255)):
            raise ValueError(f"Masque non binaire : {chemin}")

        noir = masque == 0
        if cumul is None:
            cumul = np.zeros_like(noir)
            cumul_reference = np.zeros_like(noir)
        if noir.shape != cumul.shape:
            raise ValueError(f"Dimensions incompatibles : {chemin}")

        cumul |= noir
        masque_cumule = ~cumul
        ecrire_png(sortie / f"masque_cumul_{numero}.png", masque_cumule.astype(np.uint8) * 255)
        masques_cumules[str(numero)] = masque_cumule

        annotation = lire_image(args.images / f"{numero}_GuideWire.tiff") == 0
        reference = dilation(annotation, noyau_annotation)
        scores_courants.append((numero, *mesurer(cumul, reference)))
        cumul_reference |= reference
        scores_cumules.append((numero, *mesurer(cumul, cumul_reference)))

        image = lire_image(args.images / f"frame_{numero}.png")
        vue = superposer(image, repere=cumul)
        ecrire_png(sortie / f"trajectoire_{numero}.png", vue)
        vues_cumulees.append(vue)

    exporter_scores(sortie, scores_courants)
    tracer_scores(sortie / "scores.png", scores_courants)
    exporter_scores(comparaison, scores_cumules)
    tracer_scores(comparaison / "scores.png", scores_cumules)
    planche(comparaison / "comparaison_291.png", {
        "Détections de pointe cumulées": ~cumul,
        "Annotations GuideWire cumulées": ~cumul_reference,
    }, colonnes=2)
    planche(sortie / "masques_cumules.png", masques_cumules, colonnes=3)
    exporter_gif(sortie / "trajectoire.gif", vues_cumulees)


if __name__ == "__main__":
    main()
