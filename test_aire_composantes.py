"""Compare visuellement six aires minimales après Otsu + 30."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from segmentation import lire_image, segmenter_outil
from visualisation import planche


AIRES_MINIMALES = (20, 100, 200, 275, 300, 500)


def comparer_aires(masque_otsu):
    """Produit les masques d’affichage à partir d’un seul étiquetage en 8-connexité."""
    _, etiquettes, stats, _ = cv2.connectedComponentsWithStats(
        (masque_otsu == 0).astype(np.uint8), connectivity=8,
    )
    vues = {}
    for aire in AIRES_MINIMALES:
        garder = stats[:, cv2.CC_STAT_AREA] >= aire
        garder[0] = False
        vues[f"Aire ≥ {aire} px"] = ~garder[etiquettes]
    return vues


def main():
    projet = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeros", nargs="+", type=int, default=[291])
    parser.add_argument("--images", type=Path, default=projet / "catheter")
    parser.add_argument("--sortie", type=Path,
                        default=projet / "results" / "intermediaires" / "aires_composantes")
    args = parser.parse_args()

    reference = lire_image(args.images / "frame_201.png")
    args.sortie.mkdir(parents=True, exist_ok=True)
    for numero in args.numeros:
        image = lire_image(args.images / f"frame_{numero}.png")
        etapes, _ = segmenter_outil(reference, image)
        vues = comparer_aires(etapes["Otsu + 30"])
        chemin = args.sortie / f"aires_{numero}.png"
        planche(chemin, vues, colonnes=len(vues))
        print(chemin)


if __name__ == "__main__":
    main()
