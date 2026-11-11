"""Superpose les détections en vert sur les images originales."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    projet = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=projet / "catheter")
    parser.add_argument("--masques", type=Path, default=projet / "masques_consecutifs")
    parser.add_argument("--sortie", type=Path, default=projet / "resultats_masques")
    args = parser.parse_args()

    args.sortie.mkdir(parents=True, exist_ok=True)
    for numero in range(211, 292, 10):
        chemin_image = args.images / f"frame_{numero}.png"
        chemin_masque = args.masques / f"masque_{numero}.png"
        image = cv2.imread(str(chemin_image), cv2.IMREAD_COLOR)
        masque = cv2.imread(str(chemin_masque), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"Image illisible : {chemin_image}")
        if masque is None:
            raise OSError(f"Masque illisible : {chemin_masque}")
        if masque.shape != image.shape[:2]:
            raise ValueError(f"Dimensions incompatibles pour l’image {numero}.")
        if not np.all((masque == 0) | (masque == 255)):
            raise ValueError(f"Le masque doit être binaire : {chemin_masque}")

        image[masque == 0] = (0, 255, 0)
        sortie = args.sortie / f"resultat_{numero}.png"
        if not cv2.imwrite(str(sortie), image):
            raise OSError(f"Écriture impossible : {sortie}")
        print(sortie)


if __name__ == "__main__":
    main()
