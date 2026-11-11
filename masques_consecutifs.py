"""Détecte les outils sur les images 211 à 291 par soustraction successive."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def lire_image(chemin):
    if not chemin.is_file():
        raise FileNotFoundError(f"Image introuvable : {chemin}")
    image = cv2.imread(str(chemin), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image illisible : {chemin}")
    return image


def calculer_masque(reference, image):
    """Renvoie un masque uint8 avec les outils en noir et le fond en blanc."""
    if reference.shape != image.shape:
        raise ValueError("Les deux images doivent avoir les mêmes dimensions.")

    difference = reference.astype(np.float32) - image.astype(np.float32)
    gauss = cv2.GaussianBlur(difference, (5, 5), 1)

    hauteur, largeur = gauss.shape
    y, x = np.ogrid[:hauteur, :largeur]
    distance2 = (y - hauteur // 2)**2 + (x - largeur // 2)**2
    passe_haut = 1 - np.exp(-distance2 / (2 * 10**2))
    passe_bas = np.exp(-distance2 / (2 * 60**2))
    passe_bande = (passe_haut * passe_bas).astype(np.float32)

    spectre = np.fft.fftshift(cv2.dft(gauss, flags=cv2.DFT_COMPLEX_OUTPUT), axes=(0, 1))
    spectre_filtre = np.fft.ifftshift(spectre * passe_bande[:, :, None], axes=(0, 1))
    filtree = cv2.idft(spectre_filtre, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # Estimation robuste du bruit de la différence filtrée.
    centree = filtree - np.median(filtree)
    bruit = 1.4826 * np.median(np.abs(centree))

    # Le seuil bas suit les contours. Le seuil haut valide chaque composante.
    candidats = (centree > 2 * bruit).astype(np.uint8)
    _, etiquettes, stats, _ = cv2.connectedComponentsWithStats(candidats, connectivity=8)
    graines = np.bincount(etiquettes[centree > 4 * bruit], minlength=len(stats))
    garder = (graines > 0) & (stats[:, cv2.CC_STAT_AREA] >= 60)
    garder[0] = False
    outil = garder[etiquettes].astype(np.uint8)
    noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    outil = cv2.morphologyEx(outil, cv2.MORPH_CLOSE, noyau)
    return (1 - outil) * 255


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images", nargs="?", type=Path,
        default=Path(__file__).resolve().parent / "catheter",
        help="dossier contenant les images frame_*.png",
    )
    parser.add_argument(
        "--sortie", type=Path, default=Path("masques_consecutifs"),
        help="dossier de destination des masques PNG",
    )
    args = parser.parse_args()

    reference = lire_image(args.images / "frame_201.png")
    args.sortie.mkdir(parents=True, exist_ok=True)
    for numero in range(211, 292, 10):
        image = lire_image(args.images / f"frame_{numero}.png")
        masque = calculer_masque(reference, image)
        destination = args.sortie / f"masque_{numero}.png"
        if not cv2.imwrite(str(destination), masque):
            raise OSError(f"Impossible d’enregistrer le masque : {destination}")
        print(f"{numero - 10} → {numero} : {destination}")
        reference = image


if __name__ == "__main__":
    main()
