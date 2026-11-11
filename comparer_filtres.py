"""Compare les variantes du pipeline ATRIM, un traitement à la fois."""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from segmentation import AIRE_MINIMALE_COMPOSANTE, lire_image


def filtrer_fourier(image, variante):
    """Applique la réponse fréquentielle au spectre centré."""
    if variante == "Sans filtre":
        return image
    hauteur, largeur = image.shape
    y, x = np.ogrid[:hauteur, :largeur]
    r2 = (y - hauteur // 2) ** 2 + (x - largeur // 2) ** 2
    passe_haut = 1 - np.exp(-r2 / (2 * 10**2))
    if variante == "Passe-haut Butterworth":
        reponse = 1 - 1 / (1 + (r2 / 12**2) ** 2)
    elif variante == "Passe-bande gaussien":
        reponse = passe_haut * np.exp(-r2 / (2 * 60**2))
    else:
        reponse = passe_haut
    spectre = np.fft.fftshift(cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT), axes=(0, 1))
    produit = spectre * reponse.astype(np.float32)[:, :, None]
    return cv2.idft(
        np.fft.ifftshift(produit, axes=(0, 1)),
        flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT,
    )


def transformer_histogramme(image, variante):
    """Étire la partie positive dans la plage 0–255."""
    positive = np.maximum(image, 0)
    limite = np.percentile(positive, 99) if variante == "99e percentile" else positive.max()
    if limite <= 0:
        return np.zeros_like(positive, dtype=np.uint8)
    return np.rint(255 * np.clip(positive / limite, 0, 1)).astype(np.uint8)


def seuiller(contraste, delta):
    otsu, _ = cv2.threshold(contraste, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return contraste > min(otsu + delta, 255)


def retirer_petites_composantes(outil):
    _, etiquettes, stats, _ = cv2.connectedComponentsWithStats(
        outil.astype(np.uint8), connectivity=8
    )
    garder = stats[:, cv2.CC_STAT_AREA] >= AIRE_MINIMALE_COMPOSANTE
    garder[0] = False
    return garder[etiquettes]


def fermer(outil, taille):
    noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (taille, taille))
    return cv2.morphologyEx(outil.astype(np.uint8), cv2.MORPH_CLOSE, noyau).astype(bool)


def comparer(reference, image):
    difference = reference.astype(np.float32) - image.astype(np.float32)
    gauss = cv2.GaussianBlur(difference, (5, 5), 1)
    # Conversion de sigmaColor depuis l’échelle normalisée 0–255.
    sigma_couleur = 50 * float(np.ptp(difference)) / 255
    passe_bande = "Passe-bande gaussien"
    spectres = {
        nom: filtrer_fourier(gauss, nom)
        for nom in ("Sans filtre", "Passe-haut gaussien", "Passe-haut Butterworth", passe_bande)
    }
    filtree = spectres[passe_bande]
    contraste = transformer_histogramme(filtree, "99e percentile")
    binaire = seuiller(contraste, 30)
    composantes = retirer_petites_composantes(binaire)

    return {
        "spatial": {
            "Sans filtre": difference,
            "Gaussien 5×5": gauss,
            "Bilatéral 5×5": cv2.bilateralFilter(difference, 5, sigma_couleur, 50),
            "Médian 5×5": cv2.medianBlur(difference, 5),
        },
        "spectral": spectres,
        "histogramme": {
            "Maximum": transformer_histogramme(filtree, "Maximum"),
            "99e percentile": contraste,
        },
        "segmentation": {
            "Otsu" if delta == 0 else f"Otsu + {delta}": seuiller(contraste, delta)
            for delta in (0, 15, 30, 50)
        },
        "morphologie": {
            "Seuil seul": binaire,
            f"Composantes ≥ {AIRE_MINIMALE_COMPOSANTE} px": composantes,
            "Composantes + fermeture 3×3": fermer(composantes, 3),
            "Composantes + fermeture 5×5": fermer(composantes, 5),
        },
    }


def exporter_planche(sortie, etape, variantes, numero, image):
    titres = {
        "difference": "Soustraction de la référence fixe",
        "spatial": "Filtrage spatial de la différence",
        "spectral": "Filtrage spectral",
        "histogramme": "Étirement des niveaux",
        "segmentation": "Choix du seuil de segmentation",
        "morphologie": "Nettoyage des masques",
    }
    if etape == "difference":
        difference = variantes["Soustraction I201 − Ik"]
        borne = max(float(np.percentile(np.abs(difference), 99)), 1.0)
        vues = [
            (f"Radiographie {numero}", image, 0, 255),
            (f"Différence 201 − {numero}", difference, -borne, borne),
        ]
    elif etape in ("spatial", "spectral"):
        if etape == "spatial":
            reponses = {"Avant filtrage" if nom == "Sans filtre" else nom: vue
                        for nom, vue in variantes.items()}
        else:
            reponses = {"Sans filtre spectral" if nom == "Sans filtre" else nom: vue - vue.mean()
                        for nom, vue in variantes.items()}
        borne = max(1.0, *(float(np.percentile(np.abs(vue), 99)) for vue in reponses.values()))
        vues = [(nom, vue, -borne, borne) for nom, vue in reponses.items()]
    elif etape == "histogramme":
        vues = [(nom, vue, 0, 255) for nom, vue in variantes.items()]
    else:
        vues = [(nom, ~masque, 0, 1) for nom, masque in variantes.items()]

    figure, axes = plt.subplots(1, len(vues), figsize=(3.05 * len(vues), 3.45),
                                layout="constrained")
    figure.suptitle(f"{titres[etape]} · image {numero}", fontsize=14, fontweight="bold")
    for axe, (nom, vue, minimum, maximum) in zip(axes, vues, strict=True):
        axe.imshow(vue, cmap="gray", vmin=minimum, vmax=maximum, interpolation="nearest")
        axe.set_axis_off()
        axe.set_title(nom, fontsize=10)
    figure.savefig(sortie / f"{etape}.png", dpi=145)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projet", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--sortie", type=Path)
    args = parser.parse_args()
    projet = args.projet.resolve()
    sortie = args.sortie or projet / "results" / "intermediaires" / "comparaisons_filtres"
    sortie.mkdir(parents=True, exist_ok=True)
    dossier = projet / "catheter"
    reference = lire_image(dossier / "frame_201.png")
    image_211 = lire_image(dossier / "frame_211.png")
    difference_211 = reference.astype(np.float32) - image_211.astype(np.float32)
    exporter_planche(sortie, "difference", {"Soustraction I201 − Ik": difference_211},
                     211, image_211)

    image_291 = lire_image(dossier / "frame_291.png")
    for etape, variantes in comparer(reference, image_291).items():
        exporter_planche(sortie, etape, variantes, 291, image_291)


if __name__ == "__main__":
    main()
