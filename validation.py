import csv
from pathlib import Path

import numpy as np
from skimage.morphology import dilation, disk

from segmentation import lire_image


def annotation_manuelle(dossier: Path, numero: int):
    microcatheter = lire_image(dossier / f"{numero}_MicroCath.tiff") == 0
    guide = lire_image(dossier / f"{numero}_GuideWire.tiff") == 0
    return dilation(microcatheter | guide, disk(2))


def mesurer(estimation, annotation):
    """Calcule les erreurs sur les masques binaires avec un pic égal à 1."""
    if estimation.shape != annotation.shape:
        raise ValueError("Les masques comparés doivent avoir les mêmes dimensions.")
    erreurs = estimation != annotation
    sad = int(erreurs.sum())
    mse = float(erreurs.mean())
    psnr = float(-10 * np.log10(mse)) if mse else float("inf")
    return sad, mse, psnr


def exporter_scores(dossier, scores):
    colonnes = ("SAD", "MSE", "PSNR (dB)")
    with (dossier / "scores.csv").open("w", newline="") as fichier:
        writer = csv.writer(fichier)
        writer.writerow(("Image", *colonnes))
        writer.writerows(scores)

    with (dossier / "resume.csv").open("w", newline="") as fichier:
        writer = csv.writer(fichier)
        writer.writerow(("Mesure", "Moyenne", "Écart-type (population)"))
        for nom, valeurs in zip(colonnes, np.asarray(scores)[:, 1:].T, strict=True):
            moyenne = np.mean(valeurs)
            ecart_type = np.std(valeurs) if np.isfinite(valeurs).all() else np.nan
            writer.writerow((nom, moyenne, ecart_type))
            print(f"{nom} — moyenne {moyenne:.6g}, écart-type {ecart_type:.6g}")



def exporter_scores_intermediaires(dossier, scores):
    with (dossier / "scores_etapes.csv").open("w", newline="") as fichier:
        writer = csv.writer(fichier)
        writer.writerow(("Image", "Étape", "SAD", "MSE", "PSNR (dB)"))
        writer.writerows(scores)
