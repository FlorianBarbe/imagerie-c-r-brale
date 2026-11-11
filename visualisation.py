import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def ecrire_png(chemin, image):
    if not cv2.imwrite(str(chemin), image):
        raise OSError(f"Écriture impossible : {chemin}")


def superposer(image, outil=None, repere=None):
    couleur = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if repere is not None:
        couleur[repere] = (0, 0, 255)
    if outil is not None:
        couleur[outil] = (0, 255, 0)
    return couleur


def planche(chemin, images, colonnes):
    lignes = (len(images) + colonnes - 1) // colonnes
    figure, axes = plt.subplots(lignes, colonnes, figsize=(3.2 * colonnes, 3.2 * lignes),
                                squeeze=False, layout="constrained")
    for axe in axes.flat:
        axe.axis("off")
    for axe, (titre, image) in zip(axes.flat, images.items()):
        options = {"cmap": "gray"} if image.ndim == 2 else {}
        if image.dtype == np.bool_ or (image.ndim == 2 and image.dtype == np.uint8 and image.max() <= 1):
            options.update(vmin=0, vmax=1)
        axe.imshow(image, interpolation="nearest", **options)
        axe.set_title(titre)
    figure.savefig(chemin, dpi=150)
    plt.close(figure)


def tracer_scores(chemin, scores):
    donnees = np.asarray(scores)
    figure, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, layout="constrained")
    for indice, (axe, nom) in enumerate(zip(axes, ("SAD", "MSE", "PSNR (dB)"), strict=True), 1):
        valeurs = donnees[:, indice]
        axe.plot(donnees[:, 0], np.where(np.isfinite(valeurs), valeurs, np.nan), "o-")
        for numero in donnees[np.isposinf(valeurs), 0]:
            axe.annotate("∞", (numero, 0.9), xycoords=("data", "axes fraction"), ha="center")
        axe.set_ylabel(nom)
        axe.grid(alpha=0.25)
    axes[-1].set_xticks(donnees[:, 0].astype(int))
    axes[-1].set_xlabel("Numéro d’image")
    figure.savefig(chemin, dpi=150)
    plt.close(figure)


def exporter_gif(chemin, images):
    frames = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]
    frames[0].save(chemin, save_all=True, append_images=frames[1:], duration=1000, loop=0)
