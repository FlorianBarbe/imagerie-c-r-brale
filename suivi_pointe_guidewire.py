"""Isole les nouvelles détections dans la zone du fil-guide."""

import cv2
import numpy as np

from segmentation import calculer_masque
from separation_outils import ligne_separation, zone_sans_artefacts


def suivre_pointe_guidewire(reference, images):
    """Produit un masque des nouvelles détections du fil-guide à chaque image."""
    y, x = np.ogrid[-2:3, -2:3]
    noyau_erosion = (
        (abs(x) + abs(y) <= 2) | ((abs(x) == 2) & (x == -2 * y))
    ).astype(np.uint8)
    noyau_dilatation = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    hauteur = reference.shape[0]
    zone_guidewire = (
        (np.arange(hauteur)[:, None] < ligne_separation(hauteur))
        & zone_sans_artefacts(reference.shape)
    )
    deja_detecte = np.zeros(reference.shape, dtype=bool)
    for image in images:
        detection = (calculer_masque(reference, image) == 0) & zone_guidewire
        nouveaux_pixels = detection & ~deja_detecte
        erode = cv2.erode(
            nouveaux_pixels.astype(np.uint8), noyau_erosion,
            borderType=cv2.BORDER_CONSTANT, borderValue=0,
        )
        dilate = cv2.dilate(
            erode, noyau_dilatation, borderType=cv2.BORDER_CONSTANT, borderValue=0,
        )
        deja_detecte |= nouveaux_pixels
        reference = image
        yield dilate.astype(bool) & zone_guidewire
