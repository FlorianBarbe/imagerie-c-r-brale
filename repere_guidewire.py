"""Segmente le petit repère sombre du guidewire à droite dans les radiographies du TP."""

import cv2
import numpy as np


def detecter_repere_guidewire(image):
    hauteur, largeur = image.shape
    echelle = np.array([largeur, hauteur]) / 512
    (x0, y0), (x1, y1) = np.rint(np.array([(409, 322), (439, 350)]) * echelle).astype(int)
    centre_attendu = np.array([421, 336]) * echelle

    # Renforce les objets sombres localement, puis élimine les traits fins.
    contraste = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    reponse = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, contraste)[y0:y1, x0:x1]
    masque = np.zeros_like(image, dtype=bool)
    if reponse.size == 0:
        return masque

    candidats = (reponse >= max(8, 0.45 * reponse.max())).astype(np.uint8)
    noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ouverts = cv2.morphologyEx(candidats, cv2.MORPH_OPEN, noyau,
                               borderType=cv2.BORDER_CONSTANT, borderValue=0)
    nombre, etiquettes, _, centres = cv2.connectedComponentsWithStats(ouverts, connectivity=8)
    if nombre > 1:
        distances = np.linalg.norm(centres[1:] + (x0, y0) - centre_attendu, axis=1)
        masque[y0:y1, x0:x1] = etiquettes == (1 + np.argmin(distances))
    return masque
