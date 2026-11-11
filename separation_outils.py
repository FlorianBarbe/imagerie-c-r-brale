"""Séparation spatiale des instruments après segmentation du masque complet."""

import numpy as np


# Ligne d'image repérée à la jonction sur les annotations du TP (512 × 512).
LIGNE_SEPARATION_512 = 348


def ligne_separation(hauteur):
    """Renvoie l'indice de la première ligne réservée au microcathéter."""
    return round(LIGNE_SEPARATION_512 * hauteur / 512)


def separer_outils(masque):
    """Renvoie les détections du cathéter en bas et du guidewire en haut."""
    zone_catheter = np.arange(masque.shape[0])[:, None] >= ligne_separation(masque.shape[0])
    return masque & zone_catheter, masque & ~zone_catheter


def zone_sans_artefacts(forme):
    """Écarte les régions de bord observées hors des annotations de la séquence."""
    hauteur, largeur = forme
    y, x = np.ogrid[:hauteur, :largeur]
    bord_gauche = (x < round(45 * largeur / 512)) & (y < round(240 * hauteur / 512))
    # La bande du bas s’arrête à gauche du tracé descendant du microcathéter.
    bord_bas = (
        (y >= round(485 * hauteur / 512))
        & (x >= round(290 * largeur / 512))
        & (x < round(438 * largeur / 512))
    )
    bord_droit = (x >= round(456 * largeur / 512)) & (y >= ligne_separation(hauteur))
    return ~(bord_gauche | bord_bas | bord_droit)
