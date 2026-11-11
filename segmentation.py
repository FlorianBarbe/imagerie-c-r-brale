import cv2
import numpy as np


def lire_image(chemin):
    if not chemin.is_file():
        raise FileNotFoundError(f"Image introuvable : {chemin}")
    image = cv2.imread(str(chemin), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image illisible : {chemin}")
    return image


def filtrer(reference, image):
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

    return difference, gauss, filtree


def calculer_masque(reference, image):
    """Renvoie le masque binaire utilisé pour le suivi de pointe."""
    _, _, filtree = filtrer(reference, image)
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



def segmenter_outil(reference, image):
    """Renvoie la réponse de chaque filtre et les masques de la chaîne complète."""
    difference, gauss, filtree = filtrer(reference, image)
    positive = np.maximum(filtree, 0)
    limite = np.percentile(positive, 99)
    contraste = np.zeros_like(positive, dtype=np.uint8)
    if limite > 0:
        contraste = np.rint(255 * np.clip(positive / limite, 0, 1)).astype(np.uint8)

    otsu, _ = cv2.threshold(contraste, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    outil = (contraste > min(otsu + 30, 255)).astype(np.uint8)
    _, etiquettes, stats, _ = cv2.connectedComponentsWithStats(outil, connectivity=8)
    garder = stats[:, cv2.CC_STAT_AREA] >= 20
    garder[0] = False
    composantes = garder[etiquettes].astype(np.uint8)
    noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    masque = cv2.morphologyEx(composantes, cv2.MORPH_CLOSE, noyau).astype(bool)
    return {
        "Soustraction": difference,
        "Filtrage spatial": gauss,
        "Filtrage spectral": filtree,
        "Contraste au percentile 99": contraste,
        "Otsu + 30": 1 - outil,
        "Composantes ≥ 20 px": 1 - composantes,
        "Fermeture 3×3": ~masque,
    }, masque
