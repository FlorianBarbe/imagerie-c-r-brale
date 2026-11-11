import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as io
import skimage.morphology as morph
import numpy.random
import time
import cv2
from pathlib import Path

print("Hello world")

IMDIR = Path(__file__).resolve().parent / "catheter"


# #Affichage de toutes les images
# im_counter=1
# for root, dirnames, filenames in os.walk(IMDIR):
#     for filename in filenames:
#         f=os.path.join(IMDIR, filename)
#         if f.endswith(".png"):
#             img=cv2.imread(f, cv2.IMREAD_GRAYSCALE)
#             plt.subplot(2,5,im_counter)
#             plt.imshow(img, cmap='gray')
#             plt.grid(False)
#             plt.axis('off')
#             im_counter += 1
# plt.show()
#on ouvre les images et on les convertie en floats pour faire l'opération de soustraction

#prise de la première imgae comme référence
ref=cv2.imread(str(IMDIR/"frame_201.png"), cv2.IMREAD_GRAYSCALE)
ref=np.float32(ref)


img=cv2.imread(str(IMDIR/"frame_291.png"), cv2.IMREAD_GRAYSCALE)
img=np.float32(img)
#on la soustrait à sa moyenne
img=ref-img
#on ramène l'image sur 8 bits
img=cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)



#première étape, on teste un flou pour enlever du bruit
gauss=cv2.GaussianBlur(img,(5,5),1)
gauss=cv2.normalize(gauss, None, 0, 255, cv2.NORM_MINMAX)

median=cv2.medianBlur(img,5)
bilateral = cv2.bilateralFilter(img, 5, 50, 50)  
plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("originale")
plt.subplot(2,2,2)
plt.imshow(gauss, cmap='gray')
plt.title("filtre gaussien")
plt.subplot(2,2,3)
plt.imshow(median, cmap='gray')
plt.title("filtre médian")
plt.subplot(2,2,4)
plt.imshow(bilateral, cmap='gray')
plt.title("filtre bilatéral")
plt.show()

img=gauss



#deuxième étape, on filtre dans le domaine de Fourier
entree = img.astype(np.float32)  # Mets gauss ici pour filtrer après le gaussien
spectre = cv2.dft(entree, flags=cv2.DFT_COMPLEX_OUTPUT)
spectre = np.fft.fftshift(spectre, axes=(0, 1))

h, w = entree.shape
y, x = np.ogrid[:h, :w]
distance2 = (y - h // 2)**2 + (x - w // 2)**2

passe_haut_gaussien = 1 - np.exp(-distance2 / (2 * 10**2))
passe_bas_gaussien = np.exp(-distance2 / (2 * 60**2))

D0, n = 12, 2
passe_haut_butterworth = 1 - 1 / (1 + (distance2 / D0**2)**n)

filtres = {
    "Passe-haut gaussien": passe_haut_gaussien,
    "Passe-haut Butterworth": passe_haut_butterworth,
    "Passe-bande gaussien": passe_haut_gaussien * passe_bas_gaussien,
}

# On centre la référence pour l'affichage, comme les filtres retirent la moyenne.
resultats = {"Sans filtre spectral (centrée)": entree - entree.mean()}

for nom, H in filtres.items():
    spectre_filtre = spectre * H.astype(np.float32)[:, :, None]
    spectre_filtre = np.fft.ifftshift(spectre_filtre, axes=(0, 1))
    resultats[nom] = cv2.idft(
        spectre_filtre, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT
    )

# Échelle commune autour de zéro, sans laisser les valeurs extrêmes
# écraser le contraste.
limite = max(np.percentile(np.abs(image), 99) for image in resultats.values())

fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
for ax, (nom, image) in zip(axes.flat, resultats.items()):
    ax.imshow(image, cmap="gray", vmin=-limite, vmax=limite)
    ax.set_title(nom)
    ax.axis("off")

plt.show()


img = np.float32(resultats["Passe-bande gaussien"])
positive=np.maximum(img,0)

# Le percentile évite qu'un seul pixel très fort fixe toute l'échelle.
limite = np.percentile(positive, 99)
contraste = np.zeros_like(positive, dtype=np.uint8)

if limite > 0:
    contraste = np.rint(255 * np.clip(positive / limite, 0, 1)).astype(np.uint8)



# Figure 1 : les images intermédiaires avant le seuillage
fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

limite_affichage = max(float(np.percentile(np.abs(img), 99)), 1e-6)
axes[0].imshow(img, cmap="gray", vmin=-limite_affichage, vmax=limite_affichage)
axes[0].set_title("Sortie du passe-bande")

axes[1].imshow(positive, cmap="gray", vmin=0, vmax=limite_affichage)
axes[1].set_title("Réponses positives")

axes[2].imshow(contraste, cmap="gray", vmin=0, vmax=255)
axes[2].set_title("Contraste avant seuillage")

for ax in axes:
    ax.axis("off")

# Figure 2 : un seuil par ligne, un traitement par colonne
seuil_otsu, _ = cv2.threshold(
    contraste, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU
)

deltas = (0, 5, 10, 15, 20, 30)
aire_min = 20  # Aire minimale d'une composante, en pixels
noyau3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
noyau5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

fig, axes = plt.subplots(
    len(deltas), 5, figsize=(15, 2.6 * len(deltas)), constrained_layout=True
)

for ligne, delta in enumerate(deltas):
    seuil_test = min(seuil_otsu + delta, 255)
    outil = (contraste > seuil_test).astype(np.uint8)

    fermeture3 = cv2.morphologyEx(outil, cv2.MORPH_CLOSE, noyau3)
    fermeture5 = cv2.morphologyEx(outil, cv2.MORPH_CLOSE, noyau5)

    # Chaque groupe de pixels connectés reçoit une étiquette.
    # L'étiquette 0 correspond au fond.
    _, etiquettes, stats, _ = cv2.connectedComponentsWithStats(
        outil, connectivity=8
    )
    garder = stats[:, cv2.CC_STAT_AREA] >= aire_min
    garder[0] = False
    sans_petits = garder[etiquettes].astype(np.uint8)

    sans_petits_ferme = cv2.morphologyEx(sans_petits, cv2.MORPH_CLOSE, noyau3)

    variantes = {
        "Seuil seul": outil,
        "Fermeture 3×3": fermeture3,
        "Fermeture 5×5": fermeture5,
        f"Composantes ≥ {aire_min} px": sans_petits,
        "Composantes + fermeture 3×3": sans_petits_ferme,
    }

    for ax, (nom, detection) in zip(axes[ligne], variantes.items()):
        masque = 1 - detection  # Noir = outil, blanc = fond
        ax.imshow(masque, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{nom}\nSeuil {seuil_test:.0f} (Otsu + {delta})", fontsize=9)
        ax.axis("off")

plt.show()


from skimage.morphology import dilation, disk

seuil_otsu, _ = cv2.threshold(
    contraste, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU
)
noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

for ax, delta in zip(axes[:2], (20, 30)):
    seuil_test = min(seuil_otsu + delta, 255)
    outil = (contraste > seuil_test).astype(np.uint8)

    # Retire les composantes de moins de 20 pixels.
    _, etiquettes, stats, _ = cv2.connectedComponentsWithStats(
        outil, connectivity=8
    )
    garder = stats[:, cv2.CC_STAT_AREA] >= 20
    garder[0] = False  # L'étiquette 0 correspond au fond.
    outil = garder[etiquettes].astype(np.uint8)

    outil = cv2.morphologyEx(outil, cv2.MORPH_CLOSE, noyau)
    ax.imshow(1 - outil, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"Otsu + {delta} (seuil {seuil_test:.0f})")

microcath = cv2.imread(str(IMDIR / "291_MicroCath.tiff"), cv2.IMREAD_GRAYSCALE)
guidewire = cv2.imread(str(IMDIR / "291_GuideWire.tiff"), cv2.IMREAD_GRAYSCALE)

outil_manuel = (microcath == 0) | (guidewire == 0)
outil_manuel = dilation(outil_manuel, disk(2))
masque_manuel = (~outil_manuel).astype(np.uint8)

axes[2].imshow(masque_manuel, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
axes[2].set_title("Annotation manuelle 291")

for ax in axes:
    ax.axis("off")
    # Décommente ces lignes pour zoomer sur l'arc et la boucle.
    # ax.set_xlim(60, 500)
    # ax.set_ylim(370, 180)

plt.show()

def retirer_petites_composantes(outil, aire_min=20):
    _, etiquettes, stats, _ = cv2.connectedComponentsWithStats(
        outil, connectivity=8
    )
    garder = stats[:, cv2.CC_STAT_AREA] >= aire_min
    garder[0] = False  # L'étiquette 0 correspond au fond.
    return garder[etiquettes].astype(np.uint8)


seuil_test = min(seuil_otsu + 30, 255)
outil_brut = (contraste > seuil_test).astype(np.uint8)

# Ordre actuel : on retire les petits groupes, puis on raccorde.
outil_sans_petits = retirer_petites_composantes(outil_brut)
outil_ordre_actuel = cv2.morphologyEx(outil_sans_petits, cv2.MORPH_CLOSE, noyau)

# Nouvel ordre : on raccorde d'abord, puis on retire les petits groupes restants.
outil_ferme = cv2.morphologyEx(outil_brut, cv2.MORPH_CLOSE, noyau)
outil_nouvel_ordre = retirer_petites_composantes(outil_ferme)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

comparaison = [
    ("Composantes puis fermeture", 1 - outil_ordre_actuel),
    ("Fermeture puis composantes", 1 - outil_nouvel_ordre),
    ("Annotation manuelle 291", masque_manuel),
]

for ax, (titre, masque) in zip(axes, comparaison):
    ax.imshow(masque, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(titre)
    ax.axis("off")

plt.show()

outil_reference = masque_manuel == 0

for nom, detection in (
    ("Composantes puis fermeture", outil_ordre_actuel),
    ("Fermeture puis composantes", outil_nouvel_ordre),
):
    outil_detecte = detection == 1
    faux_positifs = np.count_nonzero(outil_detecte & ~outil_reference)
    faux_negatifs = np.count_nonzero(~outil_detecte & outil_reference)

    print(f"{nom} — SAD : {faux_positifs + faux_negatifs}")
    print(f"  Fond détecté à tort : {faux_positifs} pixels")
    print(f"  Outil manqué : {faux_negatifs} pixels")


