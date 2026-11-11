"""Exécute le TP de soustraction de fond et exporte les résultats pour le notebook."""

import argparse
from pathlib import Path

import cv2

from segmentation import lire_image, segmenter_outil
from suivi_pointe import suivre_pointe
from validation import annotation_manuelle, exporter_scores, exporter_scores_intermediaires, mesurer
from visualisation import ecrire_png, exporter_gif, planche, superposer, tracer_scores


def main():
    projet = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=projet / "catheter")
    parser.add_argument("--sortie", type=Path, default=projet / "results")
    args = parser.parse_args()

    numeros = range(211, 292, 10)
    premiere = lire_image(args.images / "frame_201.png")
    images = [lire_image(args.images / f"frame_{numero}.png") for numero in numeros]
    intermediaires = args.sortie / "intermediaires"
    validation = args.sortie / "validation"
    suivi = args.sortie / "suivi_pointe"
    for dossier in (intermediaires, validation, suivi):
        dossier.mkdir(parents=True, exist_ok=True)

    scores = []
    scores_intermediaires = []
    contraste = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for numero, image in zip(numeros, images, strict=True):
        etapes, outil = segmenter_outil(premiere, image)
        manuel = annotation_manuelle(args.images, numero)
        sad, mse, psnr = mesurer(outil, manuel)
        scores.append((numero, sad, mse, psnr))
        for nom in ("Otsu + 30", "Composantes ≥ 20 px", "Fermeture 3×3"):
            scores_intermediaires.append((numero, nom, *mesurer(etapes[nom] == 0, manuel)))
        print(f"{numero} — SAD {sad}, MSE {mse:.6g}, PSNR {psnr:.3f} dB")

        vert = superposer(contraste.apply(image), outil)
        ecrire_png(validation / f"masque_{numero}.png", (~outil).astype("uint8") * 255)
        ecrire_png(validation / f"superposition_{numero}.png", vert)
        planche(validation / f"comparaison_{numero}.png", {
            "Masque estimé": ~outil,
            "Annotation dilatée": ~manuel,
            "Superposition": cv2.cvtColor(vert, cv2.COLOR_BGR2RGB),
        }, colonnes=3)
        planche(intermediaires / f"etapes_{numero}.png", etapes, colonnes=4)

    exporter_scores_intermediaires(intermediaires, scores_intermediaires)
    exporter_scores(validation, scores)
    tracer_scores(validation / "scores.png", scores)

    superpositions = []
    for numero, image, outil in zip(numeros, images, suivre_pointe(premiere, images), strict=True):
        vert = superposer(image, outil)
        ecrire_png(suivi / f"masque_{numero}.png", (~outil).astype("uint8") * 255)
        ecrire_png(suivi / f"resultat_{numero}.png", vert)
        superpositions.append(vert)
    exporter_gif(suivi / "animation.gif", superpositions)


if __name__ == "__main__":
    main()
