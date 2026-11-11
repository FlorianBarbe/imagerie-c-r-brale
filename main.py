"""Exécute le TP de soustraction de fond et exporte les résultats pour le notebook."""

import argparse
from pathlib import Path

import cv2

from segmentation import lire_image, segmenter_outil
from separation_outils import separer_outils, zone_sans_artefacts
from suivi_pointe_guidewire import suivre_pointe_guidewire
from validation import annotation_manuelle, exporter_scores, mesurer
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
    suivi = args.sortie / "suivi_pointe_guidewire"
    fusion = args.sortie / "fusion_outils"
    guidewire_rouge = args.sortie / "guidewire_rouge"
    catheter_vert = args.sortie / "catheter_vert"
    for dossier in (intermediaires, validation, suivi, fusion, guidewire_rouge, catheter_vert):
        dossier.mkdir(parents=True, exist_ok=True)

    scores = []
    superpositions_completes = []
    vues_completes = {}
    superpositions_fusion = []
    superpositions_guidewire = []
    superpositions_catheter = []
    contraste = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for numero, image in zip(numeros, images, strict=True):
        etapes, outil = segmenter_outil(premiere, image)
        outil &= zone_sans_artefacts(outil.shape)
        etapes["Exclusion des bords"] = ~outil
        manuel = annotation_manuelle(args.images, numero)
        sad, mse, psnr = mesurer(outil, manuel)
        scores.append((numero, sad, mse, psnr))
        print(f"{numero} — SAD {sad}, MSE {mse:.6g}, PSNR {psnr:.3f} dB")

        vert = superposer(contraste.apply(image), outil)
        vue = cv2.cvtColor(vert, cv2.COLOR_BGR2RGB)
        superpositions_completes.append(vert)
        vues_completes[str(numero)] = vue
        ecrire_png(validation / f"masque_{numero}.png", (~outil).astype("uint8") * 255)
        ecrire_png(validation / f"superposition_{numero}.png", vert)
        planche(validation / f"comparaison_{numero}.png", {
            "Masque estimé": ~outil,
            "Annotation dilatée": ~manuel,
            "Superposition": vue,
        }, colonnes=3)
        planche(intermediaires / f"etapes_{numero}.png", etapes, colonnes=4)

        catheter, guidewire = separer_outils(outil)
        for nom, masque in (("catheter", catheter), ("guidewire", guidewire)):
            ecrire_png(fusion / f"masque_{nom}_{numero}.png", (~masque).astype("uint8") * 255)
        combinaison = superposer(image, catheter, guidewire)
        rouge = superposer(image, repere=guidewire)
        vert = superposer(image, catheter)
        ecrire_png(fusion / f"resultat_{numero}.png", combinaison)
        ecrire_png(guidewire_rouge / f"resultat_{numero}.png", rouge)
        ecrire_png(catheter_vert / f"resultat_{numero}.png", vert)
        superpositions_fusion.append(combinaison)
        superpositions_guidewire.append(rouge)
        superpositions_catheter.append(vert)

    exporter_scores(validation, scores)
    tracer_scores(validation / "scores.png", scores)
    planche(validation / "detections_completes.png", vues_completes, colonnes=3)
    exporter_gif(validation / "animation.gif", superpositions_completes)

    exporter_gif(fusion / "animation.gif", superpositions_fusion)
    exporter_gif(guidewire_rouge / "animation.gif", superpositions_guidewire)
    exporter_gif(catheter_vert / "animation.gif", superpositions_catheter)

    superpositions = []
    for numero, image, outil in zip(numeros, images, suivre_pointe_guidewire(premiere, images), strict=True):
        rouge = superposer(image, repere=outil)
        ecrire_png(suivi / f"masque_{numero}.png", (~outil).astype("uint8") * 255)
        ecrire_png(suivi / f"resultat_{numero}.png", rouge)
        superpositions.append(rouge)
    exporter_gif(suivi / "animation.gif", superpositions)


if __name__ == "__main__":
    main()
