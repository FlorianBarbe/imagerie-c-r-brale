"""Segmente les instruments et lance la validation contre les TIFF."""

import argparse
from pathlib import Path

from segmentation import lire_image, segmenter_outil
from separation_outils import separer_outils, zone_sans_artefacts
from validation_par_outil import valider
from visualisation import ecrire_png


def main():
    projet = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=projet / "catheter")
    parser.add_argument("--sortie", type=Path, default=projet / "results")
    args = parser.parse_args()

    reference = lire_image(args.images / "frame_201.png")
    fusion = args.sortie / "fusion_outils"
    fusion.mkdir(parents=True, exist_ok=True)
    for numero in range(211, 292, 10):
        image = lire_image(args.images / f"frame_{numero}.png")
        _, masque = segmenter_outil(reference, image)
        masque &= zone_sans_artefacts(masque.shape)
        catheter, guidewire = separer_outils(masque)
        for nom, outil in (("catheter", catheter), ("guidewire", guidewire)):
            ecrire_png(fusion / f"masque_{nom}_{numero}.png",
                       (~outil).astype("uint8") * 255)

    valider(args.images, args.sortie, args.sortie / "validation_par_outil")


if __name__ == "__main__":
    main()
