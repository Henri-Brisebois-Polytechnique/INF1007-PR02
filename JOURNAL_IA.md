
|Outil|Prompt|Réponse reçue|Vérification faite|Décision|
| --- | --- | --- | --- | --- |
| Claude Code | Dans extraction.py, MediaPipe ne détecte aucune main sur environ 8 % des images du dataset (souvent quand la main est collée au bord ou coupée par le cadre). Comment améliorer le taux de détection sans fausser les coordonnées ? | Voir réponse 1 | Testé l'extraction à nouveau, observé un rendement de 99 % (vs 92 %). Puisque le dataset d'images a déjà été traité, on peut être sûrs que chaque image contient une main; un pourcentage plus élevé est donc normal. | Gardé |


# RÉPONSES COMPLÈTES

## Réponse 1
Enjeu : MediaPipe ne détecte aucune main sur environ 8 % des images du dataset. Ces images sont perdues pour l'entraînement.

Cause probable : dans ces images, la main est en gros plan, souvent coupée par le cadre ou collée au bord. Le HandLandmarker cherche d'abord une paume entière avec un détecteur de paume. Quand la paume touche le bord, ce détecteur échoue, et les 21 points de repère ne sont jamais calculés.

Solution : ajouter une marge autour de l'image avant la détection. La main se retrouve alors plus loin des bords et paraît un peu plus petite, ce qui correspond mieux aux images sur lesquelles le détecteur a été entraîné.

MARGE = 0.25  # fraction du plus grand côté ajoutée de chaque côté

def charger_image(chemin: str) -> mp.Image:
    """Charge une image et l'entoure d'une bordure (répétition des bords).

    :param chemin: chemin de l'image.
    :return: image MediaPipe.
    """
    pixels = mp.Image.create_from_file(chemin).numpy_view()[..., :3]
    p = int(max(pixels.shape[:2]) * MARGE)
    pixels = np.pad(pixels, ((p, p), (p, p), (0, 0)), mode="edge")
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(pixels))

Ensuite, dans la boucle, on appelle detecteur.detect(charger_image(chemin)) au lieu de mp.Image.create_from_file(chemin).

Choix techniques :
- mode="edge" plutôt que du noir : une bordure noire crée une arête nette qui peut tromper le détecteur. Répéter les pixels du bord prolonge le fond et le poignet de façon naturelle.
- Marge proportionnelle : la même valeur fonctionne quelle que soit la résolution des images.
- [..., :3] : enlève le canal alpha des PNG, car le format SRGB attend 3 canaux.
- ascontiguousarray : mp.Image refuse un tableau qui n'est pas contigu en découpage des canaux peuvent produire.

Pas d'impact sur les données : les coordonnées sont calculées par rapportnormaliser() centre les points sur le poignet et divise par une seulevaleur d'échelle, le décalage et le zoom disparaissent. La seule limite concerne les images non carrées : le rapport entre x et y change un peu. Ce n'est pas grave si toutes les
images ont le même format.

Comment choisir la valeur de la marge : il faut la mesurer plutôt que la traction sur un échantillon d'environ 500 images avec MARGE à 0, 0.1, 0.25et 0.5, puis on compare nb_ok / total. Trop peu de marge ne règle pas le problème. Trop de marge rend la main trop petite, et la détection recommence à baisser.

Si ça ne suffit pas, d'autres pistes :
- baisser min_hand_detection_confidence (par exemple à 0.3), en acceptant ;
- refaire une tentative avec l'image retournée horizontalement (np.fliplr) quand la première détection échoue ;
- redimensionner les images trop grandes ou trop petites avant de les pas
