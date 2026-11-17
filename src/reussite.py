import os
import pandas as pd
import matplotlib.pyplot as plt
from tools import detecteur, charger_image, extraire_vecteur, chronometrer, charger_modele
from config import RACINE, DOSSIER_IMAGES_TEST, PERSONNES_TEST,CHEMIN_MODELE

def verifier_dossier(dossier: str) -> None:
    """Vérifie que le dossier d'images existe.

    :param dossier: dossier contenant un sous-dossier par classe.
    :raises FileNotFoundError: si le dossier est absent (ou n'est pas un dossier).
    """
    if not os.path.isdir(dossier):
        raise FileNotFoundError(
            "Assurez-vous d'avoir les dossiers par classe dans ce format "
            f"à la racine du projet : asl_processed/test/ (cherché : {dossier})"
        )


def iterer_images(dossier: str):
    """Génère les couples (chemin de l'image, classe) du dossier d'images.

    :param dossier: dossier contenant un sous-dossier par classe.
    :return: générateur de tuples (chemin, classe).
    """
    for classe in sorted(os.listdir(dossier)):
        for nom in sorted(os.listdir(os.path.join(dossier, classe))):
            yield os.path.join(dossier, classe, nom), classe

def generer_graphique(donnees):
    donnees = donnees.sort_values(ascending=False, kind="stable")  # pire classe en haut
    fig, ax = plt.subplots(figsize=(8, 10))
    couleurs = ["tab:red" if taux < 50 else "tab:blue" for taux in donnees]
    barres = ax.barh(donnees.index, donnees, color=couleurs)
    ax.bar_label(barres, fmt="%.0f %%", padding=3)  # affiche aussi les 0 %
    ax.axvline(donnees.mean(), color="gray", linestyle="--",
               label=f"Moyenne des classes : {donnees.mean():.0f} %")
    ax.set_xlim(0, 110)
    ax.set_title("Taux de réussite du modèle par classe")
    ax.set_xlabel("Taux de réussite (%)")
    ax.set_ylabel("Classe")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.savefig(os.path.join(RACINE, "reussite.png"), dpi=150)
    plt.show()

@chronometrer
def main() -> None:
    verifier_dossier(DOSSIER_IMAGES_TEST)
    modele = charger_modele(CHEMIN_MODELE) # garder le retour
    lignes = [] # une entrée par image
    with detecteur() as d:
        for chemin, classe in iterer_images(DOSSIER_IMAGES_TEST):
            personne = os.path.basename(chemin).split("_")[0]
            if personne not in PERSONNES_TEST:
                continue
            vecteur = extraire_vecteur(d, charger_image(chemin, marge=True))
            if vecteur is None:
                lignes.append((classe, False)) # main non détectée = échec
                continue
            vecteur_df = pd.DataFrame([vecteur], columns=[f"{axe}{i}" for i in range(21) for axe in "xyz"], )
            prediction = modele.predict(vecteur_df)[0]
            lignes.append((classe, prediction == classe)) # (classe, False) si pas bonne prédiction, (classe, True) sinon
    df = pd.DataFrame(lignes, columns=["classe", "réussite"])
    par_classe = df.groupby("classe")["réussite"].mean() * 100 # taux par classe, en pourcentage
    generer_graphique(par_classe)
    print(f"Global : {df['réussite'].mean():.2%}")


if __name__ == "__main__":
    main()
