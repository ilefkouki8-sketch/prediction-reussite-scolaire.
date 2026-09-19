"""
app.py
======
La petite application web. Elle affiche un formulaire, récupère les infos
saisies par l'utilisateur, les envoie au modèle entraîné (train_model.py),
et affiche la prédiction.

Comment l'utiliser :
  1. D'abord lancer : python train_model.py  (une seule fois)
  2. Puis lancer     : python app.py
  3. Ouvrir dans un navigateur : http://127.0.0.1:5000
"""

from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "model/modele_reussite.pkl"

# Les champs du formulaire : (nom_technique, label affiché, type)
# type "nombre" -> champ numérique / type "choix" -> liste déroulante avec les options
CHAMPS = [
    ("Hours_Studied", "Heures d'étude par semaine", "nombre"),
    ("Attendance", "Taux de présence en cours (%)", "nombre"),
    ("Parental_Involvement", "Implication des parents", "choix", ["Low", "Medium", "High"]),
    ("Access_to_Resources", "Accès aux ressources pédagogiques", "choix", ["Low", "Medium", "High"]),
    ("Extracurricular_Activities", "Activités extra-scolaires", "choix", ["Yes", "No"]),
    ("Sleep_Hours", "Heures de sommeil par nuit", "nombre"),
    ("Previous_Scores", "Note obtenue à l'examen précédent", "nombre"),
    ("Motivation_Level", "Niveau de motivation", "choix", ["Low", "Medium", "High"]),
    ("Internet_Access", "Accès à internet", "choix", ["Yes", "No"]),
    ("Tutoring_Sessions", "Séances de tutorat par mois", "nombre"),
    ("Family_Income", "Revenu familial", "choix", ["Low", "Medium", "High"]),
    ("Teacher_Quality", "Qualité perçue des enseignants", "choix", ["Low", "Medium", "High"]),
    ("School_Type", "Type d'établissement", "choix", ["Public", "Private"]),
    ("Peer_Influence", "Influence des camarades", "choix", ["Positive", "Neutral", "Negative"]),
    ("Physical_Activity", "Heures d'activité physique par semaine", "nombre"),
    ("Learning_Disabilities", "Troubles de l'apprentissage", "choix", ["Yes", "No"]),
    ("Parental_Education_Level", "Niveau d'étude des parents", "choix", ["High School", "College", "Postgraduate"]),
    ("Distance_from_Home", "Distance domicile-école", "choix", ["Near", "Moderate", "Far"]),
    ("Gender", "Genre", "choix", ["Male", "Female"]),
]


def charger_modele():
    """Charge le modèle entraîné et les outils d'encodage, s'ils existent."""
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    modele = joblib.load(MODEL_PATH)
    encoders = joblib.load("model/encoders.pkl")
    colonnes = joblib.load("model/colonnes.pkl")
    return modele, encoders, colonnes


@app.route("/", methods=["GET", "POST"])
def accueil():
    modele, encoders, colonnes = charger_modele()
    prediction = None
    erreur = None

    if modele is None:
        erreur = "Le modèle n'a pas encore été entraîné. Lance d'abord : python train_model.py"

    if request.method == "POST" and modele is not None:
        try:
            # On récupère les valeurs saisies dans le formulaire
            donnees = {}
            for champ in CHAMPS:
                nom_technique = champ[0]
                type_champ = champ[2]
                valeur = request.form.get(nom_technique)

                if type_champ == "nombre":
                    donnees[nom_technique] = float(valeur)
                else:
                    # On encode le texte (ex: "High" -> 2) avec le même encodeur
                    # que celui utilisé pendant l'entraînement
                    encodeur = encoders[nom_technique]
                    donnees[nom_technique] = encodeur.transform([valeur])[0]

            # On remet les colonnes dans le même ordre que pendant l'entraînement
            ligne = pd.DataFrame([donnees])[colonnes]
            resultat = modele.predict(ligne)[0]
            prediction = round(resultat, 1)

        except Exception as e:
            erreur = f"Une erreur est survenue : {e}"

    return render_template("index.html", champs=CHAMPS, prediction=prediction, erreur=erreur)


if __name__ == "__main__":
    app.run(debug=True)
