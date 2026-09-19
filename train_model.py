"""
train_model.py
================
Ce script fait 4 choses, dans l'ordre :
  1. Charge les données des élèves (fichier CSV téléchargé sur Kaggle)
  2. Nettoie les données (valeurs manquantes, texte -> nombres)
  3. Entraîne DEUX modèles différents et compare leurs performances
  4. Sauvegarde le meilleur modèle pour pouvoir l'utiliser dans l'application web (app.py)

Comment l'utiliser :
  1. Télécharge le dataset ici : https://www.kaggle.com/datasets/lainguyn123/student-performance-factors
  2. Place le fichier CSV téléchargé dans le dossier data/ et renomme-le : StudentPerformanceFactors.csv
  3. Lance : python train_model.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

DATA_PATH = "data/StudentPerformanceFactors.csv"
MODEL_PATH = "model/modele_reussite.pkl"
ENCODERS_PATH = "model/encoders.pkl"

# ----------------------------------------------------------------------
# ÉTAPE 1 : Charger les données
# ----------------------------------------------------------------------
print("Étape 1/4 — Chargement des données...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Le fichier {DATA_PATH} est introuvable.\n"
        "-> Télécharge le dataset sur Kaggle (lien dans le README) et place le CSV dans data/"
    )

df = pd.read_csv(DATA_PATH)
print(f"   {len(df)} élèves chargés, {len(df.columns)} colonnes.")

# ----------------------------------------------------------------------
# ÉTAPE 2 : Nettoyer les données
# ----------------------------------------------------------------------
print("Étape 2/4 — Nettoyage des données...")

# On supprime les lignes où il manque une info importante
df = df.dropna()

# Les modèles de ML ne comprennent que des nombres.
# On transforme donc les colonnes texte (ex: "Low"/"Medium"/"High") en nombres.
colonnes_texte = df.select_dtypes(include=["object", "string"]).columns
encoders = {}

for colonne in colonnes_texte:
    encodeur = LabelEncoder()
    df[colonne] = encodeur.fit_transform(df[colonne])
    encoders[colonne] = encodeur  # on garde l'encodeur pour pouvoir décoder plus tard

print(f"   {len(colonnes_texte)} colonnes texte converties en nombres : {list(colonnes_texte)}")

# La colonne qu'on veut prédire
CIBLE = "Exam_Score"

X = df.drop(columns=[CIBLE])  # toutes les infos sur l'élève
y = df[CIBLE]                  # la note qu'on veut prédire

# On garde 20% des données de côté pour tester le modèle honnêtement
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   {len(X_train)} élèves pour l'entraînement, {len(X_test)} pour le test.")

# ----------------------------------------------------------------------
# ÉTAPE 3 : Entraîner deux modèles et comparer
# ----------------------------------------------------------------------
print("Étape 3/4 — Entraînement des modèles...")

modeles = {
    "Régression linéaire": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
}

resultats = {}

for nom, modele in modeles.items():
    modele.fit(X_train, y_train)
    predictions = modele.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)  # erreur moyenne, en points de note
    r2 = r2_score(y_test, predictions)               # 1.0 = parfait, 0 = nul

    resultats[nom] = {"modele": modele, "mae": mae, "r2": r2}
    print(f"   {nom:22s} -> erreur moyenne: {mae:.2f} points | score R²: {r2:.3f}")

# On choisit automatiquement le meilleur modèle (le plus petit MAE)
meilleur_nom = min(resultats, key=lambda k: resultats[k]["mae"])
meilleur_modele = resultats[meilleur_nom]["modele"]
print(f"\n   >> Meilleur modèle : {meilleur_nom}")

# ----------------------------------------------------------------------
# ÉTAPE 4 : Sauvegarder le modèle et les encodeurs
# ----------------------------------------------------------------------
print("Étape 4/4 — Sauvegarde du modèle...")

os.makedirs("model", exist_ok=True)
joblib.dump(meilleur_modele, MODEL_PATH)
joblib.dump(encoders, ENCODERS_PATH)
joblib.dump(list(X.columns), "model/colonnes.pkl")

print(f"   Modèle sauvegardé dans {MODEL_PATH}")
print("\nTerminé ! Tu peux maintenant lancer l'application web avec : python app.py")

# On écrit aussi un petit résumé texte, utile pour ton README GitHub
with open("model/resultats.txt", "w", encoding="utf-8") as f:
    f.write("Résultats de l'entraînement\n")
    f.write("===========================\n\n")
    for nom, r in resultats.items():
        f.write(f"{nom} :\n")
        f.write(f"  - Erreur moyenne (MAE) : {r['mae']:.2f} points\n")
        f.write(f"  - Score R² : {r['r2']:.3f}\n\n")
    f.write(f"Modèle retenu : {meilleur_nom}\n")
