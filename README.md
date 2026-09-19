<img width="988" height="840" alt="Capture d&#39;écran 2026-09-19 183354" src="https://github.com/user-attachments/assets/fc92a48a-e273-4f6d-b791-309e6c2f3574" />
<img width="918" height="908" alt="Capture d&#39;écran 2026-09-19 183350" src="https://github.com/user-attachments/assets/f3a7ceab-f6c8-4a8c-8826-3a3a64e39ded" />
<img width="907" height="876" alt="Capture d&#39;écran 2026-09-19 183344" src="https://github.com/user-attachments/assets/01270dec-f1e4-45bb-bdbd-6b2fa6952297" />
# 🎓 Prédicteur de réussite scolaire

Projet de Machine Learning réalisé dans le cadre de ma préparation à l'admission en école d'ingénieur (domaine IA2R).

## Pourquoi ce projet ?

J'ai donné des cours de soutien scolaire, ce qui m'a fait me demander : **quels facteurs influencent réellement la réussite d'un élève à un examen ?** Est-ce vraiment le nombre d'heures d'étude qui compte le plus, ou d'autres éléments (sommeil, motivation, tutorat) jouent-ils un rôle aussi important ?

Ce projet utilise un modèle de Machine Learning pour prédire la note d'un élève à partir de facteurs personnels, familiaux et scolaires — et une petite interface web pour tester le modèle facilement.

## Le dataset

Dataset **"Student Performance Factors"** (6 607 élèves, 20 colonnes) — disponible librement sur Kaggle :
🔗 https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

## Comment ça marche

Le projet se déroule en 2 étapes :

1. **`train_model.py`** — charge les données, les nettoie, entraîne deux modèles (régression linéaire et forêt aléatoire), et garde le meilleur.
2. **`app.py`** — une application web (Flask) où on remplit un formulaire avec le profil d'un élève, et qui affiche la note prédite.

## Installation et utilisation

```bash
# 1. Installer les librairies nécessaires
pip install -r requirements.txt

# 2. Télécharger le dataset sur Kaggle (lien ci-dessus)
#    et placer le fichier CSV dans le dossier data/
#    en le renommant : StudentPerformanceFactors.csv

# 3. Entraîner le modèle
python train_model.py

# 4. Lancer l'application web
python app.py

# 5. Ouvrir dans un navigateur : http://127.0.0.1:5000
```

## Résultats

*(à compléter automatiquement après l'exécution de `train_model.py` — les chiffres apparaissent dans `model/resultats.txt`)*

- Régression linéaire : erreur moyenne de 1.06 points, score R² de 0.664
- Random Forest : erreur moyenne de 1.17 points, score R² de 0.623

## Ce que j'ai appris

- Comment nettoyer un jeu de données réel (valeurs manquantes, encodage de variables catégorielles)
- La différence entre un modèle linéaire simple et un modèle plus complexe (Random Forest), et pourquoi le second capture souvent mieux les interactions entre variables
- Comment relier un modèle de Machine Learning à une interface utilisable (Flask)

## Limites et pistes d'amélioration

- Le dataset est basé sur des données déclaratives (auto-évaluées par les élèves), ce qui peut introduire des biais
- On pourrait tester d'autres modèles (Gradient Boosting, XGBoost) ou faire de l'optimisation d'hyperparamètres
- Une analyse plus poussée de l'importance de chaque variable (feature importance) permettrait d'identifier plus précisément les leviers d'action pour aider un élève en difficulté

## Technologies utilisées

Python · pandas · scikit-learn · Flask · HTML/CSS
