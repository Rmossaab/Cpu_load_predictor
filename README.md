\# 💻 CPU Load Prediction Tool  

\*\*Streamlit App for SOC Engineers — Machine Learning–based CPU Load Prediction\*\*



Bienvenue dans cette application Streamlit permettant de \*\*prédire la charge CPU\*\* à partir de données réseau (connexions, volume, pertes de paquets, alertes IDS, etc.).  

Elle est conçue pour les \*\*ingénieurs SOC\*\*, analystes et étudiants souhaitant explorer la relation entre trafic réseau et charge CPU d’un serveur/système.



---



\## 🚀 Fonctionnalités principales



\### 📊 1. Exploration des données

\- Chargement d’un fichier Excel (`.xlsx` / `.xls`)

\- Détection automatique des colonnes numériques

\- Matrice de corrélation (heatmap)

\- Graphique horizontal des corrélations avec `cpu\_load`

\- Aperçu des données



\### ⚙️ 2. Configuration du modèle (sidebar)

\- Choix du nombre de voisins \*\*K\*\* pour KNN  

\- Choix du nombre de features à sélectionner (via \*\*SequentialFeatureSelector\*\*)  

\- Choix du `random\_state`



\### 🤖 3. Modélisation

Pipeline ML complet :

`StandardScaler → PolynomialFeatures → SequentialFeatureSelector → KNN Regressor`



\- Validation croisée \*\*CV = 5\*\*

\- Affichage :

&nbsp; - Scores R² par fold

&nbsp; - Score moyen + écart-type

\- Entraînement final du modèle



\### 🧪 4. Évaluation

\- Score \*\*R²\*\* sur le set de test

\- \*\*RMSE\*\*

\- Features polynomiales sélectionnées

\- Plot \*Predictions vs Real values\*



\### 📥 5. Téléchargement

\- Export des prédictions (CSV)



\### 🔮 6. Prédiction d’un nouveau cas

\- Widgets dynamiques générés automatiquement selon les colonnes du dataset

\- Prédiction du `cpu\_load` sous forme de jauge (`st.metric`)



---



\## 📁 Exemple de structure du dataset



Le fichier Excel doit contenir au minimum :



| feature1 | feature2 | ... | cpu\_load |

|----------|-----------|-----|----------|

| numeric  | numeric   | ... | numeric  |



Exemples de features courantes :

\- `tcp\_connections`

\- `data\_volume\_MB`

\- `packet\_loss\_rate`

\- `ids\_alerts`

\- etc.



---



\## 🧑‍💻 Installation \& Lancement



\### 1️⃣ Cloner le repo

```bash

git clone https://github.com/username/cpu-load-prediction-tool.git

cd cpu-load-prediction-tool



