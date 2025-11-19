import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------
# Page Config & Titre
# ---------------------------
st.set_page_config(page_title="CPU Load Prediction Tool", layout="wide")
st.title("💻 CPU Load Prediction Tool")

st.markdown("Bienvenue dans l'application de prédiction de charge CPU destinée aux ingénieurs SOC. Cette application vous permet de :")
st.markdown("- Explorer et visualiser les corrélations des features réseau avec la charge CPU.")
st.markdown("- Sélectionner les paramètres du modèle KNN (nombre de voisins, nombre de features à utiliser).")
st.markdown("- Evaluer le modèle via validation croisée et set de test.")
st.markdown("- Prédire la charge CPU pour de nouvelles observations réseau en temps réel.")

# ---------------------------
# Sidebar : paramètres du modèle
# ---------------------------
st.sidebar.header("⚙️ Paramètres du modèle")

k_value = st.sidebar.selectbox(
    "Nombre de voisins (k)",
    [3, 5, 7, 9, 11, 13, 15],
    index=1
)

n_features = st.sidebar.selectbox(
    "Nombre de features à sélectionner",
    [2, 3, 4, 5, 6, 7, 8],
    index=1
)

random_state = st.sidebar.number_input("Random state", value=42, step=1)

# ---------------------------
# Upload file
# ---------------------------
st.subheader("⬆️ Uploadez votre fichier Excel (.xlsx)")
st.markdown("Le fichier doit contenir une colonne 'cpu_load' et plusieurs colonnes numériques représentant les features réseau.")
uploaded_file = st.file_uploader("Choisir un fichier Excel", type=["xlsx", "xls"]) 

@st.cache_data
def load_excel(file):
    return pd.read_excel(file)

if uploaded_file is None:
    st.info("⬆️ Uploadez un fichier Excel pour commencer l'analyse.")
    st.stop()

# charge le fichier
try:
    data = load_excel(uploaded_file)
except Exception as e:
    st.error(f"Erreur lors du chargement du fichier: {e}")
    st.stop()

st.subheader("Aperçu des données")
st.dataframe(data.head())

# ---------------------------
# Pré-traitement minimal
# ---------------------------
if 'cpu_load' not in data.columns:
    st.error("Le fichier doit contenir une colonne nommée 'cpu_load'.")
    st.stop()

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if 'cpu_load' not in numeric_cols:
    st.error("La colonne 'cpu_load' doit être numérique.")
    st.stop()

data_num = data[numeric_cols].copy()
st.write(f"Colonnes numériques détectées ({len(numeric_cols)}): {numeric_cols}")

# ---------------------------
# Corrélation & visualisation
# ---------------------------
st.subheader("🔗 Matrice de corrélation")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data_num.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
st.pyplot(fig)

st.subheader("📌 Corrélations avec cpu_load")
cpu_corr = data_num.corr()['cpu_load'].sort_values(ascending=False)
st.write(cpu_corr)

st.subheader("📉 Visualisation des corrélations")
cpu_corr_viz = cpu_corr.drop('cpu_load')
colors = ['green' if x > 0 else 'red' for x in cpu_corr_viz.values]
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.barh(cpu_corr_viz.index, cpu_corr_viz.values, color=colors, alpha=0.7)
ax2.set_xlabel('Corrélation avec cpu_load')
ax2.set_title("Corrélations des variables avec cpu_load")
ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax2.grid(axis='x', alpha=0.3)
st.pyplot(fig2)

# ---------------------------
# Préparation X / y
# ---------------------------
X = data_num.drop(columns=['cpu_load'])
y = data_num['cpu_load']

if X.shape[1] < 2:
    st.error("Il faut au moins 2 features numériques en plus de 'cpu_load'.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(random_state))

# ---------------------------
# Pipeline et entraînement
# ---------------------------
base_knn = KNeighborsRegressor()
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("sfs", SequentialFeatureSelector(estimator=base_knn, n_features_to_select=min(n_features, X.shape[1]), direction="forward", scoring="r2", cv=5)),
    ("knn", KNeighborsRegressor(n_neighbors=k_value))
])

st.subheader("🔄 Entraînement + Validation croisée (CV=5)")
cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2')
pipe.fit(X_train, y_train)
model = pipe
st.write(f"Scores R² par fold: {cv_scores}")
st.write(f"Score moyen R²: **{cv_scores.mean():.4f}** ± {cv_scores.std():.4f}")

# ---------------------------
# Évaluation test
# ---------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
st.write(f"R² sur le set de test: **{r2:.4f}**")
st.write(f"RMSE sur le set de test: **{rmse:.4f}**")

# Features sélectionnées
poly = model.named_steps["poly"]
poly_feature_names = poly.get_feature_names_out(X.columns)
sfs = model.named_steps["sfs"]
selected_features = poly_feature_names[sfs.get_support()].tolist()
st.write("### Features sélectionnées :")
st.write(selected_features)

# ---------------------------
# Visualisation des prédictions
# ---------------------------
st.subheader("🔍 Prédictions vs Réel (test)")
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.scatter(y_test, y_pred, alpha=0.6)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--')
ax3.set_xlabel('Valeurs réelles')
ax3.set_ylabel('Prédictions')
ax3.set_title('Prédictions vs Réel')
st.pyplot(fig3)

# ---------------------------
# Téléchargement des résultats
# ---------------------------
results_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
@st.cache_data
def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')
csv = to_csv(results_df)
st.download_button("⬇️ Télécharger les prédictions (CSV)", data=csv, file_name='predictions.csv', mime='text/csv')

# ---------------------------
# Prédiction d'un nouveau cas
# ---------------------------
st.header("4️⃣ Prédiction d'un nouveau cas")
st.write("Entrez les valeurs pour créer une nouvelle observation et prédire le CPU Load.")

new_inputs = {}
for col in X.columns:
    if data[col].dtype in ["float64", "int64"]:
        min_val = float(data[col].min())
        max_val = float(data[col].max())
        default_val = float(data[col].mean())
        new_inputs[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=default_val)
    else:
        new_inputs[col] = st.text_input(f"{col}")

new_obs_df = pd.DataFrame([new_inputs])
new_pred = model.predict(new_obs_df)[0]
st.subheader("🔮 CPU Load prédit :")
st.metric(label="CPU Load (%)", value=f"{new_pred:.2f}%")

st.info("Application prête — modifiez k, le nombre de features, ou les valeurs de nouvelles observations pour tester les prédictions.")