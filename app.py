# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Configuration de la page
st.set_page_config(
    page_title="Prévision de la demande - Dashboard",
    page_icon="📊",
    layout="wide"
)
pd.set_option('display.float_format', '{:.2f}'.format)
st.title("📊 Prévision de la demande - Tableau de bord décisionnel")
st.markdown("---")

# ============================================================
# TRAITEMENT DE DONNEES
# ============================================================

@st.cache_data
def load_and_process_data(uploaded_file):
    """Charge et agrège toutes les tables"""
    ventes = pd.read_excel(uploaded_file, sheet_name="Sales")
    production = pd.read_excel(uploaded_file, sheet_name="Production")
    inventaire = pd.read_excel(uploaded_file, sheet_name="Inventory")
    marche = pd.read_excel(uploaded_file, sheet_name="Market")
    marketing = pd.read_excel(uploaded_file, sheet_name="Marketing_Expenses")
    
    # Agrégation ventes
    ventes_agg = (ventes
        .groupby(["SIM_ROUND", "SALES_ORGANIZATION", "MATERIAL_CODE", 
                  "DISTRIBUTION_CHANNEL", "AREA", "NET_PRICE", 
                  "NET_VALUE", "COST", "QUANTITY_DELIVERED", 
                  "CONTRIBUTION_MARGIN", "MATERIAL_DESCRIPTION"],
                 as_index=False)["QUANTITY"]
        .sum()
        .rename(columns={
            "SIM_ROUND": "QUART_SIMULATION",
            "SALES_ORGANIZATION": "ENTREPRISE",
            "MATERIAL_CODE": "CODE_PRODUIT",
            "DISTRIBUTION_CHANNEL": "CANAL_DISTRIBUTION",
            "AREA": "ZONE_GEOGRAPHIQUE",
            "NET_PRICE": "PRIX_NET",
            "NET_VALUE": "VENTES_NETTES",
            "COST": "COUT",
            "CONTRIBUTION_MARGIN": "MARGE_CONTRIBUTIVE",
            "MATERIAL_DESCRIPTION": "NOM_PRODUIT",
            "QUANTITY": "DEMANDE"
        })
    )
    
    # Agrégation production
    production_agg = (production
        .groupby(["COMPANY_CODE", "SIM_ROUND", "MATERIAL_CODE"], as_index=False)["YIELD"]
        .sum()
        .rename(columns={
            "COMPANY_CODE": "ENTREPRISE",
            "SIM_ROUND": "QUART_SIMULATION",
            "MATERIAL_CODE": "CODE_PRODUIT",
            "YIELD": "PRODUCTION_TOTALE"
        })
    )
    
    # Agrégation marché
    marche_agg = (marche
        .groupby(["COMPANY_CODE", "SIM_ROUND", "DISTRIBUTION_CHANNEL"], as_index=False)["AVERAGE_PRICE"]
        .mean()
        .rename(columns={
            "COMPANY_CODE": "ENTREPRISE",
            "SIM_ROUND": "QUART_SIMULATION",
            "DISTRIBUTION_CHANNEL": "CANAL_DISTRIBUTION",
            "AVERAGE_PRICE": "PRIX_MOYEN"
        })
    )
    
    # Agrégation marketing
    marketing_agg = (marketing
        .groupby(["SIM_ROUND", "SALES_ORGANIZATION"], as_index=False)["AMOUNT"]
        .sum()
        .rename(columns={
            "SIM_ROUND": "QUART_SIMULATION",
            "SALES_ORGANIZATION": "ENTREPRISE",
            "AMOUNT": "DEPENSES_MARKETING"
        })
    )
    
    # Agrégation inventaire
    inventaire_agg = (inventaire
        .groupby(["SIM_ROUND"], as_index=False)["INVENTORY_OPENING_BALANCE"]
        .mean()
        .rename(columns={
            "SIM_ROUND": "QUART_SIMULATION",
            "INVENTORY_OPENING_BALANCE": "STOCK_INITIAL"
        })
    )
    
    # Fusion
    df = ventes_agg.merge(production_agg, on=["ENTREPRISE", "QUART_SIMULATION", "CODE_PRODUIT"], how="left")
    df = df.merge(marketing_agg, on=["QUART_SIMULATION", "ENTREPRISE"], how="left")
    df = df.merge(marche_agg, on=["QUART_SIMULATION", "CANAL_DISTRIBUTION", "ENTREPRISE"], how="left")
    df = df.merge(inventaire_agg, on="QUART_SIMULATION", how="left")
    
    return df


@st.cache_data
def prepare_company_data(df, entreprise):
    """Prépare les données pour une entreprise spécifique"""
    df_company = df[df["ENTREPRISE"] == entreprise].copy()
    df_company["CANAL_DISTRIBUTION"] = df_company["CANAL_DISTRIBUTION"].astype("object")
    
    # Création des features
    groupes = ["ENTREPRISE", "CODE_PRODUIT"]
    df_company["DEMANDE_LAG1"] = df_company.groupby(groupes, observed=True)["DEMANDE"].shift(1)
    df_company["PRIX_MOYEN_LAG1"] = df_company.groupby(groupes, observed=True)["PRIX_MOYEN"].shift(1)
    df_company["MARKETING_LAG1"] = df_company.groupby(groupes, observed=True)["DEPENSES_MARKETING"].shift(1)
    df_company["TREND"] = df_company["QUART_SIMULATION"]
    
    # Suppression des NaN
    df_company = df_company.dropna(subset=["DEMANDE_LAG1", "PRIX_MOYEN_LAG1", "MARKETING_LAG1"]).copy()
    
    # Recalculer le stock réel par produit
    df_company = calculate_real_stock_by_product(df_company)
    
    return df_company


def calculate_real_stock_by_product(df_company):
    """Recalcule le stock réel par produit à partir des flux"""
    
    # Trier par produit et quart
    df_sorted = df_company.sort_values(["CODE_PRODUIT", "QUART_SIMULATION"])
    
    # Calcul du stock réel
    df_sorted["STOCK_REEL"] = df_sorted.groupby("CODE_PRODUIT", observed=False)["PRODUCTION_TOTALE"].cumsum()
    df_sorted["STOCK_REEL"] = df_sorted["STOCK_REEL"] - df_sorted.groupby("CODE_PRODUIT", observed=False)["DEMANDE"].cumsum()
    
    # Ajouter le stock initial du premier quart
    first_stock = df_sorted.groupby("CODE_PRODUIT", observed=False)["STOCK_INITIAL"].first()
    df_sorted["STOCK_REEL"] = df_sorted["STOCK_REEL"] + df_sorted["CODE_PRODUIT"].map(first_stock)
    
    return df_sorted


def train_and_predict(df_company, model_choice):
    """Entraîne le modèle et génère les prédictions"""
    
    variables_explicatives = [
        "DEMANDE_LAG1", "PRIX_NET", "PRIX_MOYEN_LAG1",
        "DEPENSES_MARKETING", "MARKETING_LAG1", "COUT",
        "TREND", "CODE_PRODUIT", "ZONE_GEOGRAPHIQUE", "VENTES_NETTES"
    ]
    variable_cible = "DEMANDE"
    
    max_round = df_company["QUART_SIMULATION"].max()
    train = df_company[df_company["QUART_SIMULATION"] < max_round].copy()
    test = df_company[df_company["QUART_SIMULATION"] == max_round].copy()
    
    X_train = train[variables_explicatives].copy()
    y_train = train[variable_cible].copy()
    X_test = test[variables_explicatives].copy()
    y_test = test[variable_cible].copy()
    
    # Prétraitement
    numeric_features = [col for col, dtype in X_train.dtypes.items() if "float" in str(dtype) or "int" in str(dtype)]
    categorical_features = [col for col, dtype in X_train.dtypes.items() if "object" in str(dtype)]
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    
    X_train_prepare = preprocessor.fit_transform(X_train)
    X_test_prepare = preprocessor.transform(X_test)
    
    # Modèle
    if model_choice == "Ridge":
        model = Ridge(alpha=1.0)
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    
    model.fit(X_train_prepare, y_train)
    y_pred = model.predict(X_test_prepare).clip(min=0)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Prédictions futur
    last = df_company[df_company["QUART_SIMULATION"] == max_round].copy()
    X_next = last[variables_explicatives].copy()
    X_next_prepare = preprocessor.transform(X_next)
    pred_next = model.predict(X_next_prepare).clip(min=0).astype(int)
    
    return model, mae, r2, y_test, y_pred, pred_next, last, preprocessor, variables_explicatives


def calculate_indicators(df_company, pred_prod):
    """Calcule les indicateurs prévisionnels avec stock réel par produit"""
    
    # Prix moyen historique
    prix_moyen = (
        df_company.groupby("CODE_PRODUIT", as_index=False, observed=False)
        .agg({"VENTES_NETTES": "sum", "QUANTITY_DELIVERED": "sum"})
    )
    prix_moyen["PRIX_MOYEN_PREVISIONNEL"] = (
        prix_moyen["VENTES_NETTES"] / prix_moyen["QUANTITY_DELIVERED"]
    ).fillna(0)
    prix_moyen = prix_moyen[["CODE_PRODUIT", "PRIX_MOYEN_PREVISIONNEL"]]
    
    # Stock réel par produit (dernier quart)
    last_round = df_company["QUART_SIMULATION"].max()
    stock_prod = (
        df_company[df_company["QUART_SIMULATION"] == last_round]
        .groupby("CODE_PRODUIT", as_index=False, observed=False)["STOCK_REEL"]
        .first()
        .rename(columns={"STOCK_REEL": "STOCK_DISPONIBLE"})
    )
    
    # Coût unitaire moyen
    df_company["COUT_UNITAIRE"] = (df_company["COUT"] / df_company["DEMANDE"]).replace([float("inf"), -float("inf")], 0).fillna(0)
    cout_moyen = (
        df_company.groupby("CODE_PRODUIT", as_index=False, observed=False)["COUT_UNITAIRE"]
        .mean()
        .rename(columns={"COUT_UNITAIRE": "COUT_UNITAIRE_MOYEN"})
    )
    
    # Tableau final
    indicateurs = (
        pred_prod
        .merge(stock_prod, on="CODE_PRODUIT", how="left")
        .merge(prix_moyen, on="CODE_PRODUIT", how="left")
        .merge(cout_moyen, on="CODE_PRODUIT", how="left")
        .rename(columns={"PREDICTION": "DEMANDE_PREVISIONNELLE"})
    )
    
    indicateurs["STOCK_DISPONIBLE"] = indicateurs["STOCK_DISPONIBLE"].fillna(0)
    indicateurs["PRODUCTION_ESTIMEE"] = (indicateurs["DEMANDE_PREVISIONNELLE"] - indicateurs["STOCK_DISPONIBLE"]).clip(lower=0)
    indicateurs["CA_PREVISIONNEL"] = indicateurs["DEMANDE_PREVISIONNELLE"] * indicateurs["PRIX_MOYEN_PREVISIONNEL"]
    indicateurs["COUT_TOTAL_PREVISIONNEL"] = indicateurs["DEMANDE_PREVISIONNELLE"] * indicateurs["COUT_UNITAIRE_MOYEN"]
    indicateurs["MARGE_BRUTE_PREVISIONNELLE"] = indicateurs["CA_PREVISIONNEL"] - indicateurs["COUT_TOTAL_PREVISIONNEL"]
    
    return indicateurs


# ============================================================
# INTERFACE PRINCIPALE
# ============================================================

uploaded_file = st.file_uploader(
    "📁 Chargez les données",
    type=["xlsx"]
)

if uploaded_file is not None:
    # Chargement des données
    with st.spinner("Chargement et agrégation des données..."):
        df = load_and_process_data(uploaded_file)
    
    st.success("✅ Données chargées avec succès !")
    
    # Sélection de l'entreprise
    st.markdown("---")
    st.header("🏢 Sélection de l'entreprise")
    
    entreprises_disponibles = sorted(df["ENTREPRISE"].unique())
    
    col1, col2 = st.columns([2, 1])
    with col1:
        entreprise = st.selectbox(
            "Choisissez une entreprise à analyser",
            entreprises_disponibles,
            format_func=lambda x: f"Entreprise {x}"
        )
    
    with col2:
        st.metric("Entreprises disponibles", len(entreprises_disponibles))
    
    # Préparation des données de l'entreprise
    with st.spinner(f"Préparation des données pour l'entreprise {entreprise}..."):
        df_company = prepare_company_data(df, entreprise)
    
    # Informations sur l'entreprise
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Quarts de simulation", f"{df_company['QUART_SIMULATION'].nunique()}")
    with col2:
        st.metric("Produits", f"{df_company['CODE_PRODUIT'].nunique()}")
    with col3:
        st.metric("Enregistrements", f"{len(df_company)}")
    
    # ============================================================
    # ANALYSE EXPLORATOIRE DE DONNEES
    # ============================================================
    
    st.markdown("---")
    st.header("📈 Analyse exploratoire")
    
    tab1, tab2, tab3 = st.tabs(["Demande par produit", "Ventes par produit", "Statistiques"])
    
    with tab1:
        demande = (df_company
            .groupby(["QUART_SIMULATION", "CODE_PRODUIT"], observed=True)["DEMANDE"]
            .sum()
            .reset_index()
        )
        pivot_demande = demande.pivot(index="QUART_SIMULATION", columns="CODE_PRODUIT", values="DEMANDE")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_demande.plot(kind="bar", ax=ax)
        ax.set_xlabel("Quart de simulation")
        ax.set_ylabel("Demande")
        ax.set_title(f"Demande par produit - Entreprise {entreprise}")
        ax.legend(title="Produit", loc="upper left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        ventes = (df_company
            .groupby(["QUART_SIMULATION", "CODE_PRODUIT"], as_index=False)["VENTES_NETTES"]
            .sum()
        )
        pivot_ventes = ventes.pivot(index="QUART_SIMULATION", columns="CODE_PRODUIT", values="VENTES_NETTES")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_ventes.plot(kind="bar", ax=ax)
        ax.ticklabel_format(style="plain", axis="y")
        ax.set_xlabel("Quart de simulation")
        ax.set_ylabel("Ventes nettes")
        ax.set_title(f"Ventes par produit - Entreprise {entreprise}")
        ax.legend(title="Produit")
        ax.ticklabel_format(style='plain', axis='y')
        st.pyplot(fig)
    
    with tab3:
        # Exclure les colonnes de lag
        cols_a_exclure = ["DEMANDE_LAG1", "PRIX_MOYEN_LAG1", "MARKETING_LAG1", "TREND", "QUART_SIMULATION", "STOCK_REEL"]
        
        # Séparer les colonnes numériques et catégorielles
        numeric_cols = df_company.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_company.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Filtrer les colonnes numériques pour exclure les lags
        numeric_cols_filtered = [col for col in numeric_cols if col not in cols_a_exclure]
        df_num_stats = df_company[numeric_cols_filtered].copy()
        
        # Statistiques descriptives - variables numériques
        st.subheader("📊 Statistiques descriptives (variables numériques)")
        if len(df_num_stats.columns) > 0:
            st.dataframe(df_num_stats.describe().round(2))
        else:
            st.info("Aucune variable numérique à afficher")
        
        # Statistiques pour les variables catégorielles
        st.subheader("📋 Statistiques (variables catégorielles)")
        if len(categorical_cols) > 0:
            df_cat_stats = df_company[categorical_cols].describe()
            st.dataframe(df_cat_stats)
        else:
            st.info("Aucune variable catégorielle à afficher")
        
        # Corrélation - uniquement sur les colonnes numériques
        if len(df_num_stats.columns) > 1:
            st.subheader("📈 Matrice de corrélation")
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df_num_stats.corr().round(2)
            
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap="coolwarm", 
                ax=ax,
                fmt='.2f',
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8}
            )
            ax.set_title(f"Matrice de corrélation - Entreprise {entreprise}", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Afficher les corrélations les plus fortes avec DEMANDE
            if "DEMANDE" in df_num_stats.columns:
                st.subheader("🎯 Corrélations avec la demande")
                corr_with_demand = df_num_stats.corr()["DEMANDE"].drop("DEMANDE").sort_values(ascending=False)
                if len(corr_with_demand) > 0:
                    corr_df = pd.DataFrame({
                        "Variable": corr_with_demand.index,
                        "Corrélation avec DEMANDE": corr_with_demand.values
                    })
                    st.dataframe(corr_df.style.format({"Corrélation avec DEMANDE": "{:.3f}"}))
        else:
            st.info("Pas assez de variables numériques pour afficher la matrice de corrélation")
        
        # Évolution du stock réel par produit
        st.subheader("📦 Évolution du stock réel par produit")
        
        stock_evolution = df_company.groupby(
            ["QUART_SIMULATION", "CODE_PRODUIT"], observed=True
        )["STOCK_REEL"].first().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for produit in stock_evolution["CODE_PRODUIT"].unique():
            data = stock_evolution[stock_evolution["CODE_PRODUIT"] == produit]
            ax.plot(data["QUART_SIMULATION"], data["STOCK_REEL"], 
                    marker='o', linewidth=2, label=f"Produit {produit}")
        
        ax.set_xlabel("Quart de simulation")
        ax.set_ylabel("Stock réel")
        ax.set_title(f"Évolution du stock réel par produit - Entreprise {entreprise}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='plain', axis='y')
        st.pyplot(fig)
        
        # Tableau des stocks
        st.subheader("Tableau des stocks réels par produit et par quart")
        stock_pivot = stock_evolution.pivot(
            index="QUART_SIMULATION", 
            columns="CODE_PRODUIT", 
            values="STOCK_REEL"
        )
        st.dataframe(stock_pivot.style.format("{:,.0f}"))
    
    # ============================================================
    # MODÉLISATION
    # ============================================================
    
    st.markdown("---")
    st.header("🤖 Modélisation et prévisions")
    
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox(
            "Choisissez un modèle",
            ["Ridge", "Random Forest", "Gradient Boosting"]
        )
    
    if st.button("🚀 Lancer la prévision", type="primary"):
        with st.spinner(f"Entraînement du modèle {model_choice} pour l'entreprise {entreprise}..."):
            model, mae, r2, y_test, y_pred, pred_next, last, preprocessor, variables = train_and_predict(df_company, model_choice)
        
        # Métriques
        st.subheader("📊 Performance du modèle")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("MAE (Erreur absolue moyenne)", f"{mae:,.0f}")
        with col2:
            st.metric("R² (Coefficient de détermination)", f"{r2:.3f}")
        
        # Graphique des prédictions
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, color="steelblue", edgecolors="white")
        ax.plot([0, y_test.max()], [0, y_test.max()], "r--", label="Prédiction parfaite")
        ax.set_xlabel("Demande réelle")
        ax.set_ylabel("Demande prédite")
        ax.set_title(f"Prédiction de la demande\nMAE = {mae:,.0f} | R² = {r2:.3f}")
        ax.legend(loc="upper left", bbox_to_anchor=(1,0.5))
        fig.tight_layout()
        st.pyplot(fig)
        
        # Prédictions par produit
        st.subheader("📦 Demande prévisionnelle par produit")
        last["PREDICTION"] = pred_next
        pred_prod = (
            last.groupby("CODE_PRODUIT", as_index=False, observed=False)["PREDICTION"]
            .sum()
            .sort_values("PREDICTION", ascending=False)
        )
        
        st.dataframe(pred_prod.style.format({"PREDICTION": "{:,.0f}"}))
        
        # Graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(pred_prod["CODE_PRODUIT"], pred_prod["PREDICTION"], color="steelblue")
        ax.set_xlabel("Produit")
        ax.set_ylabel("Demande prévisionnelle")
        ax.set_title(f"Demande prévisionnelle par produit - {model_choice}")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Indicateurs financiers
        st.subheader("💰 Indicateurs prévisionnels")
        indicateurs = calculate_indicators(df_company, pred_prod)
        
        tableau_final = indicateurs[[
            "CODE_PRODUIT", "DEMANDE_PREVISIONNELLE", "STOCK_DISPONIBLE",
            "PRODUCTION_ESTIMEE", "PRIX_MOYEN_PREVISIONNEL",
            "CA_PREVISIONNEL", "MARGE_BRUTE_PREVISIONNELLE"
        ]].sort_values("DEMANDE_PREVISIONNELLE", ascending=False)
        
        st.dataframe(tableau_final.style.format({
            "DEMANDE_PREVISIONNELLE": "{:,.0f}",
            "STOCK_DISPONIBLE": "{:,.0f}",
            "PRODUCTION_ESTIMEE": "{:,.0f}",
            "PRIX_MOYEN_PREVISIONNEL": "{:.2f}",
            "CA_PREVISIONNEL": "{:,.2f}",
            "MARGE_BRUTE_PREVISIONNELLE": "{:,.2f}"
        }))
        
        # Synthèse financière
        st.subheader("📈 Synthèse financière")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CA total prévisionnel", f"{tableau_final['CA_PREVISIONNEL'].sum():,.0f} €")
        with col2:
            st.metric("Marge brute totale", f"{tableau_final['MARGE_BRUTE_PREVISIONNELLE'].sum():,.0f} €")
        
        # Graphique marge
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(tableau_final["CODE_PRODUIT"], tableau_final["MARGE_BRUTE_PREVISIONNELLE"], color="green")
        ax.set_xlabel("Produit")
        ax.set_ylabel("Marge brute prévisionnelle (€)")
        ax.set_title(f"Marge brute par produit - Entreprise {entreprise}")
        plt.tight_layout()
        st.pyplot(fig)
        
        # ============================================================
        # GRAPHIQUES COMPARATIFS SUPPLEMENTAIRES
        # ============================================================
        
        # Graphique comparatif Demande vs Stock vs Prédiction
        st.subheader("📊 Comparaison Demande, Stock et Prédiction")
        
        # Préparer les données pour le graphique
        comparaison_df = tableau_final[["CODE_PRODUIT", "DEMANDE_PREVISIONNELLE", "STOCK_DISPONIBLE"]].copy()
        
        # Ajouter la demande historique moyenne par produit
        demande_historique = (
            df_company.groupby("CODE_PRODUIT", as_index=False, observed=False)["DEMANDE"]
            .mean()
            .rename(columns={"DEMANDE": "DEMANDE_HISTORIQUE_MOYENNE"})
        )
        comparaison_df = comparaison_df.merge(demande_historique, on="CODE_PRODUIT", how="left")
        
        # Graphique en barres groupées
        fig, ax = plt.subplots(figsize=(12, 7))
        x = range(len(comparaison_df["CODE_PRODUIT"]))
        width = 0.25
        
        bars1 = ax.bar([i - width for i in x], comparaison_df["DEMANDE_HISTORIQUE_MOYENNE"], 
                       width, label="Demande historique moyenne", color="skyblue", edgecolor="black")
        bars2 = ax.bar(x, comparaison_df["STOCK_DISPONIBLE"], 
                       width, label="Stock disponible", color="orange", edgecolor="black")
        bars3 = ax.bar([i + width for i in x], comparaison_df["DEMANDE_PREVISIONNELLE"], 
                       width, label="Demande prévisionnelle", color="green", edgecolor="black")
        
        ax.set_xlabel("Produit", fontsize=12)
        ax.set_ylabel("Quantité", fontsize=12)
        ax.set_title(f"Comparaison Demande vs Stock vs Prédiction - Entreprise {entreprise}", fontsize=14)
        ax.set_xticks(x)
        ax.ticklabel_format(style='plain', axis='y')
        ax.set_xticklabels(comparaison_df["CODE_PRODUIT"])
        ax.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
        
        # Ajouter les valeurs sur les barres
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height):,}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=9, rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Graphique en courbes pour l'évolution temporelle
        st.subheader("📈 Évolution temporelle - Demande vs Stock")
        
        # Préparer les données temporelles
        temp_data = df_company.groupby("QUART_SIMULATION", as_index=False).agg({
            "DEMANDE": "sum",
            "STOCK_REEL": "sum"
        }).rename(columns={"STOCK_REEL": "STOCK_TOTAL"})
        
        # Ajouter les prédictions pour le quart suivant
        last_quarter = temp_data["QUART_SIMULATION"].max()
        new_row = pd.DataFrame({
            "QUART_SIMULATION": [last_quarter + 1],
            "DEMANDE": [tableau_final["DEMANDE_PREVISIONNELLE"].sum()],
            "STOCK_TOTAL": [tableau_final["STOCK_DISPONIBLE"].sum()]
        })
        temp_data = pd.concat([temp_data, new_row], ignore_index=True)
        
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(temp_data["QUART_SIMULATION"], temp_data["DEMANDE"], 
                 marker='o', linewidth=2, markersize=8, label="Demande totale", color="blue")
        ax2.plot(temp_data["QUART_SIMULATION"], temp_data["STOCK_TOTAL"], 
                 marker='s', linewidth=2, markersize=8, label="Stock total", color="orange")
        ax2.axvline(x=last_quarter + 0.5, color='red', linestyle='--', alpha=0.7, 
                    label="Zone de prédiction")
        
        ax2.set_xlabel("Quart de simulation", fontsize=12)
        ax2.set_ylabel("Quantité", fontsize=12)
        ax2.set_title(f"Évolution de la demande et du stock - Entreprise {entreprise}", fontsize=14)
        
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les points
        for i, row in temp_data.iterrows():
            ax2.annotate(f'{int(row["DEMANDE"]):,}', 
                        (row["QUART_SIMULATION"], row["DEMANDE"]),
                        textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
            ax2.annotate(f'{int(row["STOCK_TOTAL"]):,}', 
                        (row["QUART_SIMULATION"], row["STOCK_TOTAL"]),
                        textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)
        ax2.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Tableau récapitulatif
        st.subheader("📋 Tableau récapitulatif par produit")
        recap_df = comparaison_df[["CODE_PRODUIT", "DEMANDE_HISTORIQUE_MOYENNE", 
                                   "STOCK_DISPONIBLE", "DEMANDE_PREVISIONNELLE"]].copy()
        recap_df["BESOIN_NET"] = (recap_df["DEMANDE_PREVISIONNELLE"] - recap_df["STOCK_DISPONIBLE"]).clip(lower=0)
        recap_df["%_COUVERTURE_STOCK"] = (recap_df["STOCK_DISPONIBLE"] / recap_df["DEMANDE_PREVISIONNELLE"] * 100).round(1)
        
        st.dataframe(recap_df.style.format({
            "DEMANDE_HISTORIQUE_MOYENNE": "{:,.0f}",
            "STOCK_DISPONIBLE": "{:,.0f}",
            "DEMANDE_PREVISIONNELLE": "{:,.0f}",
            "BESOIN_NET": "{:,.0f}",
            "%_COUVERTURE_STOCK": "{:.1f}%"
        }).background_gradient(subset=["%_COUVERTURE_STOCK"], cmap="RdYlGn"))
        
        # Export
        csv = tableau_final.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger les résultats (CSV)",
            csv,
            f"previsions_{entreprise}_{model_choice.replace(' ', '_')}.csv",
            "text/csv"
        )

else:
    st.info(" Veuillez charger le fichier pour commencer l'analyse")
    
