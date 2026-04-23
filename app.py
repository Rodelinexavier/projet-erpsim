# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ------------------------------------------------------------
# Configuration de la page
# ------------------------------------------------------------
st.set_page_config(
    page_title="Prévision de la demande - Dashboard",
    page_icon="📊",
    layout="wide"
)

pd.set_option("display.float_format", "{:.2f}".format)

st.title("📊 Prévision de la demande - Tableau de bord décisionnel")
st.markdown("---")


# ============================================================
# CHARGEMENT ET PREPARATION DES DONNEES
# ============================================================

@st.cache_data
def load_and_process_data(uploaded_file: str) -> pd.DataFrame:
    """
    Charge les feuilles ERPsim et construit un DataFrame agrégé.
    """

    ventes = pd.read_excel(uploaded_file, sheet_name="Sales")
    production = pd.read_excel(uploaded_file, sheet_name="Production")
    inventaire = pd.read_excel(uploaded_file, sheet_name="Inventory")
    marche = pd.read_excel(uploaded_file, sheet_name="Market")
    marketing = pd.read_excel(uploaded_file, sheet_name="Marketing_Expenses")

    # Harmonisation des noms de colonnes
    ventes.columns = [c.strip().upper() for c in ventes.columns]
    production.columns = [c.strip().upper() for c in production.columns]
    inventaire.columns = [c.strip().upper() for c in inventaire.columns]
    marche.columns = [c.strip().upper() for c in marche.columns]
    marketing.columns = [c.strip().upper() for c in marketing.columns]

    # --------------------------------------------------------
    # Agrégation ventes
    # --------------------------------------------------------
    quantity_col = "QUANTITY" if "QUANTITY" in ventes.columns else "QUANTITY_DELIVERED"

    ventes_group_cols = [
        "SIM_ROUND",
        "SALES_ORGANIZATION",
        "MATERIAL_CODE",
        "DISTRIBUTION_CHANNEL",
        "AREA",
        "NET_PRICE",
        "NET_VALUE",
        "COST",
        "QUANTITY_DELIVERED",
        "CONTRIBUTION_MARGIN",
        "MATERIAL_DESCRIPTION"
    ]
    ventes_group_cols = [c for c in ventes_group_cols if c in ventes.columns]

    ventes_agg = (
        ventes.groupby(ventes_group_cols, as_index=False)[quantity_col]
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
            quantity_col: "DEMANDE"
        })
    )

    # --------------------------------------------------------
    # Agrégation production
    # --------------------------------------------------------
    production_agg = (
        production.groupby(["COMPANY_CODE", "SIM_ROUND", "MATERIAL_CODE"], as_index=False)["YIELD"]
        .sum()
        .rename(columns={
            "COMPANY_CODE": "ENTREPRISE",
            "SIM_ROUND": "QUART_SIMULATION",
            "MATERIAL_CODE": "CODE_PRODUIT",
            "YIELD": "PRODUCTION_TOTALE"
        })
    )

    # --------------------------------------------------------
    # Agrégation marché
    # --------------------------------------------------------
    marche_company_col = "COMPANY_CODE" if "COMPANY_CODE" in marche.columns else None
    if marche_company_col is not None:
        marche_agg = (
            marche.groupby([marche_company_col, "SIM_ROUND", "DISTRIBUTION_CHANNEL"], as_index=False)["AVERAGE_PRICE"]
            .mean()
            .rename(columns={
                marche_company_col: "ENTREPRISE",
                "SIM_ROUND": "QUART_SIMULATION",
                "DISTRIBUTION_CHANNEL": "CANAL_DISTRIBUTION",
                "AVERAGE_PRICE": "PRIX_MOYEN"
            })
        )
    else:
        marche_agg = (
            marche.groupby(["SIM_ROUND", "DISTRIBUTION_CHANNEL"], as_index=False)["AVERAGE_PRICE"]
            .mean()
            .rename(columns={
                "SIM_ROUND": "QUART_SIMULATION",
                "DISTRIBUTION_CHANNEL": "CANAL_DISTRIBUTION",
                "AVERAGE_PRICE": "PRIX_MOYEN"
            })
        )

    # --------------------------------------------------------
    # Agrégation marketing
    # --------------------------------------------------------
    marketing_agg = (
        marketing.groupby(["SIM_ROUND", "SALES_ORGANIZATION"], as_index=False)["AMOUNT"]
        .sum()
        .rename(columns={
            "SIM_ROUND": "QUART_SIMULATION",
            "SALES_ORGANIZATION": "ENTREPRISE",
            "AMOUNT": "DEPENSES_MARKETING"
        })
    )

    # Agrégation inventaire
   
    inventory_product_col = "MATERIAL_CODE" if "MATERIAL_CODE" in inventaire.columns else None

    if inventory_product_col is not None:
        inventaire_agg = (
            inventaire.groupby(["SIM_ROUND", inventory_product_col], as_index=False)["INVENTORY_OPENING_BALANCE"]
            .sum()
            .rename(columns={
                "SIM_ROUND": "QUART_SIMULATION",
                inventory_product_col: "CODE_PRODUIT",
                "INVENTORY_OPENING_BALANCE": "STOCK_INITIAL"
            })
        )
    else:
        inventaire_agg = (
            inventaire.groupby(["SIM_ROUND"], as_index=False)["INVENTORY_OPENING_BALANCE"]
            .mean()
            .rename(columns={
                "SIM_ROUND": "QUART_SIMULATION",
                "INVENTORY_OPENING_BALANCE": "STOCK_INITIAL_GLOBAL"
            })
        )

   
    # Fusion

    df = ventes_agg.merge(
        production_agg,
        on=["ENTREPRISE", "QUART_SIMULATION", "CODE_PRODUIT"],
        how="left"
    )

    df = df.merge(
        marketing_agg,
        on=["QUART_SIMULATION", "ENTREPRISE"],
        how="left"
    )

    if "ENTREPRISE" in marche_agg.columns:
        df = df.merge(
            marche_agg,
            on=["QUART_SIMULATION", "CANAL_DISTRIBUTION", "ENTREPRISE"],
            how="left"
        )
    else:
        df = df.merge(
            marche_agg,
            on=["QUART_SIMULATION", "CANAL_DISTRIBUTION"],
            how="left"
        )

    if inventory_product_col is not None:
        df = df.merge(
            inventaire_agg,
            on=["QUART_SIMULATION", "CODE_PRODUIT"],
            how="left"
        )
    else:
        df = df.merge(
            inventaire_agg,
            on="QUART_SIMULATION",
            how="left"
        )
        df["STOCK_INITIAL"] = df["STOCK_INITIAL_GLOBAL"]

    # Remplissage minimal
    for col in ["PRODUCTION_TOTALE", "DEPENSES_MARKETING", "PRIX_MOYEN", "STOCK_INITIAL"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


@st.cache_data
def prepare_company_data(df: pd.DataFrame, entreprise: str) -> pd.DataFrame:
    """
    Prépare les données pour une entreprise spécifique.
    """
    df_company = df[df["ENTREPRISE"] == entreprise].copy()

    if df_company.empty:
        return df_company

    df_company = df_company.sort_values(["CODE_PRODUIT", "QUART_SIMULATION"]).reset_index(drop=True)

    if "CANAL_DISTRIBUTION" in df_company.columns:
        df_company["CANAL_DISTRIBUTION"] = df_company["CANAL_DISTRIBUTION"].astype("object")

    groupes = ["ENTREPRISE", "CODE_PRODUIT"]

    df_company["DEMANDE_LAG1"] = df_company.groupby(groupes, observed=True)["DEMANDE"].shift(1)
    df_company["PRIX_MOYEN_LAG1"] = df_company.groupby(groupes, observed=True)["PRIX_MOYEN"].shift(1)
    df_company["MARKETING_LAG1"] = df_company.groupby(groupes, observed=True)["DEPENSES_MARKETING"].shift(1)
    df_company["TREND"] = df_company["QUART_SIMULATION"]

    df_company = df_company.dropna(
        subset=["DEMANDE_LAG1", "PRIX_MOYEN_LAG1", "MARKETING_LAG1"]
    ).copy()

    return df_company


def train_and_predict(df_company: pd.DataFrame, model_choice: str):
    """
    Entraîne le modèle et génère les prédictions.
    """
    variables_explicativas = [
        "DEMANDE_LAG1",
        "PRIX_NET",
        "PRIX_MOYEN_LAG1",
        "DEPENSES_MARKETING",
        "MARKETING_LAG1",
        "COUT",
        "TREND",
        "CODE_PRODUIT",
        "ZONE_GEOGRAPHIQUE",
        "VENTES_NETTES"
    ]
    variables_explicativas = [c for c in variables_explicativas if c in df_company.columns]

    variable_cible = "DEMANDE"

    max_round = df_company["QUART_SIMULATION"].max()

    train = df_company[df_company["QUART_SIMULATION"] < max_round].copy()
    test = df_company[df_company["QUART_SIMULATION"] == max_round].copy()

    if train.empty or test.empty:
        raise ValueError("Impossible de séparer les données en train/test par quart.")

    X_train = train[variables_explicativas].copy()
    y_train = train[variable_cible].copy()

    X_test = test[variables_explicativas].copy()
    y_test = test[variable_cible].copy()

    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

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

    if model_choice == "Ridge":
        model = Ridge(alpha=1.0)
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

    model.fit(X_train_prepare, y_train)

    y_pred = np.clip(model.predict(X_test_prepare), a_min=0, a_max=None)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Prédiction du quart suivant à partir du dernier quart observé
    last = df_company[df_company["QUART_SIMULATION"] == max_round].copy()
    X_next = last[variables_explicativas].copy()
    X_next_prepare = preprocessor.transform(X_next)
    pred_next = np.clip(model.predict(X_next_prepare), a_min=0, a_max=None).astype(int)

    return model, mae, r2, y_test, y_pred, pred_next, last


def calculate_indicators(df_company: pd.DataFrame, pred_prod: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les indicateurs prévisionnels.
    Utilise le STOCK DU DERNIER QUART (non recalculé).
    """
    # Prix moyen historique
    quantity_ref = "QUANTITY_DELIVERED" if "QUANTITY_DELIVERED" in df_company.columns else "DEMANDE"

    prix_moyen = (
        df_company.groupby("CODE_PRODUIT", as_index=False, observed=False)
        .agg({
            "VENTES_NETTES": "sum",
            quantity_ref: "sum"
        })
    )

    prix_moyen["PRIX_MOYEN_PREVISIONNEL"] = np.where(
        prix_moyen[quantity_ref] != 0,
        prix_moyen["VENTES_NETTES"] / prix_moyen[quantity_ref],
        0
    )
    prix_moyen = prix_moyen[["CODE_PRODUIT", "PRIX_MOYEN_PREVISIONNEL"]]

    # STOCK DISPONIBLE - Version simplifiée
    # On prend le stock initial du dernier quart pour chaque produit
    
    last_round = df_company["QUART_SIMULATION"].max()
    
    stock_prod = (
        df_company[df_company["QUART_SIMULATION"] == last_round]
        .groupby("CODE_PRODUIT", as_index=False, observed=False)["STOCK_INITIAL"]
        .first()
        .rename(columns={"STOCK_INITIAL": "STOCK_DISPONIBLE"})
    )

    # Coût unitaire moyen
    df_cost = df_company.copy()
    df_cost["COUT_UNITAIRE"] = np.where(
        df_cost["DEMANDE"] != 0,
        df_cost["COUT"] / df_cost["DEMANDE"],
        0
    )

    cout_moyen = (
        df_cost.groupby("CODE_PRODUIT", as_index=False, observed=False)["COUT_UNITAIRE"]
        .mean()
        .rename(columns={"COUT_UNITAIRE": "COUT_UNITAIRE_MOYEN"})
    )

    indicateurs = (
        pred_prod
        .merge(stock_prod, on="CODE_PRODUIT", how="left")
        .merge(prix_moyen, on="CODE_PRODUIT", how="left")
        .merge(cout_moyen, on="CODE_PRODUIT", how="left")
        .rename(columns={"PREDICTION": "DEMANDE_PREVISIONNELLE"})
    )

    indicateurs["STOCK_DISPONIBLE"] = indicateurs["STOCK_DISPONIBLE"].fillna(0)
    indicateurs["PRIX_MOYEN_PREVISIONNEL"] = indicateurs["PRIX_MOYEN_PREVISIONNEL"].fillna(0)
    indicateurs["COUT_UNITAIRE_MOYEN"] = indicateurs["COUT_UNITAIRE_MOYEN"].fillna(0)

    indicateurs["PRODUCTION_ESTIMEE"] = (
        indicateurs["DEMANDE_PREVISIONNELLE"] - indicateurs["STOCK_DISPONIBLE"]
    ).clip(lower=0)

    indicateurs["CA_PREVISIONNEL"] = (
        indicateurs["DEMANDE_PREVISIONNELLE"] * indicateurs["PRIX_MOYEN_PREVISIONNEL"]
    )

    indicateurs["COUT_TOTAL_PREVISIONNEL"] = (
        indicateurs["DEMANDE_PREVISIONNELLE"] * indicateurs["COUT_UNITAIRE_MOYEN"]
    )

    indicateurs["MARGE_BRUTE_PREVISIONNELLE"] = (
        indicateurs["CA_PREVISIONNEL"] - indicateurs["COUT_TOTAL_PREVISIONNEL"]
    )

    return indicateurs



# INTERFACE PRINCIPALE


uploaded_file = st.file_uploader("📁 Chargez les données", type=["xlsx"])

if uploaded_file is not None:
    with st.spinner("Chargement et agrégation des données..."):
        df = load_and_process_data(uploaded_file)

    st.success("✅ Chargement de données reussi !")

    st.markdown("---")
    st.header("🏢 Sélection de l'entreprise")

    entreprises_disponibles = sorted(df["ENTREPRISE"].dropna().unique().tolist())

    col1, col2 = st.columns([2, 1])
    with col1:
        entreprise = st.selectbox(
            "Choisissez une entreprise à analyser",
            entreprises_disponibles,
            format_func=lambda x: f"Entreprise {x}"
        )
    with col2:
        st.metric("Entreprises disponibles", len(entreprises_disponibles))

    with st.spinner(f"Préparation des données pour l'entreprise {entreprise}..."):
        df_company = prepare_company_data(df, entreprise)

    if df_company.empty:
        st.error("Aucune donnée exploitable après préparation pour cette entreprise.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Quarts de simulation", f"{df_company['QUART_SIMULATION'].nunique()}")
    with col2:
        st.metric("Produits", f"{df_company['CODE_PRODUIT'].nunique()}")
    with col3:
        st.metric("Enregistrements", f"{len(df_company)}")

   
    # ANALYSE EXPLORATOIRE
   
    st.markdown("---")
    st.header("📈 Analyse exploratoire")

    tab1, tab2, tab3 = st.tabs(["Demande par produit", "Ventes par produit", "Statistiques"])

    with tab1:
        demande = (
            df_company.groupby(["QUART_SIMULATION", "CODE_PRODUIT"], observed=True)["DEMANDE"]
            .sum()
            .reset_index()
        )
        pivot_demande = demande.pivot(
            index="QUART_SIMULATION",
            columns="CODE_PRODUIT",
            values="DEMANDE"
        )

        fig, ax = plt.subplots(figsize=(14, 7))
        pivot_demande.plot(kind="bar", ax=ax)
        ax.set_xlabel("Quart de simulation")
        ax.set_ylabel("Demande")
        ax.set_title(f"Demande par produit - Entreprise {entreprise}")
        ax.legend(title="Produit", loc="upper left", bbox_to_anchor=(1, 0.5))
        ax.ticklabel_format(style="plain", axis="y")
        
        # PAS DE CHIFFRES SUR CE GRAPHIQUE (barres trop nombreuses)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        ventes = (
            df_company.groupby(["QUART_SIMULATION", "CODE_PRODUIT"], as_index=False)["VENTES_NETTES"]
            .sum()
        )
        pivot_ventes = ventes.pivot(
            index="QUART_SIMULATION",
            columns="CODE_PRODUIT",
            values="VENTES_NETTES"
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_ventes.plot(kind="bar", ax=ax)
        ax.set_xlabel("Quart de simulation")
        ax.set_ylabel("Ventes nettes")
        ax.set_title(f"Ventes par produit - Entreprise {entreprise}")
        ax.legend(title="Produit", loc="upper left", bbox_to_anchor=(1, 0.5))
        ax.ticklabel_format(style="plain", axis="y")
        
        # CHIFFRES SUR LES BARRES
        for container in ax.containers:
            ax.bar_label(container, fmt='{:,.0f}', fontsize=9, rotation=0)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        cols_a_exclure = [
            "DEMANDE_LAG1", "PRIX_MOYEN_LAG1", "MARKETING_LAG1",
            "TREND", "QUART_SIMULATION"
        ]

        numeric_cols = df_company.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_company.select_dtypes(include=["object", "category"]).columns.tolist()

        numeric_cols_filtered = [col for col in numeric_cols if col not in cols_a_exclure]
        df_num_stats = df_company[numeric_cols_filtered].copy()

        st.subheader("📊 Statistiques descriptives (variables numériques)")
        if len(df_num_stats.columns) > 0:
            st.dataframe(df_num_stats.describe().round(2))
        else:
            st.info("Aucune variable numérique à afficher")

        st.subheader("📋 Statistiques (variables catégorielles)")
        if len(categorical_cols) > 0:
            st.dataframe(df_company[categorical_cols].describe())
        else:
            st.info("Aucune variable catégorielle à afficher")

        st.subheader("📦 Stock par produit au dernier quart")
        
        last_round = df_company["QUART_SIMULATION"].max()
        stock_last_quarter = (
            df_company[df_company["QUART_SIMULATION"] == last_round]
            .groupby("CODE_PRODUIT", as_index=False, observed=False)["STOCK_INITIAL"]
            .first()
            .rename(columns={"STOCK_INITIAL": "STOCK_DISPONIBLE"})
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(stock_last_quarter["CODE_PRODUIT"], stock_last_quarter["STOCK_DISPONIBLE"], color="steelblue")
        ax.set_xlabel("Produit")
        ax.set_ylabel("Stock disponible")
        ax.set_title(f"Stock disponible par produit - Quart {last_round} - Entreprise {entreprise}")
        ax.ticklabel_format(style="plain", axis="y")
        
        # CHIFFRES SUR LES BARRES
        for container in ax.containers:
            ax.bar_label(container, fmt='{:,.0f}', fontsize=9, rotation=0)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.dataframe(stock_last_quarter.style.format({"STOCK_DISPONIBLE": "{:,.0f}"}))

    
    # MODELISATION
    
    st.markdown("---")
    st.header("🤖 Modélisation et prévisions")

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox(
            "Choisissez un modèle",
            ["Ridge", "Random Forest", "Gradient Boosting"]
        )

    if st.button("🚀 Lancer la prévision", type="primary"):
        try:
            with st.spinner(f"Entraînement du modèle {model_choice} pour l'entreprise {entreprise}..."):
                model, mae, r2, y_test, y_pred, pred_next, last = train_and_predict(df_company, model_choice)
        except Exception as e:
            st.error(f"Erreur pendant l'entraînement : {e}")
            st.stop()

        st.subheader("📊 Performance du modèle")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("MAE (Erreur absolue moyenne)", f"{mae:,.0f}")
        with col2:
            st.metric("R² (Coefficient de détermination)", f"{r2:.3f}")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="white")
        
        max_val = max(float(np.max(y_test)) if len(y_test) > 0 else 0, 
                      float(np.max(y_pred)) if len(y_pred) > 0 else 0)
        if max_val > 0:
            ax.plot([0, max_val], [0, max_val], "r--", label="Prédiction parfaite")
        
        ax.set_xlabel("Demande réelle")
        ax.set_ylabel("Demande prédite")
        ax.set_title(f"Prédiction de la demande\nMAE = {mae:,.0f} | R² = {r2:.3f}")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
        ax.ticklabel_format(style="plain", axis='both')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("📦 Demande prévisionnelle par produit")
        last = last.copy()
        last["PREDICTION"] = pred_next

        pred_prod = (
            last.groupby("CODE_PRODUIT", as_index=False, observed=False)["PREDICTION"]
            .sum()
            .sort_values("PREDICTION", ascending=False)
        )

        st.dataframe(pred_prod.style.format({"PREDICTION": "{:,.0f}"}))

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(pred_prod["CODE_PRODUIT"], pred_prod["PREDICTION"], color="steelblue")
        ax.set_xlabel("Produit")
        ax.set_ylabel("Demande prévisionnelle")
        ax.set_title(f"Demande prévisionnelle par produit - {model_choice}")
        ax.ticklabel_format(style="plain", axis="y")
        
        # CHIFFRES SUR LES BARRES
        for container in ax.containers:
            ax.bar_label(container, fmt='{:,.0f}', fontsize=9, rotation=0)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("💰 Indicateurs prévisionnels")
        indicateurs = calculate_indicators(df_company, pred_prod)

        tableau_final = indicateurs[
            [
                "CODE_PRODUIT",
                "DEMANDE_PREVISIONNELLE",
                "STOCK_DISPONIBLE",
                "PRODUCTION_ESTIMEE",
                "PRIX_MOYEN_PREVISIONNEL",
                "CA_PREVISIONNEL",
                "MARGE_BRUTE_PREVISIONNELLE"
            ]
        ].sort_values("DEMANDE_PREVISIONNELLE", ascending=False)

        st.dataframe(
            tableau_final.style.format({
                "DEMANDE_PREVISIONNELLE": "{:,.0f}",
                "STOCK_DISPONIBLE": "{:,.0f}",
                "PRODUCTION_ESTIMEE": "{:,.0f}",
                "PRIX_MOYEN_PREVISIONNEL": "{:.2f}",
                "CA_PREVISIONNEL": "{:,.2f}",
                "MARGE_BRUTE_PREVISIONNELLE": "{:,.2f}"
            })
        )

        st.subheader("📈 Synthèse financière")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CA total prévisionnel", f"{tableau_final['CA_PREVISIONNEL'].sum():,.0f} €")
        with col2:
            st.metric("Marge brute totale", f"{tableau_final['MARGE_BRUTE_PREVISIONNELLE'].sum():,.0f} €")

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(tableau_final["CODE_PRODUIT"], tableau_final["MARGE_BRUTE_PREVISIONNELLE"], color="green")
        ax.set_xlabel("Produit")
        ax.set_ylabel("Marge brute prévisionnelle (€)")
        ax.set_title(f"Marge brute par produit - Entreprise {entreprise}")
        ax.ticklabel_format(style="plain", axis="y")
        
        # CHIFFRES SUR LES BARRES
        for container in ax.containers:
            ax.bar_label(container, fmt='{:,.0f}', fontsize=9, rotation=0)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("📊 Comparaison Demande, Stock et Prédiction")

        comparaison_df = tableau_final[
            ["CODE_PRODUIT", "DEMANDE_PREVISIONNELLE", "STOCK_DISPONIBLE"]
        ].copy()

        demande_historique = (
            df_company.groupby("CODE_PRODUIT", as_index=False, observed=False)["DEMANDE"]
            .mean()
            .rename(columns={"DEMANDE": "DEMANDE_HISTORIQUE_MOYENNE"})
        )
        comparaison_df = comparaison_df.merge(demande_historique, on="CODE_PRODUIT", how="left")

        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(comparaison_df["CODE_PRODUIT"]))
        width = 0.25

        bars1 = ax.bar(x - width, comparaison_df["DEMANDE_HISTORIQUE_MOYENNE"], width, 
                       label="Demande historique moyenne", color="skyblue", edgecolor="black")
        bars2 = ax.bar(x, comparaison_df["STOCK_DISPONIBLE"], width, 
                       label="Stock disponible", color="orange", edgecolor="black")
        bars3 = ax.bar(x + width, comparaison_df["DEMANDE_PREVISIONNELLE"], width, 
                       label="Demande prévisionnelle", color="green", edgecolor="black")

        ax.set_xlabel("Produit")
        ax.set_ylabel("Quantité")
        ax.set_title(f"Comparaison Demande vs Stock vs Prédiction - Entreprise {entreprise}")
        ax.set_xticks(x)
        ax.set_xticklabels(comparaison_df["CODE_PRODUIT"])
        ax.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
        ax.ticklabel_format(style="plain", axis="y")
        
        # CHIFFRES SUR TOUTES LES BARRES
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height):,}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8, rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("📈 Évolution temporelle - Demande vs Stock")

        temp_data = (
            df_company.groupby("QUART_SIMULATION", as_index=False)
            .agg({"DEMANDE": "sum", "STOCK_INITIAL": "mean"})
            .rename(columns={"STOCK_INITIAL": "STOCK_MOYEN"})
        )

        last_quarter = temp_data["QUART_SIMULATION"].max()
        new_row = pd.DataFrame({
            "QUART_SIMULATION": [last_quarter + 1],
            "DEMANDE": [tableau_final["DEMANDE_PREVISIONNELLE"].sum()],
            "STOCK_MOYEN": [tableau_final["STOCK_DISPONIBLE"].mean()]
        })
        temp_data = pd.concat([temp_data, new_row], ignore_index=True)

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(temp_data["QUART_SIMULATION"], temp_data["DEMANDE"], 
                 marker="o", linewidth=2, label="Demande totale", color="blue")
        ax2.plot(temp_data["QUART_SIMULATION"], temp_data["STOCK_MOYEN"], 
                 marker="s", linewidth=2, label="Stock moyen", color="orange")
        ax2.axvline(x=last_quarter + 0.5, color="red", linestyle="--", alpha=0.7, 
                    label="Zone de prédiction")

        ax2.set_xlabel("Quart de simulation")
        ax2.set_ylabel("Quantité")
        ax2.set_title(f"Évolution de la demande et du stock - Entreprise {entreprise}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style="plain", axis="y")
        
        # CHIFFRES SUR LES POINTS
        for _, row in temp_data.iterrows():
            ax2.annotate(f'{int(row["DEMANDE"]):,}', 
                        (row["QUART_SIMULATION"], row["DEMANDE"]),
                        textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
            ax2.annotate(f'{int(row["STOCK_MOYEN"]):,}', 
                        (row["QUART_SIMULATION"], row["STOCK_MOYEN"]),
                        textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        st.subheader("📋 Tableau récapitulatif par produit")
        recap_df = comparaison_df[
            ["CODE_PRODUIT", "DEMANDE_HISTORIQUE_MOYENNE", "STOCK_DISPONIBLE", "DEMANDE_PREVISIONNELLE"]
        ].copy()

        recap_df["BESOIN_NET"] = (
            recap_df["DEMANDE_PREVISIONNELLE"] - recap_df["STOCK_DISPONIBLE"]
        ).clip(lower=0)

        recap_df["%_COUVERTURE_STOCK"] = np.where(
            recap_df["DEMANDE_PREVISIONNELLE"] != 0,
            recap_df["STOCK_DISPONIBLE"] / recap_df["DEMANDE_PREVISIONNELLE"] * 100,
            0
        ).round(1)

        st.dataframe(
            recap_df.style.format({
                "DEMANDE_HISTORIQUE_MOYENNE": "{:,.0f}",
                "STOCK_DISPONIBLE": "{:,.0f}",
                "DEMANDE_PREVISIONNELLE": "{:,.0f}",
                "BESOIN_NET": "{:,.0f}",
                "%_COUVERTURE_STOCK": "{:.1f}%"
            }).background_gradient(subset=["%_COUVERTURE_STOCK"], cmap="RdYlGn")
        )

        csv = tableau_final.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger les résultats (CSV)",
            csv,
            f"previsions_{entreprise}_{model_choice.replace(' ', '_')}.csv",
            "text/csv"
        )

else:
    st.info("Veuillez charger le fichier pour commencer l'analyse.")
