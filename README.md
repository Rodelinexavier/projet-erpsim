ERPsim – Prévision de la demande et aide à la décision

Ce projet vise à analyser et prévoir la demande dans un contexte manufacturier à partir des données issues de la simulation ERPsim (Muesli). Il s’inscrit dans une démarche d’aide à la décision, en mobilisant des modèles de machine learning pour améliorer le pilotage de la performance.

🎯 Objectif

L’objectif est de répondre à la question suivante :
Dans quelle mesure les modèles prédictifs permettent-ils d’améliorer la qualité des décisions de gestion ?

L’application permet :

de prévoir la demande par produit,
d’estimer les besoins de production,
d’évaluer les indicateurs clés (stock, chiffre d’affaires, marge),
de comparer les performances par entreprise.


⚙️ Méthodologie
Préparation et agrégation des données ERPsim (ventes, production, stock, marketing)
Feature engineering (lags, variables économiques, tendance)
Entraînement de modèles de régression :
Ridge
Random Forest
Gradient Boosting
Évaluation des performances (MAE, R²)
Génération de prévisions et d’indicateurs décisionnels



📈 Application

Une application Streamlit permet de :

visualiser les données par produit et par entreprise,
lancer les modèles de prévision,
analyser les résultats via des tableaux et graphiques interactifs,
exporter les résultats.


🧠 Apports
Mise en œuvre concrète de modèles prédictifs en contexte opérationnel
Intégration de la dimension décisionnelle (production, stock, rentabilité)
Approche data-driven du pilotage de la performance
🛠️ Technologies utilisées
Python (Pandas, NumPy)
Scikit-learn
Matplotlib
Streamlit


📂 Données

Les données proviennent de la simulation ERPsim (HEC Montréal).
