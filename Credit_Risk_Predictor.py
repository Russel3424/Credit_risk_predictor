import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
dataset=pd.read_csv("cs-training.csv")

print("forme du dataset:",dataset.shape)
print(dataset.head())
print("nom des colonnes du dataset:\n",dataset.columns,"\n")
print("info sur le dataset:\n",dataset.info(),"\n")
print("description du dataset:\n",dataset.describe())
print("nombre de valeur manquante dans chaque colonne du dataset: \n",dataset.isnull().sum(),"\n")
print("le nombre de valeurs dupliquées est:",dataset.duplicated().sum())

dataset = dataset.drop(columns=["Unnamed: 0"])
print(dataset.shape)

#Analyse de la variable cible
print("describtion de la variable cible:\n",dataset["SeriousDlqin2yrs"].describe())
print(dataset['SeriousDlqin2yrs'].skew())
print("Distribution de la variable cible :\n",dataset["SeriousDlqin2yrs"].value_counts())
print("\nDistribution en pourcentage :")
print(dataset["SeriousDlqin2yrs"].value_counts(normalize=True) * 100)

plt.figure(figsize=(6,4))
sns.countplot(data=dataset, x="SeriousDlqin2yrs")
plt.title("Distribution de la variable cible")
plt.xlabel("Défaut sérieux dans les 2 ans")
plt.ylabel("Nombre de clients")
plt.show()

#Analyse des variables numérique
print(dataset.describe().T)
#1)distribution de variables numérique
dataset.hist(figsize=(15, 12), bins=50)
plt.suptitle("Distribution des variables numériques", fontsize=16)
plt.tight_layout()
plt.show()

#2)Comparons chaque variables numérique avec la variable cible
plt.figure(figsize=(8,5))
sns.boxplot(data=dataset,x="SeriousDlqin2yrs",y="age")
plt.title("Âge selon le statut de défaut")
plt.xlabel("Défaut sérieux dans les 2 ans")
plt.ylabel("Âge")
plt.show()

taux_defaut_age = dataset.groupby("age")["SeriousDlqin2yrs"].mean()
print("le taux d'endettement selon l'âge est: \n",taux_defaut_age)
plt.figure(figsize=(10,5))
plt.plot(taux_defaut_age.index, taux_defaut_age.values)
plt.title("Taux de défaut selon l'âge")
plt.xlabel("Âge")
plt.ylabel("Taux de défaut")
plt.show()

# 1. DESCRIPTION DE CHAQUE FEATURE

features = {
    "RevolvingUtilizationOfUnsecuredLines" : "Taux d'utilisation des crédits revolving (0 à 1)",
    "age"                                  : "Âge du client (en années)",
    "NumberOfTime30-59DaysPastDueNotWorse" : "Nb de fois en retard de 30-59 jours",
    "DebtRatio"                            : "Ratio dettes / revenus mensuels",
    "MonthlyIncome"                        : "Revenu mensuel (en $)",
    "NumberOfOpenCreditLinesAndLoans"      : "Nb de lignes de crédit ouvertes",
    "NumberOfTimes90DaysLate"              : "Nb de fois en retard de 90+ jours",
    "NumberRealEstateLoansOrLines"         : "Nb de prêts immobiliers",
    "NumberOfTime60-89DaysPastDueNotWorse" : "Nb de fois en retard de 60-89 jours",
    "NumberOfDependents"                   : "Nb de personnes à charge"
}

print("-" * 60)
print("DESCRIPTION DES FEATURES")
print("-" * 60)
for col, desc in features.items():
    print(col,":")
    print(desc)
    print("Min",dataset[col].min(),"|","Max",dataset[col].max(),"|","Moyenne",dataset[col].mean())

# 2. DISTRIBUTIONS DES VARIABLES NUMÉRIQUES

num_cols = dataset.drop(columns=["SeriousDlqin2yrs"]).columns.tolist()
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(dataset[col].dropna(), bins=50,color="steelblue", edgecolor="none", alpha=0.8)
    axes[i].set_title(col, fontsize=8)
    axes[i].set_ylabel("Fréquence")

plt.suptitle("Distribution de chaque feature", fontsize=13)
plt.tight_layout()
plt.show()

# 4. TAUX DE DÉFAUT PAR TRANCHE DE REVENU

print("\n" + "=" * 60)
print("TAUX DE DÉFAUT PAR TRANCHE DE REVENU")
print("=" * 60)
dataset["revenu_tranche"] = pd.cut(dataset["MonthlyIncome"].clip(upper=20000),
                                   bins=[0,1000, 2000, 3000, 4000, 5000, 6000,7000,8000,9000,10000,20000],
                                   labels=["<1k","1k-2k", "2k-3k","3k-4k","4k-5k","5k-6k", "7k-8k","8k-9k", "9k-10k", "10k-11k","+12k"])
taux_defaut_revenu = dataset.groupby("revenu_tranche", observed=True)["SeriousDlqin2yrs"].mean() * 100
print(taux_defaut_revenu.round(2))
plt.figure(figsize=(8, 4))
taux_defaut_revenu.plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Taux de défaut par tranche de revenu mensuel (%)")
plt.xlabel("Revenu mensuel ($)")
plt.ylabel("Taux de défaut (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# TAUX DE DÉFAUT Pour les  AUTRES VARIABLES 
# ── 1. DebtRatio ────────────────────────────────────────────

dataset["debtratio_tranche"] = pd.cut(dataset["DebtRatio"].clip(upper=2),
                                      bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0],labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%", ">100%"])
taux_debtratio = dataset.groupby("debtratio_tranche", observed=True)["SeriousDlqin2yrs"].mean() * 100

# ── 2. NumberRealEstateLoansOrLines ─────────────────────────

dataset["immobilier_tranche"] = pd.cut(dataset["NumberRealEstateLoansOrLines"].clip(upper=6),
                                       bins=[-1, 0, 1, 2, 3, 4, 6],
                                       labels=["0", "1", "2", "3", "4", "5+"])
taux_immobilier = dataset.groupby("immobilier_tranche", observed=True)["SeriousDlqin2yrs"].mean() * 100

# ── 3. NumberOfOpenCreditLinesAndLoans ──────────────────────

dataset["credit_lines_tranche"] = pd.cut(dataset["NumberOfOpenCreditLinesAndLoans"].clip(upper=20),
                                         bins=[-1, 2, 5, 8, 11, 14, 20],
                                         labels=["0-2", "3-5", "6-8", "9-11", "12-14", "15+"])

taux_credit_lines = dataset.groupby("credit_lines_tranche", observed=True)["SeriousDlqin2yrs"].mean() * 100

# ── 4. RevolvingUtilizationOfUnsecuredLines ─────────────────

dataset["revolving_tranche"] = pd.cut(dataset["RevolvingUtilizationOfUnsecuredLines"].clip(upper=1),
                                      bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                      labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])
taux_revolving = dataset.groupby("revolving_tranche", observed=True)["SeriousDlqin2yrs"].mean() * 100

# AFFICHAGE DES RÉSULTATS
print("TAUX DE DÉFAUT PAR DEBT RATIO:\n",taux_debtratio.round(2))
print("TAUX DE DÉFAUT PAR NB DE PRÊTS IMMOBILIERS:\n",taux_immobilier.round(2))
print("TAUX DE DÉFAUT PAR NB DE LIGNES DE CRÉDIT OUVERTES:\n",taux_credit_lines.round(2))
print("TAUX DE DÉFAUT PAR TAUX D'UTILISATION REVOLVING:\n",taux_revolving.round(2))

# VISUALISATION — 4 GRAPHIQUES EN GRILLE
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# ── Graphique 1 — DebtRatio ──
axes[0, 0].bar(taux_debtratio.index, taux_debtratio.values,color="tomato", edgecolor="black")
axes[0, 0].set_title("Taux de défaut par Debt Ratio", fontsize=11)
axes[0, 0].set_xlabel("Tranche de Debt Ratio")
axes[0, 0].set_ylabel("Taux de défaut (%)")
for i, v in enumerate(taux_debtratio.values):
    axes[0, 0].text(i, v + 0.1, f"{v:.1f}%",ha="center", fontsize=9, fontweight="bold") #ajoute les valeurs numériques au-dessus de chaque barre du graphique.
# ── Graphique 2 — NumberRealEstateLoansOrLines ──
axes[0, 1].bar(taux_immobilier.index, taux_immobilier.values,color="steelblue", edgecolor="black")
axes[0, 1].set_title("Taux de défaut par nb de prêts immobiliers", fontsize=11)
axes[0, 1].set_xlabel("Nombre de prêts immobiliers")
axes[0, 1].set_ylabel("Taux de défaut (%)")
for i, v in enumerate(taux_immobilier.values):
    axes[0, 1].text(i, v + 0.1, f"{v:.1f}%",ha="center", fontsize=9, fontweight="bold")
# ── Graphique 3 — NumberOfOpenCreditLinesAndLoans ──
axes[1, 0].bar(taux_credit_lines.index, taux_credit_lines.values,color="seagreen", edgecolor="black")
axes[1, 0].set_title("Taux de défaut par nb de lignes de crédit", fontsize=11)
axes[1, 0].set_xlabel("Nombre de lignes de crédit ouvertes")
axes[1, 0].set_ylabel("Taux de défaut (%)")
for i, v in enumerate(taux_credit_lines.values):
    axes[1, 0].text(i, v + 0.1, f"{v:.1f}%",ha="center", fontsize=9, fontweight="bold")
# ── Graphique 4 — RevolvingUtilization ──
axes[1, 1].bar(taux_revolving.index, taux_revolving.values,color="darkorange", edgecolor="black")
axes[1, 1].set_title("Taux de défaut par taux d'utilisation revolving", fontsize=11)
axes[1, 1].set_xlabel("Taux d'utilisation des crédits revolving")
axes[1, 1].set_ylabel("Taux de défaut (%)")
for i, v in enumerate(taux_revolving.values):
    axes[1, 1].text(i, v + 0.1, f"{v:.1f}%",ha="center", fontsize=9, fontweight="bold")
plt.suptitle("Taux de défaut par variable clé", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

