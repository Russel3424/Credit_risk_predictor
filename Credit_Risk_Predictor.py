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

# 3. TAUX DE DÉFAUT PAR Tranche d'âge

print("\n" + "=" * 60)
print("TAUX DE DÉFAUT PAR TRANCHE D'ÂGE")
print("=" * 60)

plt.figure(figsize=(10,5))
plt.plot(taux_defaut_age.index,taux_defaut_age.values)
plt.show()
dataset["age_tranche"] = pd.cut(dataset["age"],
                                bins=[0, 20,30, 40, 50, 60, 70, 80,90,100,109],
                                labels=["<20","20-30", "30-40", "40-50","50-60", "60-70", "70-80","80-90","90-100","100-109"])
taux_defaut_age = dataset.groupby("age_tranche", observed=True)["SeriousDlqin2yrs"].mean() * 100
print(taux_defaut_age.round(2))
plt.figure(figsize=(8, 4))
taux_defaut_age.plot(kind="bar", color="red", edgecolor="black")
plt.title("Taux de défaut par tranche d'âge (%)")
plt.xlabel("Tranche d'âge")
plt.ylabel("Taux de défaut (%)")
plt.xticks(rotation=0)
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

# 5. CORRÉLATIONS de chaque variable AVEC LA TARGET

corr = dataset.drop(columns=["age_tranche", "revenu_tranche","credit_lines_tranche","revolving_tranche","immobilier_tranche","debtratio_tranche"]).corr()["SeriousDlqin2yrs"].drop("SeriousDlqin2yrs").sort_values(ascending=False)

print("corrélation de chaque variable avec SeriousDlqin2yrs:\n",corr.round(4))

plt.figure(figsize=(8, 5))
corr.plot(kind="barh", color=["tomato" if x > 0
                               else "steelblue" for x in corr],
          edgecolor="black")
plt.title("Corrélation des features avec le défaut de crédit")
plt.xlabel("Corrélation")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.show()

# 6. DÉTECTION DES OUTLIERS CRITIQUES

print("\n" + "=" * 60)
print("DÉTECTION DES VALEURS ABERRANTES")
print("=" * 60)

# Valeurs suspectes dans les colonnes de retard de paiement
cols_retard = ["NumberOfTime30-59DaysPastDueNotWorse","NumberOfTimes90DaysLate","NumberOfTime60-89DaysPastDueNotWorse"]
for col in cols_retard:
    val_98 = dataset[col].quantile(0.98)
    suspects = (dataset[col] > val_98).sum()
    print(f"\n{col}")
    print(f"   98ème percentile : {val_98}")
    print(f"   Valeurs > 98ème  : {suspects} ({suspects/len(dataset)*100:.2f}%)")
    print(f"   Max              : {dataset[col].max()}")

# Age aberrant
print(f"\nAge minimum : {dataset['age'].min()} ans ← suspect si = 0")
print(f"Age maximum : {dataset['age'].max()} ans")

# RevolvingUtilization > 1
suspect_revolving = (dataset["RevolvingUtilizationOfUnsecuredLines"] > 1).sum()
print(f"\nRevolvingUtilization > 1 (impossible) : {suspect_revolving} lignes")

#Etape3: Nettoyage du dataset et préparation des données

# 1. SUPPRESSION DES COLONNES TEMPORAIRES DE L'EDA

colonnes_temp = ["age_tranche", "revenu_tranche", "debtratio_tranche","immobilier_tranche", "credit_lines_tranche","revolving_tranche"]
dataset = dataset.drop(columns=colonnes_temp, errors="ignore")

# 2. TRAITEMENT DES VALEURS ABERRANTES

print("\n" + "=" * 60)
print("TRAITEMENT DES VALEURS ABERRANTES")
print("=" * 60)

# ── 2.1 Colonnes de retard  ──────

cols_retard = ["NumberOfTime30-59DaysPastDueNotWorse","NumberOfTimes90DaysLate","NumberOfTime60-89DaysPastDueNotWorse"]

#On cherche les valeurs maximales de chaque variable lorsqu'on exclu les outliers 96 et 98 et on fait elles nos seuils
for col in cols_retard:
    max_reel = dataset[col][dataset[col] < 90].max()
    print(f"{col}")
    print(f"   Max réel (sans 96/98) : {max_reel}\n")
    
# On remplace les valeurs > à son seuil par la médiane de chaque colonne
seuils_retard = {"NumberOfTime30-59DaysPastDueNotWorse" : 13,"NumberOfTimes90DaysLate": 17,"NumberOfTime60-89DaysPastDueNotWorse" : 11}

for col, seuil in seuils_retard.items():
    # Calcul de la médiane sur les valeurs réelles uniquement
    mediane = dataset[col][dataset[col] <= seuil].median()
    # Nombre de valeurs aberrantes
    nb_aberrants = (dataset[col] > seuil).sum()
    # Remplacement
    dataset.loc[dataset[col] > seuil, col] = mediane
    print(f"\n── {col}")
    print(f"Seuil réel max    : {seuil}")
    print(f"Médiane calculée  : {mediane}")
    print(f"Valeurs corrigées : {nb_aberrants}")

# ── 2.2 Remplaçons les âges 0 par la médiane ───────────────────────────────────

nb_age_zero = (dataset["age"] == 0).sum()
mediane_age  = dataset["age"][dataset["age"] > 0].median()
dataset.loc[dataset["age"] == 0, "age"] = mediane_age

print(f"\nage = 0")
print(f"{nb_age_zero} valeurs remplacées par médiane ({mediane_age})")

# ── 2.3 RevolvingUtilization — écrêtage à 1 ─────────────────

nb_revolving = (dataset["RevolvingUtilizationOfUnsecuredLines"] > 1).sum()
dataset["RevolvingUtilizationOfUnsecuredLines"] = \
    dataset["RevolvingUtilizationOfUnsecuredLines"].clip(upper=1)
print(f"\nRevolvingUtilizationOfUnsecuredLines > 1")
print(f"{nb_revolving} valeurs écrêtées à 1.0")

# ── 2.4 DebtRatio — écrêtage à 1 ────────────────────────────

nb_debtratio = (dataset["DebtRatio"] > 1).sum()
dataset["DebtRatio"] = dataset["DebtRatio"].clip(upper=1)
print(f"\nDebtRatio > 1")
print(f"{nb_debtratio} valeurs écrêtées à 1.0")

# 3. TRAITEMENT DES VALEURS MANQUANTES

# ── 3.1 MonthlyIncome — 19.8% manquants ─────────────────────

# On impute par la médiane — plus robuste que la moyenne
# face aux revenus extrêmes encore présents
mediane_revenu = dataset["MonthlyIncome"].median()
dataset["MonthlyIncome"] = dataset["MonthlyIncome"].fillna(mediane_revenu)

# ── 3.2 NumberOfDependents — 2.6% manquants ─────────────────

mode_dependents = dataset["NumberOfDependents"].mode()[0]
dataset["NumberOfDependents"] = dataset["NumberOfDependents"].fillna(mode_dependents)

# 4. FEATURE ENGINEERING
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# ── 4.1 Total des incidents de retard ───────────────────────
dataset["TotalRetards"] = (dataset["NumberOfTime30-59DaysPastDueNotWorse"] +dataset["NumberOfTimes90DaysLate"] +dataset["NumberOfTime60-89DaysPastDueNotWorse"])
corr_total_retards = dataset["TotalRetards"].corr(dataset["SeriousDlqin2yrs"])
print(f"TotalRetards créé")
print(f"Corrélation avec défaut : {corr_total_retards:.4f}")

# ── 4.2 Revenu par personne à charge ────────────────────────

dataset["RevenuParPersonne"] = (dataset["MonthlyIncome"] /(dataset["NumberOfDependents"] + 1))
corr_revenu_personne = dataset["RevenuParPersonne"].corr(dataset["SeriousDlqin2yrs"])
print(f"\nRevenuParPersonne créé")
print(f"Corrélation avec défaut : {corr_revenu_personne:.4f}")

# ── 4.3 Log du revenu mensuel ───────────────────────────────

dataset["MonthlyIncome_log"] = np.log1p(dataset["MonthlyIncome"])
print(f"\nMonthlyIncome_log créé")
print(f"Skewness avant log : {dataset['MonthlyIncome'].skew():.3f}")
print(f"Skewness après log : {dataset['MonthlyIncome_log'].skew():.3f}")

# 5. VÉRIFICATION FINALE
print("\n" + "=" * 60)
print("VÉRIFICATION FINALE")
print("=" * 60)
print(f"\nDimensions finales : {dataset.shape}")
print(f"\nValeurs manquantes restantes :")
manquants = dataset.isnull().sum()
if manquants.sum() == 0:
    print("   ✅ Aucune valeur manquante")

# vérifions si il y'a des clients avec un MonthlyIncome=0
print("Valeurs MonthlyIncome = 0 :",(dataset["MonthlyIncome"] == 0).sum())
print("\nDistribution de MonthlyIncome_log :")
print(dataset["MonthlyIncome_log"].describe())

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(dataset["MonthlyIncome"].clip(upper=20000),
         bins=50, color="steelblue", edgecolor="none")
plt.title("MonthlyIncome (clipé à 20k)")

plt.subplot(1, 2, 2)
plt.hist(dataset["MonthlyIncome_log"],
         bins=50, color="seagreen", edgecolor="none")
plt.title("MonthlyIncome_log")

plt.tight_layout()
plt.show()