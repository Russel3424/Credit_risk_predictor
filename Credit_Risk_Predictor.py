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

