# Importation des bibliothèques nécessaires
import pandas as pd  # Pour la manipulation des données (dataframes)
import numpy as np  # Pour les opérations mathématiques (log transformation)

# Chargement des données depuis un fichier CSV
df = pd.read_csv('TCS.csv')  # Lecture du fichier CSV nommé 'TCS.csv' et stockage dans un dataframe (df)

# Conversion de la colonne 'Volume' en type float
df['Volume'] = df['Volume'].astype(float)  # Assurez-vous que la colonne 'Volume' est de type numérique (float)

# Conversion de la colonne 'Date' en type datetime (date au format standard)
df['Date'] = pd.to_datetime(df['Date'], format ='%Y-%m-%d')  # Conversion de la colonne 'Date' au format datetime (année-mois-jour)

# Sélection des colonnes à utiliser pour l'analyse
stock_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]  # Colonnes liées aux prix des actions et au volume
date_columns = ["Date"]  # La colonne contenant les dates

# Application d'une transformation logarithmique aux colonnes des prix
df[stock_columns] = np.log1p(df[stock_columns] + 1e-6)  # Transformation logarithmique (log(1+x)) des colonnes de stock.
# L'ajout de '1e-6' permet de gérer les cas où il pourrait y avoir des valeurs égales à 0 (évite log(0), qui est infini).

# Définition des caractéristiques (features) et des cibles (targets) pour l'entraînement d'un modèle
features = stock_columns  # Les colonnes 'Open', 'High', 'Low', 'Close', 'Adj Close', et 'Volume' sont utilisées comme caractéristiques
targets = ["Open", "Close"]  # Nous voulons prédire les valeurs de 'Open' et 'Close' comme cibles

# Définition de la taille des données d'entraînement (80% des données totales)
train_size = int(len(df) * 0.8)  # 80% des données seront utilisées pour l'entraînement, le reste pour les tests

# Séparation des données en ensemble d'entraînement et ensemble de test
train_data = df[:train_size]  # Les 80% premiers des données sont utilisés pour l'entraînement
test_data = df[train_size:]  # Les 20% restants sont utilisés pour les tests

# Optionnellement, vous pouvez afficher les ensembles d'entraînement et de test en décommentant ces lignes
# print(train_data)  # Affiche les données d'entraînement
# print(test_data)   # Affiche les données de test
