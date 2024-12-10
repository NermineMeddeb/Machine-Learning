import os
# Réduire les messages d'avertissement de TensorFlow pour un affichage plus propre
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense
from sklearn.impute import SimpleImputer
import warnings
import csv

# Créer un dictionnaire pour stocker les résultats des modèles GRU
gru_results = {}

# Importer les données préparées (train_data, test_data, features, targets)
from Data_pre import train_data, test_data, features, targets

# Définir les différents horizons de prédiction (en jours)
prediction_horizons = [1, 3, 10, 30]

# Fonction pour créer un modèle GRU
def create_gru_model(input_shape):
    """
    Crée un modèle GRU avec trois couches GRU et une couche Dense en sortie.
    :param input_shape: Forme des données d'entrée pour le modèle.
    :return: Modèle GRU compilé.
    """
    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=input_shape))
    model.add(GRU(units=50, return_sequences=True))
    model.add(GRU(units=50))
    model.add(Dense(1))  # Une seule sortie car c'est une prédiction univariée
    model.compile(optimizer='adam', loss='mean_squared_error')  # Optimiseur et fonction de perte
    return model

# Définir le fichier CSV pour stocker les résultats
csv_filename = 'gru_results.csv'

# Ouvrir le fichier CSV en mode écriture
with open(csv_filename, 'w', newline='') as csv_file:
    # Créer un écrivain CSV
    csv_writer = csv.writer(csv_file)

    # Écrire la ligne d'en-tête
    csv_writer.writerow(['Parameter', 'Prediction Horizon (n)', 'MAE', 'MSE', 'RMSE', 'MAPE'])

    # Parcourir chaque cible (paramètre) et horizon de prédiction
    for param in targets:
        for n in prediction_horizons:
            # Séparer les données en X (caractéristiques) et y (cible)
            X_train = train_data[features]
            y_train = train_data[param]  # La cible est le paramètre actuel
            X_test = test_data[features]
            y_test = test_data[param]  # La cible est le paramètre actuel

            # Créer une nouvelle variable cible pour l'horizon de prédiction
            y_train_shifted = y_train.shift(-n)  # Décaler les valeurs cibles de n jours vers l'avenir

            # Supprimer les lignes contenant des NaN dans la cible décalée
            X_train = X_train[:-n]
            y_train_shifted = y_train_shifted.dropna()

            # Normaliser les données d'entrée (X)
            scaler_X = MinMaxScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)

            # Normaliser la variable cible (y)
            scaler_y = MinMaxScaler()
            y_train_shifted_scaled = scaler_y.fit_transform(np.array(y_train_shifted).reshape(-1, 1))

            # Créer des séquences pour le modèle GRU
            sequence_length = 20  # Longueur des séquences temporelles
            X_train_sequences = []
            y_train_sequences = []

            # Créer des séquences pour l'entraînement
            for i in range(sequence_length, len(X_train_scaled)):
                X_train_sequences.append(X_train_scaled[i - sequence_length:i, :])
                y_train_sequences.append(y_train_shifted_scaled[i, 0])

            # Convertir les séquences en tableaux NumPy
            X_train_sequences = np.array(X_train_sequences)
            y_train_sequences = np.array(y_train_sequences)

            # Reshaper les séquences pour qu'elles correspondent à l'entrée GRU
            X_train_sequences = np.reshape(X_train_sequences, (
                X_train_sequences.shape[0], X_train_sequences.shape[1], X_train_sequences.shape[2]))

            # Créer et entraîner le modèle GRU
            gru_model = create_gru_model((X_train_sequences.shape[1], X_train_sequences.shape[2]))
            gru_model.fit(X_train_sequences, y_train_sequences, epochs=50, batch_size=30)

            # Préparer les données de test pour la prédiction
            X_test_sequences = []

            for i in range(sequence_length, len(X_test_scaled) - n):
                X_test_sequences.append(X_test_scaled[i - sequence_length:i, :])

            X_test_sequences = np.array(X_test_sequences)
            X_test_sequences = np.reshape(X_test_sequences, (
                X_test_sequences.shape[0], X_test_sequences.shape[1], X_test_sequences.shape[2]))

            # Effectuer des prédictions sur les données de test
            gru_predictions_scaled = gru_model.predict(X_test_sequences)
            gru_predictions = scaler_y.inverse_transform(gru_predictions_scaled)

            # Aligner les longueurs de y_test et gru_predictions
            y_test = y_test.iloc[sequence_length:-n].values
            gru_predictions = gru_predictions[:len(y_test)]

            # Supprimer les valeurs NaN des deux ensembles
            nan_indices = np.isnan(y_test)
            y_test = y_test[~nan_indices].flatten()
            gru_predictions = gru_predictions[~nan_indices].flatten()

            # Calculer les métriques d'erreur
            mae = np.mean(np.abs(y_test - gru_predictions))
            mse = np.mean((y_test - gru_predictions) ** 2)
            rmse = np.sqrt(mse)

            def calculate_mape(y_true, y_pred):
                # Fonction pour calculer l'erreur en pourcentage absolu moyen
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            mape = calculate_mape(y_test, gru_predictions)

            # Afficher les métriques pour ce paramètre et cet horizon
            print(f"Parameter: {param}, Prediction Horizon (n): {n} days")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
            print()

            # Écrire les résultats dans le fichier CSV
            csv_writer.writerow([param, n, mae, mse, rmse, mape])

# Afficher un message lorsque les résultats sont enregistrés
print(f"GRU Results saved to {csv_filename}")
