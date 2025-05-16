import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras

class ModelEvaluator:
    """Klasse zur Evaluierung von trainierten Modellen."""
    
    def __init__(self, model_manager):
        """
        Initialisiert den ModelEvaluator.
        
        Args:
            model_manager (ModelManager): Manager für das zu evaluierende Modell
        """
        self.model_manager = model_manager
    
    def evaluate(self, X_test, y_test):
        """
        Evaluiert das Modell mit den Testdaten.
        
        Args:
            X_test (np.ndarray): Test-Eingabedaten
            y_test (np.ndarray): Test-Zielwerte
            
        Returns:
            dict: Evaluierungsmetriken
        """
        # Erstelle optimierten Datensatz für die Evaluation
        test_dataset = self.model_manager.create_optimized_dataset(
            X_test, y_test, 
            self.model_manager.model_config['batch_size'], 
            is_training=False
        )
        
        # Evaluiere das Modell
        metrics = self.model_manager.model.evaluate(test_dataset, return_dict=True)
        print(f"Evaluierungsmetriken: {metrics}")
        
        # Erstelle Vorhersagen
        predictions = self.model_manager.model.predict(X_test)
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'y_true': y_test
        }
    
    def visualize(self, evaluation_results, model_folder_path=None):
        """
        Visualisiert nur den Trainingsverlauf.
        
        Args:
            evaluation_results (dict): Ergebnisse der evaluate-Methode
            model_folder_path (str, optional): Pfad zum Speichern der Plots
        """
        # Prüfe, ob ein Modellordner angegeben wurde und ob die History-Datei existiert
        if model_folder_path and os.path.exists(os.path.join(model_folder_path, 'history.npy')):
            # Lade den Trainingsverlauf
            history = np.load(os.path.join(model_folder_path, 'history.npy'), allow_pickle=True).item()
            
            # Erstelle einen Plot für den Trainingsverlauf
            plt.figure(figsize=(10, 6))
            
            # Verlust plotten
            plt.subplot(1, 2, 1)
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochen')
            plt.ylabel('Verlust')
            plt.legend()
            plt.title('Trainingsverlauf - Verlust')
            
            # MAE plotten, falls vorhanden
            plt.subplot(1, 2, 2)
            if 'mae' in history and 'val_mae' in history:
                plt.plot(history['mae'], label='Training MAE')
                plt.plot(history['val_mae'], label='Validation MAE')
                plt.xlabel('Epochen')
                plt.ylabel('MAE')
                plt.legend()
                plt.title('Trainingsverlauf - MAE')
            elif 'mse' in history and 'val_mse' in history:
                plt.plot(history['mse'], label='Training MSE')
                plt.plot(history['val_mse'], label='Validation MSE')
                plt.xlabel('Epochen')
                plt.ylabel('MSE')
                plt.legend()
                plt.title('Trainingsverlauf - MSE')
            
            plt.tight_layout()
            
            # Speichere den Plot, wenn ein Pfad angegeben ist
            if model_folder_path:
                plot_path = os.path.join(model_folder_path, 'training_history.png')
                plt.savefig(plot_path)
                print(f"Trainingsplot gespeichert in: {plot_path}")
            
            plt.show()
        else:
            print("Kein Trainingsverlauf gefunden. Keine Plots erstellt.")
    
    def load_and_evaluate(self, model_path, X_test, y_test, model_folder_path=None):
        """
        Lädt ein Modell und evaluiert es.
        
        Args:
            model_path (str): Pfad zur Modelldatei
            X_test (np.ndarray): Test-Eingabedaten
            y_test (np.ndarray): Test-Zielwerte
            model_folder_path (str, optional): Pfad zum Modellordner
            
        Returns:
            dict: Evaluierungsergebnisse
        """
        # Lade das Modell
        self.model_manager.load_model(model_path)
        
        # Evaluiere das Modell
        results = self.evaluate(X_test, y_test)
        
        # Visualisiere nur den Trainingsverlauf
        self.visualize(results, model_folder_path)
        
        return results