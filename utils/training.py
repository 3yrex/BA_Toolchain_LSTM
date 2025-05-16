import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Korrigierter LRLogger-Callback
class LRLogger(keras.callbacks.Callback):
    def __init__(self):
        super(LRLogger, self).__init__()
        self.lr_values = []
    
    def on_epoch_begin(self, epoch, logs=None):
        # Speichert die Learning Rate als konkreten Wert
        lr = float(keras.backend.get_value(self.model.optimizer.lr))
        self.lr_values.append(lr)
    
    def on_train_end(self, logs=None):
        # Fügt die lr_values zum History-Objekt hinzu
        self.model.history.history['lr'] = self.lr_values

class ModelTrainer:
    """Klasse zum Training von ML-Modellen."""
    
    def __init__(self, model_manager, config):
        """
        Initialisiert den ModelTrainer.
        
        Args:
            model_manager (ModelManager): Manager für das zu trainierende Modell
            config (dict): Konfigurationsdaten
        """
        self.model_manager = model_manager
        self.config = config
        self.model_config = config['model']
        self.epochs = self.model_config['epochs']
        self.batch_size = self.model_config['batch_size']
    
    def train(self, X_train, y_train, X_val=None, y_val=None, dataset_name=None):
        """
        Trainiert das Modell mit den Trainingsdaten.
        
        Args:
            X_train (np.ndarray): Trainings-Eingabedaten
            y_train (np.ndarray): Trainings-Zielwerte
            X_val (np.ndarray, optional): Validierungs-Eingabedaten
            y_val (np.ndarray, optional): Validierungs-Zielwerte
            dataset_name (str, optional): Name des Datensatzes für die Modellspeicherung
            
        Returns:
            dict: Trainingsverlauf
        """
        # Überprüfen, ob Validierungsdaten verfügbar sind
        if X_val is None or y_val is None:
            # Teile den Trainingsdatensatz auf, wenn keine separaten Validierungsdaten übergeben wurden
            val_split = 0.2
            val_count = int(len(X_train) * val_split)
            X_val, y_val = X_train[-val_count:], y_train[-val_count:]
            X_train, y_train = X_train[:-val_count], y_train[:-val_count]
            print(f"Trainingsdaten in {len(X_train)} Trainings- und {len(X_val)} Validierungssequenzen aufgeteilt.")
        
        # Optimierte Datensätze erstellen
        train_dataset = self.model_manager.create_optimized_dataset(
            X_train, y_train, self.batch_size, is_training=True
        )
        val_dataset = self.model_manager.create_optimized_dataset(
            X_val, y_val, self.batch_size, is_training=False
        )
        
        # Early Stopping Callback erstellen
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',      # Überwachen der Validierungsverluste
            patience=100,             # Anzahl der Epochen ohne Verbesserung, bevor das Training gestoppt wird
            mode='min',              # Wir wollen den Verlust minimieren
            restore_best_weights=True,  # Die besten Gewichte wiederherstellen
            verbose=1                # Fortschritt anzeigen
        )
        
        # Learning Rate Scheduler erstellen
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',      # Überwacht den Validierungsverlust
            factor=0.5,              # Reduziert die LR um den Faktor 0.5
            patience=5,             # Wartet 10 Epochen ohne Verbesserung
            min_lr=1e-6,             # Mindestwert für die Learning Rate
            verbose=1                # Gibt Meldungen aus
        )
        
        # LR Logger erstellen - zeichnet die Learning Rate in jeder Epoche auf
        lr_logger = LRLogger()
        
        # Model Checkpoint Callback erstellen
        if dataset_name:
            model_folder_path, _ = self.model_manager.get_model_path(dataset_name)
            os.makedirs(model_folder_path, exist_ok=True)
            checkpoint_path = os.path.join(model_folder_path, 'checkpoint.h5')
            
            checkpoint = keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
            callbacks = [early_stopping, lr_scheduler, lr_logger, checkpoint]
        else:
            callbacks = [early_stopping, lr_scheduler, lr_logger]
        
        # Trainingsstart-Zeit
        start_time = time.time()
        print(f"Training startet mit {self.epochs} Epochen und Batch-Größe {self.batch_size}...")
        print(f"Initial Learning Rate: {self.model_manager.learning_rate}")
        
        # Modell trainieren
        history = self.model_manager.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Trainingsende-Zeit
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training abgeschlossen in {training_time:.2f} Sekunden ({training_time/60:.2f} Minuten).")
        
        # Debug: History-Objekt prüfen
        print("Verfügbare History-Schlüssel:", list(history.history.keys()))
        
        # Speichere den Trainingsverlauf
        if dataset_name:
            model_folder_path, _ = self.model_manager.get_model_path(dataset_name)
            os.makedirs(model_folder_path, exist_ok=True)
            
            # Stelle sicher, dass die LR im History-Objekt ist
            if 'lr' not in history.history and hasattr(lr_logger, 'lr_values'):
                history.history['lr'] = lr_logger.lr_values
            elif 'lr' not in history.history:
                # Fallback: Konstante Learning Rate
                history.history['lr'] = [self.model_manager.learning_rate] * len(history.history['loss'])
                print("Learning Rate aus Model Manager verwendet")
            
            history_path = os.path.join(model_folder_path, 'history.npy')
            np.save(history_path, history.history)
            
            # Speichere den Trainingsverlauf als Plot
            self._plot_training_history(history.history, model_folder_path)
        
        # Speichere das Modell
        if dataset_name:
            self.model_manager.save_model(dataset_name)
            print(f"Training abgeschlossen und Modell gespeichert.")
        
        return history.history
    
    def _plot_training_history(self, history, model_folder_path):
        """
        Erstellt einen Plot des Trainingsverlaufs.
        
        Args:
            history (dict): Trainingsverlauf
            model_folder_path (str): Pfad zum Speichern des Plots
        """
        plt.figure(figsize=(15, 5))
        
        # Plot für den Verlust
        plt.subplot(1, 3, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochen')
        plt.ylabel('Verlust')
        plt.title('Trainingsverlauf - Verlust')
        plt.legend()
        
        # Plot für MAE
        plt.subplot(1, 3, 2)
        if 'mae' in history and 'val_mae' in history:
            plt.plot(history['mae'], label='Training MAE')
            plt.plot(history['val_mae'], label='Validation MAE')
            plt.xlabel('Epochen')
            plt.ylabel('MAE')
            plt.title('Trainingsverlauf - MAE')
            plt.legend()
        elif 'mse' in history and 'val_mse' in history:
            plt.plot(history['mse'], label='Training MSE')
            plt.plot(history['val_mse'], label='Validation MSE')
            plt.xlabel('Epochen')
            plt.ylabel('MSE')
            plt.title('Trainingsverlauf - MSE')
            plt.legend()
        
        # Plot für Learning Rate (auch wenn konstant)
        plt.subplot(1, 3, 3)
        if 'lr' in history:
            # Debug-Ausgabe
            print(f"Learning Rate Werte: {history['lr'][:5]} ... (insgesamt {len(history['lr'])} Werte)")
            
            # Erstelle einen Plot für die Learning Rate
            plt.plot(history['lr'])
            plt.xlabel('Epochen')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Verlauf')
            plt.yscale('log')  # Logarithmische Skala für bessere Visualisierung
        else:
            # Fallback - dieser Code sollte nie ausgeführt werden, da wir 'lr' oben hinzufügen
            print("WARNUNG: Keine Learning Rate Daten gefunden!")
            if hasattr(self, 'model_manager') and hasattr(self.model_manager, 'learning_rate'):
                init_lr = self.model_manager.learning_rate
                epochs = len(history['loss'])
                plt.plot([init_lr] * epochs)
                plt.xlabel('Epochen')
                plt.ylabel('Learning Rate')
                plt.title('Learning Rate Verlauf (konstant)')
                plt.yscale('log')
                print(f"Fallback verwendet: Konstante Learning Rate {init_lr} für {epochs} Epochen")
        
        plt.tight_layout()
        
        # Speichere den Plot
        plot_path = os.path.join(model_folder_path, 'training_history.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Training History Plot gespeichert in: {plot_path}")