import os
import pandas as pd
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split

class DataProcessor:
    """Klasse zur Verarbeitung von Zeitreihendaten für ML-Modelle."""
    
    def __init__(self, config):
        """
        Initialisiert den DataProcessor mit der angegebenen Konfiguration.
        
        Args:
            config (dict): Konfigurationsdaten aus config.yaml
        """
        self.config = config
        self.dataset_file = config['dataset']['file']
        self.data_path = config['dataset']['path']
        self.sorted_path = config['dataset']['sorted_path']
        
        # Sicherstellen, dass die Verzeichnisse existieren
        os.makedirs(self.sorted_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
    
    def check_sequences_exist(self):
        """Überprüft, ob die Trainings- und Testsequenzen bereits existieren."""
        train_file = os.path.join(self.sorted_path, f"{self.dataset_file}_train_sequences.mat")
        test_file = os.path.join(self.sorted_path, f"{self.dataset_file}_test_sequences.mat")
        return os.path.isfile(train_file) and os.path.isfile(test_file)
    
    def check_master_data_exists(self):
        """Überprüft, ob die Master-Datendatei existiert."""
        csv_file = os.path.join(self.data_path, f"{self.dataset_file}.csv")
        return os.path.isfile(csv_file)
    
    def create_sequences(self):
        """
        Erstellt Trainings- und Testsequenzen aus den Masterdaten.
        
        Returns:
            tuple: Pfade zu den erstellten Trainings- und Testsequenzdateien
        """
        # Pfad zur Master_Data.csv Datei
        csv_file = os.path.join(self.data_path, f"{self.dataset_file}.csv")
        
        if not self.check_master_data_exists():
            raise FileNotFoundError(f"{csv_file} nicht gefunden. Bitte überprüfen Sie den Datensatzpfad.")
        
        print(f"Lade Daten aus {csv_file}...")
        
        # Die Master Data wird geladen
        master_data = pd.read_csv(csv_file)
        
        # Die eindeutigen OPs werden extrahiert
        unique_ops = master_data['OP'].unique()
        
        # Filtern der Daten für alle OPs ohne Einschränkung des Zeitbereichs
        filtered_data = master_data[master_data['OP'].isin(unique_ops)]
        
        # Bestimmen des maximalen Werts von time
        sequence_length = filtered_data['time'].max()
        
        # Entfernen der OP- und time-Spalte
        filtered_data = filtered_data.drop(columns=['OP', 'time'])
        
        print(f"Erstelle Sequenzen mit Länge {sequence_length}...")
        
        # Erstellen der Sequenzen
        sequences = self._create_sequence_arrays(filtered_data, sequence_length)
        
        # Aufteilen der Sequenzen in Trainings- und Testdatensätze
        train_sequences, test_sequences = train_test_split(
            sequences, test_size=0.3, random_state=42
        )
        
        # Speichern der Sequenzen
        train_file = os.path.join(self.sorted_path, f"{self.dataset_file}_train_sequences.mat")
        test_file = os.path.join(self.sorted_path, f"{self.dataset_file}_test_sequences.mat")
        
        savemat(train_file, {'train_sequences': train_sequences})
        savemat(test_file, {'test_sequences': test_sequences})
        
        print(f"Sequenzen erstellt und gespeichert:")
        print(f"Train sequences shape: {train_sequences.shape}")
        print(f"Test sequences shape: {test_sequences.shape}")
        
        return train_file, test_file
    
    def _create_sequence_arrays(self, data, sequence_length):
        """
        Erstellt Sequenzarrays aus den gefilterten Daten.
        
        Args:
            data (pd.DataFrame): Die gefilterten Daten
            sequence_length (int): Die Länge der zu erstellenden Sequenzen
            
        Returns:
            np.ndarray: Array von Sequenzen
        """
        sequences = []
        for i in range(0, len(data), sequence_length):
            seq = data.iloc[i:i+sequence_length].values
            if len(seq) == sequence_length:
                sequences.append(seq)
        
        return np.array(sequences)
    
    def load_sequences(self):
        """
        Lädt die Trainings- und Testsequenzen aus .mat-Dateien.
        
        Returns:
            tuple: (X_train, y_train), (X_test, y_test) Arrays für Training und Evaluation
        """
        if not self.check_sequences_exist():
            train_file, test_file = self.create_sequences()
        else:
            train_file = os.path.join(self.sorted_path, f"{self.dataset_file}_train_sequences.mat")
            test_file = os.path.join(self.sorted_path, f"{self.dataset_file}_test_sequences.mat")
        
        print(f"Lade Sequenzen aus {train_file} und {test_file}...")
        
        # Laden der Trainingssequenzen
        X_train = loadmat(train_file)['train_sequences']
        y_train = X_train[:, :, 0]  # Erste Spalte als Zielwert
        y_train = np.expand_dims(y_train, axis=-1)
        X_train = np.delete(X_train, 0, axis=2)  # Erste Spalte entfernen
        
        # Laden der Testsequenzen
        X_test = loadmat(test_file)['test_sequences']
        y_test = X_test[:, :, 0]  # Erste Spalte als Zielwert
        y_test = np.expand_dims(y_test, axis=-1)
        X_test = np.delete(X_test, 0, axis=2)  # Erste Spalte entfernen
        
        print(f"Daten geladen:")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        return (X_train, y_train), (X_test, y_test)
    
    def get_sequence_file_paths(self):
        """
        Gibt die Pfade zu den Trainings- und Testsequenzdateien zurück.
        
        Returns:
            tuple: (train_file, test_file) Pfade zu den Sequenzdateien
        """
        train_file = os.path.join(self.sorted_path, f"{self.dataset_file}_train_sequences.mat")
        test_file = os.path.join(self.sorted_path, f"{self.dataset_file}_test_sequences.mat")
        return train_file, test_file