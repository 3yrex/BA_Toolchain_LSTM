import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

class ModelManager:
    """Klasse zur Verwaltung des LSTM-Modells."""
    
    def __init__(self, config):
        """
        Initialisiert den ModelManager mit der angegebenen Konfiguration.
        
        Args:
            config (dict): Konfigurationsdaten aus config.yaml
        """
        self.config = config
        self.model_config = config['model']
        # Wir erzwingen hier den Modelltyp LSTM, unabhängig von der Konfiguration
        self.model_type = 'LSTM'
        self.learning_rate = self.model_config['learning_rate']
        self.loss_function = self.model_config['loss_function']
        self.neurons = self.model_config['neurons_per_layer']
        self.hidden_layers = self.model_config['hidden_layers']
        self.model = None
        
        # Konfiguriere GPU, wenn verfügbar
        if config.get('gpu', {}).get('enabled', False):
            self._configure_gpu(config['gpu'])
    
    def _configure_gpu(self, gpu_config):
        """
        Konfiguriert die GPU für optimale Performance.
        
        Args:
            gpu_config (dict): GPU-Konfigurationsparameter
        """
        # GPU-Geräte auflisten
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                # Dynamisches Speicherwachstum aktivieren, wenn konfiguriert
                if gpu_config.get('memory_growth', True):
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                
                # Mixed-Precision-Training aktivieren, wenn konfiguriert
                if gpu_config.get('mixed_precision', True) and tf.__version__ >= '2.4.0':
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    print("Mixed Precision Training aktiviert (FP16)")
                
                # Thread-Konfiguration
                num_threads = gpu_config.get('threads', 16)
                tf.config.threading.set_inter_op_parallelism_threads(num_threads)
                tf.config.threading.set_intra_op_parallelism_threads(num_threads)
                os.environ['OMP_NUM_THREADS'] = str(num_threads)
                os.environ['TF_NUM_INTEROP_THREADS'] = str(num_threads)
                
                print(f"GPU erfolgreich konfiguriert: {len(gpus)} GPU(s) gefunden")
            except RuntimeError as e:
                print(f"GPU-Konfigurationsfehler: {e}")
        else:
            print("Keine GPU gefunden. CPU wird verwendet.")
    
    def create_model(self, input_shape):
        """
        Erstellt ein LSTM-Modell.
        
        Args:
            input_shape (tuple): Form der Eingabedaten
            
        Returns:
            ModelManager: Selbst für Methodenverkettung
        """
        self._create_lstm_model(input_shape)
        
        # Modell kompilieren
        self.compile_model()
        
        return self
    
    def _create_lstm_model(self, input_shape):
        """
        Erstellt ein LSTM-Modell.
        
        Args:
            input_shape (tuple): Form der Eingabedaten
        """
        model = keras.Sequential()
        model.add(keras.Input(shape=input_shape))
        
        for i in range(self.hidden_layers):
            return_sequences = (i < self.hidden_layers - 1) or self.hidden_layers == 1
            model.add(keras.layers.LSTM(
                self.neurons,
                activation='tanh',
                return_sequences=return_sequences,
                use_bias=True,
                implementation=2  # Schnellere Implementierung für GPUs
            ))
            model.add(keras.layers.BatchNormalization())
        
        model.add(keras.layers.Dense(1, activation='linear', use_bias=True))
        self.model = model
    
    def compile_model(self):
        """Kompiliert das Modell mit den konfigurierten Parametern."""
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            epsilon=1e-7,
            amsgrad=False,
            beta_1=0.9,
            beta_2=0.999
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=self.loss_function,
            metrics=['mae', 'mse']
        )
        
        # Modellzusammenfassung ausgeben
        self.model.summary()
        
        return self
    
    def create_optimized_dataset(self, X, y, batch_size, is_training=True):
        """
        Erstellt einen optimierten TensorFlow-Datensatz für Training oder Evaluation.
        
        Args:
            X (np.ndarray): Eingabedaten
            y (np.ndarray): Zieldaten
            batch_size (int): Batch-Größe
            is_training (bool): Ob der Datensatz für das Training ist
            
        Returns:
            tf.data.Dataset: Optimierter Datensatz
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if is_training:
            # Durchmischen für bessere Generalisierung
            dataset = dataset.shuffle(buffer_size=10000)
        
        # Cache und Prefetching für bessere Performance
        dataset = dataset.cache()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def save_model(self, dataset_name):
        """
        Speichert das LSTM-Modell in verschiedenen Formaten.
        
        Args:
            dataset_name (str): Name des Datensatzes für die Benennung
            
        Returns:
            tuple: (model_folder_path, h5_path) Pfade zum Modellordner und zur H5-Datei
        """
        if self.model is None:
            raise ValueError("Kein Modell zum Speichern vorhanden. Erstellen Sie zuerst ein Modell.")
        
        # Erstelle dynamischen Ordnernamen (ohne Learning Rate)
        dynamic_folder_name = (
            f"{dataset_name}_lstm_ly{self.hidden_layers}_n{self.neurons}_"
            f"ep{self.model_config['epochs']}_b{self.model_config['batch_size']}_"
            f"ls{self.loss_function}"
        )
        
        # Erstelle Pfade
        model_folder_path = os.path.join('models', dynamic_folder_name)
        os.makedirs(model_folder_path, exist_ok=True)
        
        # Speichere im H5-Format (ohne Learning Rate im Namen)
        h5_filename = f"{dataset_name}_lstm_ly{self.hidden_layers}_n{self.neurons}_ep{self.model_config['epochs']}_b{self.model_config['batch_size']}_ls{self.loss_function}.h5"
        h5_path = os.path.join(model_folder_path, h5_filename)
        self.model.save(h5_path)
        print(f"Modell im H5-Format gespeichert: {h5_path}")
        
        # Speichere im SavedModel-Format
        saved_model_path = os.path.join(model_folder_path, f'{dataset_name}_savedmodel')
        self.model.save(saved_model_path, save_format='tf')
        print(f"Modell im SavedModel-Format gespeichert: {saved_model_path}")
        
        # Speichere Gewichte im MAT-Format für Simulink
        from scipy.io import savemat
        mat_filename = os.path.join(model_folder_path, "lstm_net.mat")
        weights = self.model.get_weights()
        weights_dict = {'weights': np.array(weights, dtype=object)}
        savemat(mat_filename, weights_dict)
        print(f"Modellgewichte im MAT-Format für Simulink gespeichert: {mat_filename}")
        
        return model_folder_path, h5_path
    
    def load_model(self, model_path):
        """
        Lädt ein gespeichertes Modell.
        
        Args:
            model_path (str): Pfad zur Modelldatei
            
        Returns:
            ModelManager: Selbst für Methodenverkettung
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Keine Datei oder kein Verzeichnis unter {model_path} gefunden")
        
        self.model = keras.models.load_model(model_path)
        print(f"Modell aus {model_path} geladen")
        
        # Modellzusammenfassung ausgeben
        self.model.summary()
        
        return self
    
    def get_model_path(self, dataset_name):
        """
        Gibt den Pfad zum Modell basierend auf den Konfigurationsparametern zurück.
        
        Args:
            dataset_name (str): Name des Datensatzes
            
        Returns:
            tuple: (model_folder_path, model_path) Pfade zum Modellordner und zur Modelldatei
        """
        # Erstelle dynamischen Ordnernamen (ohne Learning Rate)
        dynamic_folder_name = (
            f"{dataset_name}_lstm_ly{self.hidden_layers}_n{self.neurons}_"
            f"ep{self.model_config['epochs']}_b{self.model_config['batch_size']}_"
            f"ls{self.loss_function}"
        )
        
        # Erstelle Pfade
        model_folder_path = os.path.join('models', dynamic_folder_name)
        model_filename = f"{dataset_name}_lstm_ly{self.hidden_layers}_n{self.neurons}_ep{self.model_config['epochs']}_b{self.model_config['batch_size']}_ls{self.loss_function}.h5"
        model_path = os.path.join(model_folder_path, model_filename)
        
        return model_folder_path, model_path