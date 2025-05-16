#!/usr/bin/env python3
import os
import sys
import yaml
import argparse
from utils.data_processor import DataProcessor
from utils.models import ModelManager
from utils.training import ModelTrainer
from utils.evaluation import ModelEvaluator

def load_config(config_file='config.yaml'):
    """Lädt die Konfiguration aus einer YAML-Datei."""
    if not os.path.exists(config_file):
        sys.exit(f"Konfigurationsdatei {config_file} nicht gefunden!")
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Hauptfunktion, die die Toolchain koordiniert."""
    # Argumente parsen
    parser = argparse.ArgumentParser(description='ML Toolchain für Zeitreihenmodelle')
    parser.add_argument('--config', default='config.yaml', help='Pfad zur Konfigurationsdatei')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'all'], default='all',
                        help='Ausführungsmodus: train, evaluate oder all')
    args = parser.parse_args()
    
    # Konfiguration laden
    config = load_config(args.config)
    print(f"Konfiguration aus {args.config} geladen.")
    
    # Datenverarbeitung
    data_processor = DataProcessor(config)
    
    # Überprüfen, ob die Trainings- und Testsequenzen existieren
    if not data_processor.check_sequences_exist():
        if not data_processor.check_master_data_exists():
            sys.exit(f"Die Masterdaten ({config['dataset']['file']}.csv) wurden nicht gefunden. "
                    f"Bitte prüfen Sie den Pfad und den Dateinamen.")
        
        print(f"Trainings- und Testsequenzen werden erstellt...")
        data_processor.create_sequences()
    else:
        print(f"Trainings- und Testsequenzen gefunden.")
    
    # Daten laden
    (X_train, y_train), (X_test, y_test) = data_processor.load_sequences()
    
    # Modellerstellung und -verwaltung
    model_manager = ModelManager(config)
    
    # Bestimme input_shape basierend auf dem Modelltyp
    input_shape = (None, X_train.shape[2]) if config['model']['type'].upper() == 'LSTM' else (X_train.shape[1], X_train.shape[2])
    
    # Modellpfade für später
    dataset_name = config['dataset']['file']
    model_folder_path, model_path = model_manager.get_model_path(dataset_name)
    
    # Modell erstellen oder laden je nach Modus
    if args.mode in ['train', 'all']:
        # Überprüfen, ob das Modell bereits trainiert wurde
        if os.path.exists(model_path) and args.mode == 'all':
            print(f"Trainiertes Modell unter {model_path} gefunden.")
            print(f"Überspringe Training und lade das bestehende Modell...")
            model_manager.load_model(model_path)
        else:
            print(f"Erstelle und trainiere ein neues {config['model']['type']}-Modell...")
            model_manager.create_model(input_shape)
            
            # Modelltrainer
            trainer = ModelTrainer(model_manager, config)
            trainer.train(X_train, y_train, X_test, y_test, dataset_name)
    
    if args.mode in ['evaluate', 'all']:
        # Wenn wir nur evaluieren, muss das Modell geladen werden
        if args.mode == 'evaluate' or not model_manager.model:
            print(f"Lade Modell für Evaluation...")
            model_manager.load_model(model_path)
        
        # Modell evaluieren
        evaluator = ModelEvaluator(model_manager)
        results = evaluator.evaluate(X_test, y_test)
        evaluator.visualize(results, model_folder_path)
    
    print("Toolchain erfolgreich abgeschlossen!")

if __name__ == "__main__":
    main()