"""
Módulo para carregar o modelo treinado e avaliá-lo no conjunto de teste.
"""

import tensorflow as tf
from src.config import Config
from src.data_loader import load_and_preprocess_data

def evaluate_model():
    """
    Carrega o melhor modelo salvo e o avalia com os dados de teste.
    """
    config = Config()
    
    print("Carregando dados de teste...")
    _, (test_images, test_labels) = load_and_preprocess_data()
    
    print(f"Carregando modelo de {config.model_save_path}...")
    try:
        model = tf.keras.models.load_model(config.model_save_path)
    except OSError:
        print("Erro: Modelo não encontrado. Treine o modelo primeiro.")
        return
    
    print("Avaliando modelo...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    
    print(f"\nAcurácia no Teste: {test_acc:.4f}")
    print(f"Loss no Teste: {test_loss:.4f}")

if __name__ == "__main__":
    evaluate_model()
