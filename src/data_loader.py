"""
Módulo responsável por carregar e pré-processar o dataset CIFAR-10.
"""

import tensorflow as tf
from tensorflow.keras import datasets
from typing import Tuple
import numpy as np

def load_and_preprocess_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Carrega o dataset CIFAR-10 e normaliza os pixels das imagens.
    
    Retorna:
        Tuple contendo os dados de treino e teste no formato:
        ((train_images, train_labels), (test_images, test_labels))
    """
    # Carregando dados do Keras
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalizando valores de pixel para o intervalo [0, 1]
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    return (train_images, train_labels), (test_images, test_labels)

def get_class_names() -> list:
    """
    Retorna a lista de nomes das classes do CIFAR-10.
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']
