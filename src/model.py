"""
Módulo responsável por definir a arquitetura da Rede Neural Convolucional (CNN).
"""

from tensorflow.keras import layers, models
from .config import Config

def build_model(config: Config) -> models.Sequential:
    """
    Constrói a arquitetura sequencial da CNN.
    
    Argumentos:
        config: Objeto de configuração contendo o shape da imagem e número de classes.
        
    Retorna:
        Modelo Keras compilado pronto para treinamento.
    """
    model = models.Sequential()
    
    # Primeira camada convolucional
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=config.image_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Segunda camada convolucional
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Terceira camada convolucional
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Camadas densas para classificação
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(config.num_classes)) # Saída sem ativação softmax (usamos from_logits=True no loss)
    
    return model
