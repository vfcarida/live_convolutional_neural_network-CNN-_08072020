"""
Configurações globais para o pipeline de Machine Learning.
Define os hiperparâmetros e caminhos de arquivo usados em todo o projeto.
"""

from dataclasses import dataclass

@dataclass
class Config:
    """Classe de configuração contendo os hiperparâmetros."""
    batch_size: int = 64
    epochs: int = 15
    learning_rate: float = 0.001
    image_shape: tuple = (32, 32, 3)
    num_classes: int = 10
    model_save_path: str = "models/cnn_model.h5"
