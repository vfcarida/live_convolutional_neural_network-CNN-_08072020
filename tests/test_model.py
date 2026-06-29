"""
Testes unitários para a arquitetura do modelo.
"""

import tensorflow as tf
from src.config import Config
from src.model import build_model

def test_model_architecture():
    """
    Verifica se o modelo é construído corretamente com o shape de saída esperado.
    """
    config = Config()
    model = build_model(config)
    
    # Testa a quantidade de classes na última camada
    output_shape = model.layers[-1].output_shape
    assert output_shape == (None, config.num_classes)
    
    # Testa se o modelo tem múltiplas camadas
    assert len(model.layers) >= 5
    
    # Testa integridade da passagem de um tensor falso (Forward Pass)
    dummy_input = tf.random.normal((1, 32, 32, 3))
    predictions = model(dummy_input)
    assert predictions.shape == (1, 10)
