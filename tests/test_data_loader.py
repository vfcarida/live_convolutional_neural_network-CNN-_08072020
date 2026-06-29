"""
Testes unitários para o carregamento e pré-processamento de dados.
"""

from src.data_loader import load_and_preprocess_data, get_class_names

def test_data_shapes_and_normalization():
    """
    Verifica se os dados carregados possuem o formato e normalização corretos.
    """
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
    
    # Testa os formatos
    assert len(train_images.shape) == 4
    assert train_images.shape[1:] == (32, 32, 3)
    assert train_images.shape[0] == train_labels.shape[0]
    
    # Testa a normalização (pixels entre 0 e 1)
    assert train_images.max() <= 1.0
    assert train_images.min() >= 0.0

def test_class_names():
    """
    Verifica se as 10 classes do CIFAR-10 estão corretas.
    """
    classes = get_class_names()
    assert len(classes) == 10
    assert "airplane" in classes
    assert "truck" in classes
