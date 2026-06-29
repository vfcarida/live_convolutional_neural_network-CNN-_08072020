"""
Módulo para realizar o treinamento da rede neural com suporte a MLOps (Callbacks).
"""

import os
import tensorflow as tf
from src.config import Config
from src.data_loader import load_and_preprocess_data
from src.model import build_model

def train_model() -> tf.keras.Model:
    """
    Executa o loop de treinamento da CNN utilizando callbacks para Early Stopping e Checkpoints.
    
    Retorna:
        O modelo treinado.
    """
    config = Config()
    
    # Garantir que o diretório de modelos exista
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    
    print("Carregando dados...")
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
    
    print("Construindo modelo...")
    model = build_model(config)
    
    # Compilação do modelo com Adam optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    # Callbacks para MLOps
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.model_save_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir="./logs")
    ]
    
    print("Iniciando treinamento...")
    history = model.fit(
        train_images, train_labels,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(test_images, test_labels),
        callbacks=callbacks
    )
    
    return model

if __name__ == "__main__":
    train_model()
