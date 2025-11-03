import tensorflow as tf
from tensorflow.keras import models, layers, regularizers

def create_dnn_model(input_dim):
    """Create Deep Neural Network model"""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),
        layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),
        layers.Dense(2, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


    return model

def create_mlp_model(input_dim):
    """Create smaller MLP model"""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.2),
        layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.2),
        layers.Dense(2, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def create_autoencoder_dnn_model(input_dim):
    """Create Autoencoder-based DNN classifier"""
    # Autoencoder
    inputs = tf.keras.Input(shape=(input_dim,))
    # Encoder
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)
    latent = layers.Dense(128, activation="relu")(x)
    # Decoder
    y = layers.Dense(256, activation="relu")(latent)
    y = layers.Dense(512, activation="relu")(y)
    decoded = layers.Dense(input_dim, activation="sigmoid")(y)

    # Autoencoder model
    autoencoder = tf.keras.Model(inputs, decoded, name="autoencoder")
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")

    # Classifier
    classifier_output = layers.Dense(2, activation="softmax")(latent)
    classifier = tf.keras.Model(inputs, classifier_output, name="autoencoder_classifier")
    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return autoencoder, classifier