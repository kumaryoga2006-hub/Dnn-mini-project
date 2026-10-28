"""CNN feature extractor and recurrent temporal classifier."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_model(sequence_length: int, num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(sequence_length, 42, 3), name="hand_landmarks")
    frame_features = layers.TimeDistributed(
        layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")
    )(inputs)
    frame_features = layers.TimeDistributed(layers.BatchNormalization())(
        frame_features
    )
    frame_features = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(
        frame_features
    )
    frame_features = layers.TimeDistributed(
        layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")
    )(frame_features)
    frame_features = layers.TimeDistributed(layers.GlobalAveragePooling1D())(
        frame_features
    )
    temporal_features = layers.LSTM(128, return_sequences=True)(frame_features)
    temporal_features = layers.Dropout(0.35)(temporal_features)
    temporal_features = layers.LSTM(64)(temporal_features)
    temporal_features = layers.Dense(64, activation="relu")(temporal_features)
    temporal_features = layers.Dropout(0.3)(temporal_features)
    outputs = layers.Dense(num_classes, activation="softmax", name="sign")(temporal_features)
    model = keras.Model(inputs=inputs, outputs=outputs, name="sign_cnn_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

