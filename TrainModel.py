import os
import random

import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split

import keras
from keras import layers, models, utils

# ─── Reproducibility ─────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
utils.set_random_seed(42)

# ─── Constants ──────────────────────────────────────────────────────────────────
DATASET_PATH = './Data/genres_original'
GENRES = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]

# ─── Data Augmentation ──────────────────────────────────────────────────────────
def add_noise(y, noise_factor=None):
    if noise_factor is None:
        noise_factor = random.uniform(0.001, 0.01)
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def time_stretch(y, rate=None):
    if rate is None:
        rate = random.uniform(0.8, 1.2)
    return librosa.effects.time_stretch(y, rate)

def pitch_shift(y, sr, n_steps=None):
    if n_steps is None:
        n_steps = random.randint(-3, 3)
    return librosa.effects.pitch_shift(y, sr, n_steps=n_steps)

# ─── Feature Extraction ─────────────────────────────────────────────────────────
def extract_features(file_path, max_frames=660, n_mels=128):
    y, sr = librosa.load(file_path, duration=30)
    variants = [
        y,
        add_noise(y),
        time_stretch(y),
        pitch_shift(y, sr),
    ]
    feats = []
    for var in variants:
        S    = librosa.feature.melspectrogram(y=var, sr=sr, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        if S_db.shape[1] >= max_frames:
            feats.append(S_db[:, :max_frames])
    return feats

# ─── Data Loading ───────────────────────────────────────────────────────────────
def load_dataset():
    X, y = [], []
    for label, genre in enumerate(GENRES):
        folder = os.path.join(DATASET_PATH, genre)
        for fname in os.listdir(folder):
            if not fname.endswith('.wav'):
                continue
            path = os.path.join(folder, fname)
            for feat in extract_features(path):
                X.append(feat)
                y.append(label)

    X = np.array(X)[..., np.newaxis]  # add channel dim
    y = np.array(y)
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ─── Model Definition ──────────────────────────────────────────────────────────
def build_model(input_shape, n_classes=len(GENRES)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    X_train, X_test, y_train, y_test = load_dataset()

    model = build_model(input_shape=X_train.shape[1:])
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    main()
