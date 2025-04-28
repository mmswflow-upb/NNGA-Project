import os
import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import random


DATASET_PATH = './Data/genres_original'   
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']


def add_noise(data, noise_factor=None):
    if noise_factor is None:
        noise_factor = random.uniform(0.001, 0.01)
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def time_stretch(data, rate=None):
    if rate is None:
        rate = random.uniform(0.8, 1.2)
    return librosa.effects.time_stretch(data, rate)

def pitch_shift(data, sr, n_steps=None):
    if n_steps is None:
        n_steps = random.randint(-3, 3)
    return librosa.effects.pitch_shift(data, sr, n_steps=n_steps)


def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    variations = [y, add_noise(y), time_stretch(y), pitch_shift(y, sr)]

    features = []
    for variation in variations:
        S = librosa.feature.melspectrogram(y=variation, sr=sr, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)
        if S_DB.shape[1] >= 660:
            S_DB = S_DB[:, :660]  # Fix width
            features.append(S_DB)

    return features


X = []
y = []

for label, genre in enumerate(genres):
    genre_folder = os.path.join(DATASET_PATH, genre)
    for filename in os.listdir(genre_folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(genre_folder, filename)
            feature_list = extract_features(file_path)
            for feature in feature_list:
                X.append(feature)
                y.append(label)

X = np.array(X)
y = np.array(y)

# Reshape for CNN (samples, height, width, channels)
X = X[..., np.newaxis]

print(f"X shape: {X.shape}, y shape: {y.shape}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(genres), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))


test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
