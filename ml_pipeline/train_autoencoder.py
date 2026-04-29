import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Fix random seeds so threshold and model weights are reproducible across runs.
# Without this, build/threshold.txt changes every training run, requiring manual
# updates to CRASH_THRESHOLD in firmware and THRESHOLD in the simulator.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Phase 1: Data Synthesis
def generate_synthetic_data(num_samples=10000, seq_length=50):
    """
    Generates synthetic 6-axis IMU data (3 accel, 3 gyro).
    """
    print("Generating synthetic 'normal' riding data (road vibrations)...")
    # Normal data: Small random noise representing engine vibrations and road bumps
    normal_data = np.random.normal(loc=0.0, scale=0.5, size=(num_samples, seq_length, 6))
    
    print("Injecting synthetic 'anomalies' (high-G impacts, accidents)...")
    # Anomalous data: High spikes representing crash or extreme lean
    # We create anomalies by injecting massive spikes in normal data
    anomalous_data = np.copy(normal_data[:int(num_samples * 0.1)])
    for i in range(len(anomalous_data)):
        spike_idx = np.random.randint(10, seq_length - 10)
        # Add high-G forces to accel (e.g. indices 0,1,2) or intense gyro (3,4,5)
        # A massive impact could be 10G+ of acceleration
        anomalous_data[i, spike_idx:spike_idx+5, :] += np.random.normal(loc=15.0, scale=5.0, size=(5, 6))
        
    return normal_data, anomalous_data

# Phase 2: Building Autoencoder
def build_autoencoder(seq_length=50, num_features=6):
    """
    Constructs an unsupervised Autoencoder to learn 'normal' riding patterns.
    """
    model = models.Sequential([
        # Encoder
        layers.Input(shape=(seq_length, num_features)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        
        # Bottleneck (forces the network to compress and learn primary features)
        layers.Dense(16, activation='relu'),
        
        # Decoder (attempts to reconstruct the original signal)
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(seq_length * num_features, activation='linear'),
        layers.Reshape((seq_length, num_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    print("\n--- PHASE 1: DATA SYNTHESIS ---")
    seq_length = 50
    normal_data, anomalous_data = generate_synthetic_data(num_samples=5000, seq_length=seq_length)
    
    # Split into train and test
    split = int(0.8 * len(normal_data))
    x_train = normal_data[:split]
    x_test_normal = normal_data[split:]
    x_test_anomalous = anomalous_data
    
    print("\n--- PHASE 2: BUILDING & TRAINING AUTOENCODER ---")
    autoencoder = build_autoencoder(seq_length=seq_length)
    
    # Train only on NORMAL data so it learns to perfectly reconstruct standard riding
    autoencoder.fit(
        x_train, x_train,
        epochs=15,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )
    
    print("\n--- PHASE 3: THRESHOLD TUNING ---")
    # Predict on test normal data to see the standard reconstruction error
    reconstructions = autoencoder.predict(x_test_normal, verbose=0)

    # Compute per-sample MSE manually to match the firmware's calculation exactly:
    # firmware sums (input[i] - output[i])^2 over all 300 values then divides by 300.
    # tf.keras.losses.mse() on shape (N, 50, 6) returns (N, 50) — averaging over
    # features only — which does NOT match. We reduce over all 300 values instead.
    diff = reconstructions - x_test_normal                          # (N, 50, 6)
    train_loss = np.mean(diff ** 2, axis=(1, 2))                    # (N,)  ← correct

    # Calculate threshold (Mean + 3 Std Deviations above normal loss)
    mean_normal_loss = np.mean(train_loss)
    std_normal_loss = np.std(train_loss)
    threshold = mean_normal_loss + (3 * std_normal_loss)

    print(f"Normal Data MSE Mean:       {mean_normal_loss:.4f}")
    print(f"Anomaly Threshold Computed: {threshold:.4f}")
    print(f"-> Any reconstruction error above this threshold triggers SOS!\n")

    # Test on crash anomalies to see if it catches them
    reconstructions_anom = autoencoder.predict(x_test_anomalous, verbose=0)
    diff_anom = reconstructions_anom - x_test_anomalous             # (N, 50, 6)
    anom_loss = np.mean(diff_anom ** 2, axis=(1, 2))                # (N,)

    mean_anom_loss = np.mean(anom_loss)
    print(f"Crash Anomalous MSE Mean:   {mean_anom_loss:.4f}")
    if mean_anom_loss > threshold:
        print("[SUCCESS] The autoencoder successfully distinguished the crash data!")
    else:
        print("[WARNING] The threshold might need adjustment.")

    # --- Export threshold so firmware and simulator stay in sync ---
    os.makedirs('build', exist_ok=True)
    threshold_path = 'build/threshold.txt'
    with open(threshold_path, 'w') as f:
        f.write(f"{threshold:.6f}\n")
    print(f"\nThreshold saved to {threshold_path}")
    print(f"Update CRASH_THRESHOLD in CrashDetector.cpp and THRESHOLD in script.js to: {threshold:.6f}")

    print("\n--- PHASE 4: EDGE CONVERSION (.tflite) ---")
    # Convert for edge hardware (ESP32/Flutter)
    converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
    tflite_model = converter.convert()
    
    os.makedirs('build', exist_ok=True)
    tflite_path = 'build/crash_detection_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Successfully exported TensorFlow Lite model to: {tflite_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB (Ready for edge device deployment)")

if __name__ == "__main__":
    main()
