#include "CrashDetector.h"

// Define the threshold we organically found in Phase 3
#define CRASH_THRESHOLD 0.30f

CrashDetector::CrashDetector() : model(nullptr), interpreter(nullptr), input(nullptr), output(nullptr), last_mse(0), initialized(false) {}

bool CrashDetector::begin() {
    // Prevent re-initialization — static interpreter cannot be safely rebuilt.
    // Call begin() only once at startup.
    if (initialized) return true;
    // Phase 1: Map the model into a usable data structure pointing to PROGMEM
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        // Model Schema mismatch
        return false;
    }

    // Phase 2: Pull in all the mathematical operations the autoencoder uses 
    // We only need basic operations since it's an un-optimized Dense network
    static tflite::MicroMutableOpResolver<5> micro_op_resolver;
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddRelu();
    micro_op_resolver.AddReshape();

    // Phase 3: Build an interpreter to orchestrate memory and flow
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize, nullptr);
    interpreter = &static_interpreter;

    // Phase 4: Allocate memory out of standard RAM to load the matrices
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        // Insufficient RAM or Memory leak
        return false;
    }

    // Lock pointers to the input and output nodes of our neural network
    input = interpreter->input(0);
    output = interpreter->output(0);

    initialized = true;
    return true;
}

bool CrashDetector::isCrashDetected(float imu_data[50][6]) {
    // If the ML engine failed to boot up, default safe to prevent infinite SOS loops
    if (!interpreter || !input || !output) return false;

    // Step 1: Flatten the multi-dimensional 50x6 IMU array into the model's 1D input tensor buffer.
    // Also keep a local copy so we can compute MSE after Invoke() — the input tensor
    // buffer may be overwritten internally by TFLite during inference.
    float input_copy[300];
    int index = 0;
    for (int i = 0; i < 50; i++) {
        for (int j = 0; j < 6; j++) {
            input_copy[index] = imu_data[i][j];
            input->data.f[index] = imu_data[i][j];
            index++;
        }
    }

    // Step 2: Run Inference! The Autoencoder attempts to correctly rebuild the waveform.
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        return false; // Hardware crash or math fault
    }

    // Step 3: Compute the Mean Squared Error (MSE) between the original live IMU
    // and the AI's rebuilt state. Use input_copy (not input->data.f) because
    // TFLite may overwrite the input buffer during Invoke().
    float mse = 0.0f;
    for (int i = 0; i < 300; i++) { // 50 timestamps * 6 axes = 300 total points
        float diff = input_copy[i] - output->data.f[i];
        mse += (diff * diff);
    }
    mse /= 300.0f; // Calculate proper statistical mean
    
    // Store it for optional bluetooth transmission or debouncing
    last_mse = mse;
    
    // Step 4: Final Trigger Hook! Check against our threshold parameter
    if (mse > CRASH_THRESHOLD) {
        return true; // We encountered chaotic physics - Trigger SOS!
    }
    
    return false; // Proceed normally
}

float CrashDetector::getLastMSE() const {
    return last_mse; // Grab the realtime error metric
}
