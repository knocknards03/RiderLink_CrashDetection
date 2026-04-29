import os
import binascii
import sys

def convert_tflite_to_c_array(tflite_path, output_path, array_name="g_model"):
    """
    Reads a .tflite file and converts it into a C++ header containing a byte array.
    """
    # Validate input file exists before attempting to open it
    if not os.path.isfile(tflite_path):
        print(f"[ERROR] TFLite model not found at '{tflite_path}'.")
        print("        Run ml_pipeline/train_autoencoder.py first to generate the model.")
        sys.exit(1)

    print(f"Reading {tflite_path}...")
    
    with open(tflite_path, 'rb') as f:
        tflite_data = f.read()

    model_size_kb = len(tflite_data) / 1024
    print(f"Model size: {model_size_kb:.2f} KB")
    # Warn if the model is larger than a typical ESP32 PROGMEM budget (~3 MB safe limit)
    if model_size_kb > 3072:
        print(f"[WARNING] Model is {model_size_kb:.1f} KB — this may exceed ESP32 PROGMEM limits.")
    
    hex_data = binascii.hexlify(tflite_data).decode('utf-8')
    
    # Format the hex string into 0xXX comma-separated bytes, 12 per line
    formatted_hex = []
    for i in range(0, len(hex_data), 2):
        formatted_hex.append("0x" + hex_data[i:i+2])
        
    line_length = 12
    formatted_lines = []
    for i in range(0, len(formatted_hex), line_length):
        formatted_lines.append("  " + ", ".join(formatted_hex[i:i+line_length]))
        
    c_array = f"// Auto-generated from {os.path.basename(tflite_path)}\n"
    c_array += f"// Size: {len(tflite_data)} bytes\n\n"
    c_array += "#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n"
    c_array += f"const unsigned int {array_name}_len = {len(tflite_data)};\n"
    c_array += f"const unsigned char {array_name}[] = {{\n"
    c_array += ",\n".join(formatted_lines)
    c_array += "\n};\n\n"
    c_array += "#endif // MODEL_DATA_H\n"
    
    # Ensure the output directory exists.
    # os.path.dirname() returns '' for a bare filename — guard against that.
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Writing C array to {output_path}...")
    with open(output_path, 'w') as f:
        f.write(c_array)
        
    print("Conversion complete! Ready for ESP32 PROGMEM.")

if __name__ == "__main__":
    tflite_model_path = "build/crash_detection_model.tflite"
    output_header_path = "firmware_integration/model_data.h"
    
    convert_tflite_to_c_array(tflite_model_path, output_header_path)
