from mlx_audio.tts.utils import quantize_model, load_model
import json
import mlx.core as mx
import os

# Define the model path
model_path = 'prince-canuma/Kokoro-82M'

# Load the model
model = load_model(model_path)

# Quantize to 8-bit
group_size = 64
bits = 8
weights, config = quantize_model(model, model.config, group_size, bits)

# Create the output directory if it doesn't exist
output_dir = './8bit'
os.makedirs(output_dir, exist_ok=True)

# Save the quantized model configuration
config_path = os.path.join(output_dir, 'config.json')
with open(config_path, 'w') as f:
    json.dump(config, f)

# Save the quantized model weights
weights_path = os.path.join(output_dir, 'kokoro-v1_0.safetensors')
mx.save_safetensors(weights_path, weights, metadata={"format": "mlx"})
