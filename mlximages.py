from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)

images = ["womenholdingateapot.jpg", "pascal2023.jpg"]
prompt = "Compare these two images."

formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=len(images)
)

output = generate(model, processor, formatted_prompt, images, verbose=False)
print(output)