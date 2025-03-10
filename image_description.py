import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

def describe_image():
    # Load the model
    model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    model, processor = load(model_path)
    config = load_config(model_path)

    # Prepare input
    image = ["pascal2023.jpg"]  # Local image file
    prompt = "Describe this image in detail."

    # Apply chat template
    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images=len(image)
    )

    # Generate output
    output = generate(model, processor, formatted_prompt, image, verbose=False)
    print("Image Description:")
    print(output)

if __name__ == "__main__":
    describe_image()
