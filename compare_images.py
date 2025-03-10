import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

def compare_images():
    # Load the model
    model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    model, processor = load(model_path)
    config = load_config(model_path)

    # Prepare input with both images
    images = ["pascal2023.jpg", "woman.jpg"]
    prompt = "You are an expert image analyst. I will show you two images labeled as 'pascal2023.jpg' and 'woman.jpg'. Your task is to analyze them with extreme attention to detail.\n\nFirst image (pascal2023.jpg):\nProvide UNIQUE details about:\n1. The exact pose and position of the person\n2. Their specific clothing (colors, patterns, style)\n3. Any objects they are holding or interacting with\n4. The precise architectural details visible in the background\n\nSecond image (woman.jpg):\nProvide UNIQUE details about:\n1. The exact pose and position of the person\n2. Their specific clothing (colors, patterns, style)\n3. Any objects they are holding or interacting with\n4. The precise architectural details visible in the background\n\nFinally, list THREE specific differences between the images that would make them easily distinguishable to someone else."

    # Apply chat template
    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images=len(images)
    )

    # Generate output
    output = generate(model, processor, formatted_prompt, images, verbose=False)
    print("Image Comparison:")
    print(output)

if __name__ == "__main__":
    compare_images()
