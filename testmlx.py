import soundfile as sf
from IPython.display import Audio, display
from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.utils import load_model

# Initialize the model
model_id = 'prince-canuma/Kokoro-82M'
model = load_model(model_id)

# Create a pipeline with American English
pipeline = KokoroPipeline(lang_code='a', model=model, repo_id=model_id)

# Generate audio
text = "pascal-maker production!"
for _, _, audio in pipeline(text, voice='af_heart', speed=1, split_pattern=r'\n+'):
    # Display audio in notebook (if applicable)
    display(Audio(data=audio, rate=24000, autoplay=False))
    
    # Save audio to file
    sf.write('audio.wav', audio[0], 24000)
