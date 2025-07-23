import os
import uuid
import random
import numpy as np
from flask import Flask, request, jsonify, send_file
from gtts import gTTS
from moviepy.editor import (VideoClip, TextClip, CompositeVideoClip, 
                           concatenate_videoclips, AudioFileClip, ImageClip, ColorClip)
from moviepy.audio.AudioClip import AudioArrayClip, CompositeAudioClip
from PIL import Image
import openai
import requests
from io import BytesIO
import tempfile
from textblob import TextBlob  # For simple sentiment analysis

app = Flask(__name__)

# Configuration
OPENAI_API_KEY = 'your-openai-api-key-here'  # Replace with your actual OpenAI API key
openai.api_key = OPENAI_API_KEY
API_KEYS = {}  # In-memory storage for generated API keys

# AI Helper: Expand text using OpenAI
def ai_expand_text(text, max_length=500):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Expand this text into a detailed story/script up to {max_length} words: {text}"}]
    )
    return response.choices[0].message.content

# AI Helper: Detect emotion
def ai_detect_emotion(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0.3: return "happy"
    elif sentiment < -0.3: return "sad"
    else: return "neutral"

# AI Helper: Generate image prompts
def ai_generate_image_prompts(script, num_images, style="realistic like Veo 3"):
    prompts = []
    for i in range(num_images):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Generate a detailed image prompt for scene {i+1} from this script in {style} style: {script[:200]}"}]
        )
        prompts.append(response.choices[0].message.content)
    return prompts

# Helper: Generate background music with emotion
def generate_background_music(duration_sec, emotion="neutral"):
    sample_rate = 44100
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    if emotion == "happy":
        freq = 523.25  # Higher pitch
    elif emotion == "sad":
        freq = 349.23  # Lower pitch
    else:
        freq = 440
    note = np.sin(2 * np.pi * freq * t)
    audio = note * 0.3
    audio_stereo = np.array([audio, audio]).T
    return AudioArrayClip(audio_stereo, fps=sample_rate)

# Helper: Generate speech audio with emotion modulation
def generate_speech(text, language='en', emotion="neutral"):
    if language not in ['en', 'hi', 'te']:
        raise ValueError("Unsupported language.")
    tts = gTTS(text=text, lang=language, slow=(emotion == "sad"))
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        speech_path = tmp_file.name
        tts.save(speech_path)
    return speech_path

# Helper: Generate images from AI prompts
def generate_images_from_text(prompts, num_images=3):
    images = []
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    for prompt in prompts:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        img_data = requests.get(image_url).content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(img_data)
            images.append(tmp_file.name)
    return images

# Helper: Create video
def create_video(text, language='en', duration_multiplier=1, expand_text=False, style="realistic", num_images=3):
    emotion = ai_detect_emotion(text)
    if expand_text:
        text = ai_expand_text(text)
    script = text
    total_duration = len(text) / 20 * duration_multiplier  # Estimate duration
    speech_path = generate_speech(text, language, emotion)
    speech_audio = AudioFileClip(speech_path)
    music_clip = generate_background_music(total_duration, emotion).volumex(0.3)
    combined_audio = CompositeAudioClip([speech_audio, music_clip])
    
    prompts = ai_generate_image_prompts(script, num_images, style)
    image_paths = generate_images_from_text(prompts, num_images)
    
    clips = []
    img_duration = total_duration / len(image_paths)
    for img_path in image_paths:
        img_clip = ImageClip(img_path, duration=img_duration).resize((1920, 1080)).fadein(0.5).fadeout(0.5)
        text_overlay = TextClip(text[:50], fontsize=40, color='white').set_duration(img_duration).set_position('bottom')
        composite = CompositeVideoClip([img_clip, text_overlay])
        clips.append(composite)
    
    final_video = concatenate_videoclips(clips).set_audio(combined_audio)
    video_path = f'video_{uuid.uuid4()}.mp4'
    final_video.write_videofile(video_path, fps=24, codec='libx264', audio_codec='aac')
    
    os.remove(speech_path)
    for p in image_paths:
        os.remove(p)
    return video_path

# API Endpoints (same as before, with added params for features)
@app.route('/generate_api_key', methods=['POST'])
def generate_api_key():
    data = request.get_json() or {}
    user_id = data.get('user_id', str(uuid.uuid4()))
    api_key = str(uuid.uuid4())
    API_KEYS[api_key] = user_id
    return jsonify({'api_key': api_key})

@app.route('/generate_video', methods=['POST'])
def generate_video_api():
    api_key = request.headers.get('X-API-KEY')
    if api_key not in API_KEYS:
        return jsonify({'error': 'Invalid API key'}), 401
    data = request.json
    text = data.get('text')
    language = data.get('language', 'en')
    duration_multiplier = data.get('duration_multiplier', 1)
    expand_text = data.get('expand_text', False)
    style = data.get('style', 'realistic')
    num_images = data.get('num_images', 3)
    if not text:
        return jsonify({'error': 'Missing text'}), 400
    try:
        video_path = create_video(text, language, duration_multiplier, expand_text, style, num_images)
        return send_file(video_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == '__main__':
    app.run(debug=True)