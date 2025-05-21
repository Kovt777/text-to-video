import os
import tempfile
import sys
import platform
import requests
from flask import Flask, render_template, request, send_file, jsonify
import torch

# Фикс для huggingface_hub
from huggingface_hub import hf_hub_download
sys.modules['huggingface_hub'].cached_download = hf_hub_download

from diffusers import DiffusionPipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

class Translator:
    def __init__(self):
        self.is_macos = platform.system() == "Darwin"
        self.API_KEY = "____"
        self.MODEL = "deepseek/deepseek-r1:free"
    
    def translate(self, text):
        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты профессиональный переводчик с русского на английский. Переведи текст максимально точно."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            "temperature": 0.3
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Возвращаем оригинал при ошибке

# Инициализация переводчика
translator = Translator()

# Инициализация модели генерации видео
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained(
    "cerspense/zeroscope_v2_576w",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
pipe.safety_checker = None

STYLE_PRESETS = {
    "Джеки Чан": "martial arts fight scene, dynamic camera angles",
    "Симпсоны": "yellow cartoon characters, Simpsons style",
    "Футурама": "futuristic cartoon, sci-fi elements",
    "Киберпанк": "neon lights, rainy cityscape",
    "Прозрачная одежда": "see-through clothing, translucent fabric"
}

@app.route('/')
def index():
    return render_template('index.html', styles=STYLE_PRESETS.keys())

@app.route('/generate', methods=['POST'])
def generate_video():
    try:
        prompt_ru = request.form['prompt']
        selected_style = request.form.get('style', '')
        
        # Перевод через нейросеть
        prompt_en = translator.translate(prompt_ru)
        
        if selected_style in STYLE_PRESETS:
            prompt_en += ", " + STYLE_PRESETS[selected_style]
        
        result = pipe(
            prompt_en,
            num_inference_steps=40,
            height=320,
            width=576
        )
        
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, 'output.mp4')
        result.frames[0].save(output_path)
        
        return jsonify({
            'status': 'success',
            'video_url': f'/download?path={output_path}'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/download')
def download():
    path = request.args.get('path')
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
