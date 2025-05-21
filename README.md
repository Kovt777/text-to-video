app.py
static/outputs/
templates/index.html


## Установка зависимостей

1. **Основные зависимости**:
```bash
pip install flask torch diffusers==0.20.0 transformers==4.30.2 huggingface_hub==0.16.4 requests
Для улучшения качества видео:

bash
git clone https://github.com/xinntao/BasicSR.git
cd BasicSR
pip install -r requirements.txt
python setup.py develop
pip install realesrgan
Дополнительные репозитории:

bash
git clone https://github.com/cerspense/zeroscope.git
git clone https://github.com/camenduru/Text-To-Video-Finetuning.git
