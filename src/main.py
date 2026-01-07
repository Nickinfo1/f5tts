from f5_tts.api import F5TTS  # или просто ваш модуль, если вы его запускаете локально
from importlib.resources import files

# Пути
ref_audio = "./../work_data/audio/ref.wav"           # референсное аудио (< 30 сек)
ref_text = "Где-то падал снег, а я был в теплом месте. Где все ваши инструменты? Необходимо их взять с собой."  # точная транскрипция
gen_text = "Привет! Это синтезированная речь с использованием локальной модели F5-TTS."

# Инициализация модели с локальным чекпоинтом
f5tts = F5TTS(
    model="F5TTS_v1_Base",                     # ← имя YAML-файла в configs/ (без .yaml)
    # ckpt_file="./model_last_inference.safetensors",  # ← ваш файл
    ckpt_file="./../work_data/models/model_last_inference.safetensors",  # ← ваш файл
    device="cpu"  # или "cpu"
)

# Запуск синтеза
wav, sr, spec = f5tts.infer(
    ref_file=ref_audio,
    ref_text=ref_text,
    gen_text=gen_text,
    file_wave="./output.wav",      # куда сохранить аудио
    remove_silence=True,           # убрать тишину в конце
    seed=42                        # для воспроизводимости
)

print("Синтез завершён! Аудио сохранено в output.wav")