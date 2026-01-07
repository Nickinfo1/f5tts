from socket_server import TTSStreamingProcessor

import gc
import logging
import socket
import traceback
# from importlib.resources import files
# import numpy as np
# from huggingface_hub import hf_hub_download
# from hydra.utils import get_class
# from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_client(conn, processor):
    try:
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                data = conn.recv(1024)
                if not data:
                    processor.first_package = True
                    break
                data_str = data.decode("utf-8").strip()
                logger.info(f"Received text: {data_str}")

                try:
                    processor.generate_stream(data_str, conn)
                except Exception as inner_e:
                    logger.error(f"Error during processing: {inner_e}")
                    traceback.print_exc()
                    break
    except Exception as e:
        logger.error(f"Error handling client: {e}")
        traceback.print_exc()

def start_server(host, port, processor):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        logger.info(f"Server started on {host}:{port}")
        while True:
            conn, addr = s.accept()
            logger.info(f"Connected by {addr}")
            handle_client(conn, processor)

try:
    # Initialize the processor with the model and vocoder
    processor = TTSStreamingProcessor(
        model="F5TTS_v1_Base",
        ckpt_file="./../../work_data/models/model_last_inference.safetensors",
        vocab_file="",
        ref_audio="./../../work_data/audio/ref.wav",
        ref_text="Где-то падал снег, а я был в теплом месте. Где все ваши инструменты? Необходимо их взять с собой.",
        device="cpu"
    )

    # Start the server
    start_server("0.0.0.0", 9998, processor)

except KeyboardInterrupt:
    gc.collect()