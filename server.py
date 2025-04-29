from contextlib import asynccontextmanager
from typing import Optional, cast
import numpy as np
import torch
import ffmpeg
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutomaticSpeechRecognitionPipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import tempfile
import os

# Configuration
MODEL_ID = "openai/whisper-large-v3-turbo"
SAMPLE_RATE = 16000
CHUNK_LENGTH = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Global components with type hints
model: Optional[AutoModelForSpeechSeq2Seq] = None
processor: Optional[AutoProcessor] = None
pipe: Optional[AutomaticSpeechRecognitionPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for model initialization and cleanup"""
    global model, processor, pipe

    # Initialize model
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="flash_attention_2"
        ).to(DEVICE)

        if DEVICE == "cuda":
            model = model.to_bettertransformer()

        processor = AutoProcessor.from_pretrained(MODEL_ID)

        pipe = cast(
            AutomaticSpeechRecognitionPipeline,
            pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=TORCH_DTYPE,
                device=DEVICE,
                chunk_length_s=CHUNK_LENGTH,
            )
        )

        # Warmup with correct input type
        warmup_audio = np.random.rand(SAMPLE_RATE).astype(np.float32)
        pipe(warmup_audio, sampling_rate=SAMPLE_RATE)

    except Exception as e:
        raise RuntimeError(f"Initialization failed: {str(e)}")

    yield  # App is running

    # Cleanup
    del model, processor, pipe
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

def convert_audio(input_path: str) -> np.ndarray:
    """Convert audio to 16kHz mono float32 numpy array"""
    try:
        out, _ = (
            ffmpeg.input(input_path)
            .output('pipe:', format='s16le', ac=1, ar=SAMPLE_RATE)
            .run(capture_stdout=True, quiet=True)
        )
        audio_array = np.frombuffer(out, dtype=np.int16)
        return audio_array.astype(np.float32) / 32768.0
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")

@app.post("/transcribe")
async def transcribe_audio(
        audio_file: UploadFile = File(...),
        language: Optional[str] = None,
        task: str = "transcribe"
) -> JSONResponse:
    if not pipe:
        raise HTTPException(503, "Service unavailable")

    temp_path = None
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            await audio_file.seek(0)
            tmp.write(await audio_file.read())
            temp_path = tmp.name

        # Process audio
        audio_array = convert_audio(temp_path)

        # Type-checked pipeline execution
        result = pipe(
            audio_array,
            sampling_rate=SAMPLE_RATE,
            generate_kwargs={
                "task": task,
                "language": language,
                "max_new_tokens": 128,
                "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            }
        )

        return JSONResponse({"text": result["text"].strip()})

    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
async def health_check() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "device": DEVICE,
        "dtype": str(TORCH_DTYPE),
        "model": MODEL_ID
    })