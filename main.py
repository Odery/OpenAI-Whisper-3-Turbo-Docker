from contextlib import asynccontextmanager
from typing import Optional
import numpy as np
import torch
import ffmpeg
import subprocess
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)
import tempfile
import os

# Configuration
MODEL_ID = "openai/whisper-large-v3-turbo"
SAMPLE_RATE = 16000
CHUNK_LENGTH = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Global components (populated on startup)
model: Optional[AutoModelForSpeechSeq2Seq] = None
processor: Optional[AutoProcessor] = None
pipe: Optional[AutomaticSpeechRecognitionPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events for loading and unloading the Whisper model."""
    global model, processor, pipe
    try:
        # Load model
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="flash_attention_2"
        ).to(DEVICE)

        # (Optional) BetterTransformer speedup
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            # optimum[bettertransformer] not installed; skip
            pass

        # Load processor
        processor = AutoProcessor.from_pretrained(MODEL_ID)

        # Build pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor,          # processor handles tokenization
            feature_extractor=processor,  # and feature extraction
            chunk_length_s=CHUNK_LENGTH,
            device=0 if DEVICE == "cuda" else -1,
            torch_dtype=TORCH_DTYPE
        )

        # Warmup call
        warmup_audio = np.random.rand(SAMPLE_RATE).astype(np.float32)
        pipe(warmup_audio, sampling_rate=SAMPLE_RATE)

    except Exception as e:
        raise RuntimeError(f"Failed to initialize ASR pipeline: {e}")

    yield  # app is ready

    # Cleanup on shutdown
    del model, processor, pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

def convert_audio(input_path: str) -> np.ndarray:
    """Use ffmpeg to read and convert audio into a float32 numpy array."""
    try:
        out, _ = (
            ffmpeg.input(input_path)
            .output('pipe:', format='f32le', ac=1, ar=SAMPLE_RATE)
            .run(capture_stdout=True, quiet=True)
        )
        return np.frombuffer(out, dtype=np.float32)
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode(errors='ignore') if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg error: {msg}")

@app.post("/transcribe")
async def transcribe_audio(
        audio_file: UploadFile = File(...),
        language: Optional[str] = None,
        task: str = "transcribe"
) -> JSONResponse:
    if pipe is None:
        raise HTTPException(503, detail="Service unavailable, pipeline not initialized")

    temp_path = None
    try:
        # Save upload to temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            audio_file.file.seek(0)
            tmp.write(await audio_file.read())
            temp_path = tmp.name

        # Convert to numpy array
        audio_array = convert_audio(temp_path)

        # Run recognition
        result = pipe(
            audio_array,
            sampling_rate=SAMPLE_RATE,
            generate_kwargs={
                "task": task,
                "language": language,
                "max_new_tokens": 128,
                # Whisper doesn't accept list temps: use single float
                "temperature": 0.0
            }
        )

        return JSONResponse({"text": result.get("text", "").strip()})

    except Exception as e:
        raise HTTPException(500, detail=f"Processing error: {e}")
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
