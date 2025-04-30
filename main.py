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
        # Determine attention implementation only if flash_attn is installed
        attn_impl = None
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = None

        # Load model
        model_kwargs = {
            "torch_dtype": TORCH_DTYPE,
            "low_cpu_mem_usage": True,
            "use_safetensors": True
        }
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            **model_kwargs
        )
        model.to(DEVICE)  # <-- Explicit device placement
        print(f"Model device: {next(model.parameters()).device}")  # <-- Verification

        # (Optional) BetterTransformer speedup
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            pass

        # Load processor (tokenizer + feature_extractor)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

        # Build pipeline with explicit tokenizer & feature extractor
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=CHUNK_LENGTH,
            device=0 if DEVICE == "cuda" else -1,
            torch_dtype=TORCH_DTYPE,
            # Add these kwargs:
            model_kwargs={"use_flash_attention_2": True} if attn_impl else {},
            tokenizer_kwargs={"padding": True}  # Required for pad_token_id
        )

        if DEVICE == "cuda":
            assert torch.cuda.is_available(), "CUDA not available despite DEVICE=cuda"
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

        # Warmup (no sampling_rate argument needed)
        warmup_audio = np.random.rand(SAMPLE_RATE).astype(np.float32)
        pipe(warmup_audio)

    except Exception as e:
        raise RuntimeError(f"Failed to initialize ASR pipeline: {e}")

    yield

    # Cleanup
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
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            audio_file.file.seek(0)
            tmp.write(await audio_file.read())
            temp_path = tmp.name

        audio_array = convert_audio(temp_path)
        result = pipe(
            audio_array,
            generate_kwargs={
                "task": task,
                "language": language,
                "max_new_tokens": 128,
                "temperature": 0.0
            }
        )

        return JSONResponse({"text": result.get("text", "").strip()})

    except Exception as e:
        raise HTTPException(500, detail=f"Processing error: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        torch.cuda.empty_cache()  # Force CUDA memory cleanup

@app.get("/health")
async def health_check() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "device": DEVICE,
        "dtype": str(TORCH_DTYPE),
        "model": MODEL_ID
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)