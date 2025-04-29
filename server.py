from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import tempfile
import os

app = FastAPI(title="Whisper ASR API")

# Global variables for model components
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3-turbo"

model = None
processor = None
pipe = None

@app.on_event("startup")
async def load_model():
    global model, processor, pipe
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Whisper ASR API is running"}

@app.post("/transcribe")
async def transcribe_audio(
        audio_file: UploadFile = File(..., description="Audio file to transcribe"),
        language: str = None,
        task: str = "transcribe",
        return_timestamps: bool = False
):
    if not pipe:
        raise HTTPException(status_code=503, detail="Model not loaded")

    temp_file_path = None
    try:
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            contents = await audio_file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        # Configure generation parameters
        generate_kwargs = {"task": task}
        if language:
            generate_kwargs["language"] = language

        # Run inference
        result = pipe(
            temp_file_path,
            generate_kwargs=generate_kwargs,
            return_timestamps=return_timestamps
        )

        # Prepare response
        response = {"text": result["text"]}
        if return_timestamps:
            response["chunks"] = result.get("chunks", [])

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)