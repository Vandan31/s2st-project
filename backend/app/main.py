import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from backend.pipeline import run_pipeline
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI()

# ✅ Enable CORS so frontend (localhost:3000) can talk to backend (localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories for uploads and outputs
UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("backend/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    """Handle file upload and run ASR -> Translate -> TTS pipeline"""
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Run pipeline
        result = run_pipeline(str(file_path), out_dir=str(OUTPUT_DIR))
        tts_path = Path(result["tts_wav"])

        return {
            "en_text": result["en_text"],
            "hi_text": result["hi_text"],
            "tts_url": f"/audio/{tts_path.name}"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated TTS audio"""
    p = OUTPUT_DIR / filename
    if not p.exists():
        return JSONResponse(status_code=404, content={"error": "file not found"})
    return FileResponse(str(p), media_type="audio/wav", filename=filename)


# Run with: uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
