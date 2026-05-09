import os
import shutil
import subprocess
import sys
import threading
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile

app = FastAPI(title="SadTalker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SADTALKER_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(SADTALKER_DIR, "tmp_uploads")
os.makedirs(TMP_DIR, exist_ok=True)

# Global lock to prevent concurrent generations from overloading the system
generation_lock = threading.Lock()

@app.post("/generate")
def generate_video(
    image: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(TMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    try:
        # Save uploaded files
        image_ext = os.path.splitext(image.filename)[1] or ".png"
        audio_ext = os.path.splitext(audio.filename)[1] or ".wav"
        
        image_path = os.path.join(job_dir, f"input_image{image_ext}")
        audio_path = os.path.join(job_dir, f"input_audio{audio_ext}")
        
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
            
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
            
        # Run SadTalker inference
        # Using typical params for newsStudio (still mode, crop)
        output_dir = os.path.join(job_dir, "results")
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            sys.executable, "inference.py",
            "--driven_audio", audio_path,
            "--source_image", image_path,
            "--result_dir", output_dir,
            "--still",
            "--preprocess", "crop"
        ]
        
        with generation_lock:
            print(f"[{job_id}] Acquired generation lock. Running SadTalker: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                cwd=SADTALKER_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Stream output for debugging
            for line in process.stdout:
                print(f"[{job_id}] {line.strip()}")
                
            process.wait()
        
        if process.returncode != 0:
            raise Exception("SadTalker inference failed. Check logs.")
            
        # Find the generated mp4
        # SadTalker outputs to: result_dir/YYYY_MM_DD_HH.MM.SS.mp4
        mp4_files = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]
        if not mp4_files:
            raise Exception("SadTalker completed but no MP4 was generated.")
            
        final_video = os.path.join(output_dir, mp4_files[0])
        
        # Return the file directly
        # Note: In production you might want to stream this or serve it statically 
        # and delete it via a background task, but this is simple and effective.
        return FileResponse(
            final_video, 
            media_type="video/mp4", 
            filename="sadtalker_output.mp4"
        )
        
    except Exception as e:
        print(f"[{job_id}] ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    # Note: We keep the files around in the tmp_uploads dir so you can debug if needed.
    # In a fully production system, you'd add a background task to clean them up.
