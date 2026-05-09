import os
import shutil
import subprocess
import sys
import threading
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tempfile
import json
import re

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

# Mount the tmp_uploads directory so generated videos can be accessed via URL
app.mount("/outputs", StaticFiles(directory=TMP_DIR), name="outputs")

# Global lock to prevent concurrent generations from overloading the system
generation_lock = threading.Lock()

@app.post("/generate")
async def generate_video(
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
        
        def progress_generator():
            try:
                with generation_lock:
                    print(f"[{job_id}] Acquired generation lock. Running SadTalker: {' '.join(cmd)}")
                    
                    process = subprocess.Popen(
                        cmd,
                        cwd=SADTALKER_DIR,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )
                    
                    # Regex to find progress: e.g. landmark Det::  85%|████████▍ | 90/106
                    progress_re = re.compile(r"(?:(.*)::)?\s*(\d+)%\|.*\| (\d+)/(\d+)")
                    
                    for line in process.stdout:
                        clean_line = line.strip()
                        if not clean_line:
                            continue
                        print(f"[{job_id}] {clean_line}")
                        
                        # Try to parse progress
                        match = progress_re.search(clean_line)
                        if match:
                            stage = match.group(1) or "Processing"
                            percent = int(match.group(2))
                            current = int(match.group(3))
                            total = int(match.group(4))
                            
                            # Clean up common stage names for better UI
                            stage_map = {
                                "landmark Det": "Landmark Detection",
                                "3DMM Extraction In Video": "3DMM Extraction",
                                "mel": "Audio Processing",
                                "audio2exp": "Expression Mapping",
                                "Face Renderer": "Face Rendering"
                            }
                            display_stage = stage_map.get(stage.strip(), stage.strip())
                            
                            yield json.dumps({
                                "status": "generating",
                                "progress": percent,
                                "message": f"{display_stage}: {current}/{total} ({percent}%)"
                            }) + "\n"
                        elif "Full image loop" in clean_line:
                            yield json.dumps({"status": "generating", "progress": 98, "message": "Finalizing video..."}) + "\n"
                            
                    process.wait()
                
                if process.returncode != 0:
                    yield json.dumps({"status": "error", "message": "SadTalker inference failed."}) + "\n"
                    return

                # Find result
                mp4_files = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]
                if not mp4_files:
                    yield json.dumps({"status": "error", "message": "No MP4 generated."}) + "\n"
                    return
                    
                final_video_rel = os.path.join(job_id, "results", mp4_files[0])
                video_url = f"/outputs/{final_video_rel}"
                
                yield json.dumps({
                    "status": "done", 
                    "progress": 100,
                    "presenterVideoPath": video_url,
                    "message": "Generation complete!"
                }) + "\n"
                
            except Exception as e:
                print(f"[{job_id}] ERROR: {str(e)}")
                yield json.dumps({"status": "error", "message": str(e)}) + "\n"

        return StreamingResponse(progress_generator(), media_type="application/x-ndjson")
        
    except Exception as e:
        print(f"[{job_id}] ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    # Note: We keep the files around in the tmp_uploads dir so you can debug if needed.
    # In a fully production system, you'd add a background task to clean them up.
