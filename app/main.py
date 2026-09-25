# Build the Final Image and Video FastAPI Application

from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from inference.config import PRODUCTION_CONFIG
from inference.engine import RoadDamageInferenceEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolov26s_rdd2022"
    / "best.pt"
)

UPLOAD_DIR = PROJECT_ROOT / "runtime" / "uploads"
VIDEO_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "video"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
}


app = FastAPI(
    title="Road Damage Detection API",
    version="1.1.0",
    description=(
        "Production API for YOLO26s road damage detection "
        "on images and videos."
    ),
)


engine = RoadDamageInferenceEngine(
    model_path=MODEL_PATH,
    config=PRODUCTION_CONFIG,
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "runtime": "FP16",
        "device": PRODUCTION_CONFIG.device,
    }


def process_video(video_path: Path, output_path: Path):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError("Unable to open the uploaded video.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Unable to create the output video.")

    frames_processed = 0
    total_detections = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            results = engine.model.predict(
                source=frame,
                imgsz=PRODUCTION_CONFIG.image_size,
                conf=PRODUCTION_CONFIG.confidence,
                device=PRODUCTION_CONFIG.device,
                max_det=PRODUCTION_CONFIG.max_det,
                half=PRODUCTION_CONFIG.half,
                save=False,
                verbose=False,
            )

            result = results[0]
            total_detections += len(result.boxes)

            annotated_frame = result.plot()
            writer.write(annotated_frame)

            frames_processed += 1

    finally:
        capture.release()
        writer.release()

    if frames_processed == 0:
        raise RuntimeError("No frames were processed.")

    return {
        "video": output_path.name,
        "video_size": [width, height],
        "fps": round(fps, 2),
        "frames_processed": frames_processed,
        "total_detections": total_detections,
        "runtime": "FP16",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        input_type = "image"

    elif suffix in VIDEO_EXTENSIONS:
        input_type = "video"

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format.",
        )

    upload_path = (
        UPLOAD_DIR
        / f"{uuid4().hex}_{filename}"
    )

    try:
        content = await file.read()
        upload_path.write_bytes(content)

        if input_type == "image":
            result = engine.predict_image(upload_path)
            result["input_type"] = "image"
            result["image"] = filename
            return result

        output_filename = (
            f"{Path(filename).stem}_annotated_{uuid4().hex[:8]}.mp4"
        )

        output_path = VIDEO_OUTPUT_DIR / output_filename

        result = process_video(
            upload_path,
            output_path,
        )

        result["input_type"] = "video"
        result["original_video"] = filename
        result["video_url"] = f"/video/{output_filename}"
        result["watch_url"] = f"/watch/{output_filename}"

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        )

    finally:
        if upload_path.exists():
            upload_path.unlink()

# Add Direct Annotated Video Download Endpoint

@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    filename = Path(file.filename or "upload.mp4").name
    suffix = Path(filename).suffix.lower()

    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only video files are supported.",
        )

    upload_path = (
        UPLOAD_DIR
        / f"{uuid4().hex}_{filename}"
    )

    output_filename = (
        f"{Path(filename).stem}_annotated.mp4"
    )

    output_path = VIDEO_OUTPUT_DIR / output_filename

    try:
        content = await file.read()
        upload_path.write_bytes(content)

        process_video(
            upload_path,
            output_path,
        )

        if not output_path.exists():
            raise RuntimeError(
                "Annotated video was not created."
            )

        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename=output_filename,
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{output_filename}"'
                )
            },
        )

    except Exception as exc:
        if output_path.exists():
            output_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Video inference failed: {exc}",
        )

    finally:
        if upload_path.exists():
            upload_path.unlink()
            
# Add Video Playback and Media Player Actions

from fastapi.responses import FileResponse, HTMLResponse


@app.get("/video/{filename}")
def get_video(filename: str):
    safe_filename = Path(filename).name
    video_path = VIDEO_OUTPUT_DIR / safe_filename

    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Video result not found.",
        )

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=safe_filename,
        headers={
            "Content-Disposition": (
                f'inline; filename="{safe_filename}"'
            )
        },
    )


@app.get("/download/{filename}")
def download_video(filename: str):
    safe_filename = Path(filename).name
    video_path = VIDEO_OUTPUT_DIR / safe_filename

    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Video result not found.",
        )

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=safe_filename,
        headers={
            "Content-Disposition": (
                f'attachment; filename="{safe_filename}"'
            )
        },
    )


@app.get("/watch/{filename}", response_class=HTMLResponse)
def watch_video(filename: str):
    safe_filename = Path(filename).name
    video_path = VIDEO_OUTPUT_DIR / safe_filename

    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Video result not found.",
        )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Road Damage Detection Result</title>

        <meta
            name="viewport"
            content="width=device-width, initial-scale=1.0"
        >

        <style>
            body {{
                margin: 0;
                background: #111;
                color: white;
                font-family: Arial, sans-serif;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 24px;
            }}

            .player {{
                width: 95vw;
                max-width: 1400px;
            }}

            video {{
                width: 100%;
                max-height: 75vh;
                background: black;
                border-radius: 12px;
            }}

            .actions {{
                display: flex;
                gap: 14px;
                flex-wrap: wrap;
                justify-content: center;
            }}

            .button {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 12px 20px;
                border-radius: 10px;
                text-decoration: none;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
            }}

            .play {{
                background: #2563eb;
                color: white;
            }}

            .external {{
                background: #16a34a;
                color: white;
            }}

            .button:hover {{
                opacity: 0.9;
            }}
        </style>
    </head>

    <body>

        <div class="player">
            <video
                controls
                autoplay
                muted
                playsinline
                src="/video/{safe_filename}">
            </video>
        </div>

        <div class="actions">

            <a
                class="button play"
                href="/video/{safe_filename}"
                target="_blank">
                ▶️ Play in Browser
            </a>

            <a
                class="button external"
                href="/download/{safe_filename}">
                🎬 Open in Default Media Player
            </a>

        </div>

    </body>
    </html>
    """