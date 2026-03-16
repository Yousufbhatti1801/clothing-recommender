from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(prefix="/upload", tags=["Image Upload"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image and save it to the local directory.
    """
    try:
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            buffer.write(await file.read())
        return {"message": "File uploaded successfully", "file_path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")