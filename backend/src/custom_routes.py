import datetime
import io
import json
import pathlib
from typing import Annotated

from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from llama_index.readers.file import PandasCSVReader
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db_utils import get_db, get_db_cm, get_dbb
from src.vector_collections import catalog_index, faq_index, other_index, troubleshooting_index

ALLOWED_EXTENSIONS = {".pdf", ".csv", ".docx"}

app = FastAPI()

collections_map = {
    "catalog": catalog_index,
    "faq": faq_index,
    "troubleshooting": troubleshooting_index,
    "other": other_index,
}


@app.get("/custom/fetch")
async def fetch_some_data(db: Session = Depends(get_db_cm)):
    result = await db.execute(text("SELECT * FROM assistant"))
    rows = result.fetchall()
    return {"rows": rows[0][2]}


@app.get("/custom/files")
async def list_files(db: Session = Depends(get_db_cm)):
    files = await db.execute(
        text(
            "SELECT DISTINCT metadata_->>'filename' AS filename "
            "FROM public.data_techmart_other WHERE metadata_::jsonb ? 'filename'",
        ),
    )
    files = files.scalars().all()
    return {"files": files}


@app.post("/custom/uploadfile")
async def upload_file(file: UploadFile, category: Annotated[str, Form()] = "documents"):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    file_suffix = pathlib.Path(file.filename).suffix
    if file_suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not allowed")

    file_bytes = await file.read()

    async def event_stream():
        try:
            # Step 1 processing
            yield f"data: {json.dumps({'step': 'processing', 'message': 'Processing & embedding…'})}\n\n"
            if file_suffix == ".csv":
                reader = PandasCSVReader(concat_rows=False, pandas_config={"header": None})
                docs = reader.load_data(io.BytesIO(file_bytes))
                for doc in docs:
                    doc.metadata = {
                        "filename": file.filename,
                        "filetype": file_suffix,
                        "category": category,
                        "uploaded_at": datetime.datetime.now(tz=datetime.UTC),
                    }
                print(docs)

            for doc in docs:
                collections_map[category].insert(doc)

            # Step 2 done
            yield f"data: {json.dumps({'step': 'done', 'message': 'Done', 'filename': file.filename})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'step': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/custom/file/{filename}")
async def delete_file(filename: str, db: Session = Depends(get_db_cm)):
    await db.execute(
        text(
            "DELETE FROM public.data_techmart_other WHERE metadata_->>'filename' = :filename"
        ),
        {"filename": filename}
    )
    return {"status": "deleted"}
