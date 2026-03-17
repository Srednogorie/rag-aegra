import anyio
from fastapi import FastAPI, UploadFile

app = FastAPI()


@app.get("/custom/files")
async def list_files():
    files = [x.name async for x in anyio.Path("src/files").iterdir()]
    return {"files": files}


@app.post("/custom/uploadfile")
async def upload_file(file: UploadFile):
    write_file = anyio.Path(f"src/files/{file.filename}")
    await write_file.write_bytes(await file.read())
    return {"received": file.filename, "status": "processed"}


@app.delete("/custom/file/{filename}")
async def delete_file(filename: str):
    await anyio.Path(f"src/files/{filename}").unlink()
    return {"status": "deleted"}
