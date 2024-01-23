from fastapi import FastAPI, File, UploadFile
from typing import List
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
import os
from pathlib import Path
from utils.file_utils import size_converter

import uvicorn


ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
UPLOAD_FILE_PATH = '/Users/lvshuo/Desktop/Hybrid-Search-Engine/data/files'

upload_dir = Path(UPLOAD_FILE_PATH)

RESET_HEADERS = {
    "WWW-Authenticate": "Basic",
    "Pragma": "no-cache",
    "Expires": "0",
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Clear-Site-Data": '"cache", "cookies", "storage", "executionContexts"'
}

html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>File Sumbit</h1>
    <form method=post enctype=multipart/form-data>
         <p>FILE: <input type=file name=file><br></p>
         <p>URL PARSE: <input name="url" type="url" class="form-control" id="url"><br></p>
         <p>INPUT SOURCE: <input name="source" type="input_source" class="form-control" id="source"><br></p>
         <p>INPUT CONTEXT: <textarea name="context" rows="10" cols="40" type="context" class="form-control" id="context"></textarea><br></p>
         <input type=submit value=Sumbit>
    </form>
    '''

app = FastAPI()


if not os.path.isdir(upload_dir):
    os.makedirs(upload_dir)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.post("/upload-files", tags=['Upload files to directory'])
async def upload_files(files: List[UploadFile] = File(..., description="Upload files.")):
    """Upload handler.

    Args:
        files: Takes list[UploadFile] as an argument. Use list[bytes] to upload using bytes.

    Returns:
        str:
        File names and the size of each file.
    """
    return_val = []
    if not [f for f in files if f.filename]:
        return JSONResponse(
            content={
                "error_message": "No input received."
            },
            status_code=404,
            headers=RESET_HEADERS
        )
    for file in files:
        data = await file.read()
        file_name = file.filename
        file_format = file_name.rsplit('.', 1)[1]
        if allowed_file(file_name):
            with open(os.path.join(upload_dir, file.filename), 'wb') as file_stream:
                file_stream.write(data)
            return_val.append(
                f"{file.filename}{''.join([' ' for _ in range(60 - len(file.filename))])}{size_converter(len(data))}"
            )
        else:
            return JSONResponse(
                content={
                    "error_message": f"File format {file_format} currently NOT Supported yet."
                },
                status_code=404,
                headers=RESET_HEADERS
            )

    return JSONResponse(content={
        "message": return_val,
        "success": True,
    })


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
