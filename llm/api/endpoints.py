from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from llm.services.analysis import process_data, generate_sentiment_plot
import pandas as pd
from io import BytesIO

router = APIRouter()

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    data = pd.read_json(BytesIO(contents))
    analysis_result = process_data(data)
    return JSONResponse(content=analysis_result)

@router.post("/plot_sentiment/")
async def plot_sentiment(file: UploadFile = File(...)):
    contents = await file.read()
    data = pd.read_json(BytesIO(contents))
    image_base64 = generate_sentiment_plot(data)
    return JSONResponse(content={"image": image_base64})
