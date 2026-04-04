from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from app.services.gemini_service import GeminiService

#Инициализиране основното приложение на FastAPI
app = FastAPI()
gemini_service = GeminiService()

# Свръзваме папката static с URL за зареждане на frontend-a
app.mount('/static', StaticFiles(directory='static'), name='static')

#Модели за анализ и предикт заявки към API-то
class AnalyzeRequest(BaseModel):
    match_info: str = ''


class PredictRequest(BaseModel):
    match_details: str = ''

#Връщаме index.html при зареждане на главната страница
@app.get('/')
async def index():
    return FileResponse('static/index.html')

#Ендпоинт за анализ на мач 
@app.post('/analyze')
async def analyze_match(data: AnalyzeRequest):
    match_info = data.match_info
    
    if not match_info:
        raise HTTPException(status_code=400, detail='No match information provided')
    
    analysis = gemini_service.analyze_match(match_info)
    return {'analysis': analysis}

#Ендпоинт за предикт на мач
@app.post('/predict')
async def predict_outcome(data: PredictRequest):
    match_details = data.match_details
    
    if not match_details:
        raise HTTPException(status_code=400, detail='No match details provided')
    
    prediction = gemini_service.predict_outcome(match_details)
    return {'prediction': prediction}
#Стартиране на приложението
if __name__ == '__main__':
    uvicorn.run('main:app', host='127.0.0.1', port=5000, reload=True)
