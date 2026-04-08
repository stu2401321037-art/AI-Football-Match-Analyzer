import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

#Зареждане на GEMINI_API_KEY
load_dotenv()
#Инициализация на клиента
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiService:
    def __init__(self):
        #Взима API ключа от environment променливите
        api_key = os.getenv('GEMINI_API_KEY')

        #Ако няма ключ хвърля грешка
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        #Идентификатор на модела 
        model_id = "gemini-2.0-flash"
        self.model_id = os.getenv('GEMINI_MODEL', model_id)
    
    #Промпт който да се показва в сайта като пример за анализ на мач 
    def analyze_match(self, match_info: str) -> str:
        """
        Analyze a football match using Gemini AI.
        
        Args:
            match_info: Details about the match to analyze
            
        Returns:
            Analysis results from the AI model
        """
        prompt = f"""You are an expert football analyst. Analyze the following match and provide:
        1. Tactical analysis
        2. Key player performances
        3. Team strengths and weaknesses
        4. Critical moments
        5. Overall match assessment
        
        Match Information:
        {match_info}
        
        Provide a comprehensive and detailed analysis."""
        
        try:
            #Списък с частите, които изпращаме на ИИ
            contents = []
            contents.append(prompt)
            response = client.models.generate_content(
                model=self.model_id,    #Моделът, който използваме
                contents=contents,  #Входния текст
                config=types.GenerateContentConfig(
                    temperature=0.7,    #Тук показваме колко креативен да е отговора
                ),
            )
            return response.text #Връщаме текста от отговора

        #Ако има случай при който гемени не може да генерира отговор,връщаме грешка    
        except Exception as exc:
            return f"Error generating analysis: {exc}"

    #Същия като горния промпт само че за предикт
    def predict_outcome(self, match_details: str) -> str:
        """
        Predict the outcome of an upcoming football match.
        
        Args:
            match_details: Details about the upcoming match
            
        Returns:
            Prediction and reasoning from the AI model
        """
        prompt = f"""You are an expert football prediction analyst. Based on the provided information, 
        predict the outcome of the match and provide detailed reasoning including:
        1. Likelihood of each outcome (Win, Draw, Loss for each team)
        2. Key factors influencing the prediction
        3. Probable scoreline
        4. Important player matchups
        5. Risk factors to watch
        
        Match Details:
        {match_details}
        
        Provide a thorough prediction with percentages and clear reasoning."""
        
        try:
            contents = []
            contents.append(prompt)
            response = client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                ),
            )
            return response.text
        except Exception as exc:
            return f"Error generating prediction: {exc}"
