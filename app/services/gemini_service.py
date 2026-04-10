import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Зареждане на променливите от .env файла
load_dotenv()


class GeminiService:
    def __init__(self):
        # Взима API ключа от environment променливите
        api_key = os.getenv('GEMINI_API_KEY')

        # Ако няма ключ, хвърля грешка
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        # Инициализация на клиента като атрибут на класа
        self.client = genai.Client(api_key=api_key)

        # Идентификатор на модела
        model_id = "gemini-2.5-flash"
        self.model_id = os.getenv('GEMINI_MODEL', model_id)

    def analyze_match(self, match_info: str) -> str:
        """
        Analyze a football match using Gemini AI.
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
            # Използваме self.client и подаваме промпта директно
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                ),
            )
            return response.text

        except Exception as exc:
            return f"Error generating analysis: {exc}"

    def predict_outcome(self, match_details: str) -> str:
        """
        Predict the outcome of an upcoming football match.
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
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                ),
            )
            return response.text
        except Exception as exc:
            return f"Error generating prediction: {exc}" 