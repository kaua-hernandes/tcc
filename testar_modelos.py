import os
from dotenv import load_dotenv
from google import genai
from config.settings import Settings

env_path = os.path.join(Settings.BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("=== MODELOS DISPONÍVEIS NA SUA CONTA ===")
for m in client.models.list():
    # Filtra modelos que suportam geração de texto
    if "generateContent" in m.supported_actions:
        print(f"Nome do modelo: {m.name}")