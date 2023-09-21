import os

from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv(".env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]