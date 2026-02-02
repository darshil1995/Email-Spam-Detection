from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import re
import string
import emoji
from nltk.stem import PorterStemmer
import uvicorn

# Initialize FastAPI and Templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")
porter_stemmer = PorterStemmer()

# Load the model and CountVectorizer
# Note: Ensure these files exist in the 'models' folder
try:
    model = joblib.load('saved_models/spam_model.joblib')
    cv = joblib.load('saved_models/cv.joblib')
    print("Model and Vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")


# Exact cleaning function from your training notebook
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'URL_TOKEN', text)
    text = emoji.demojize(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([porter_stemmer.stem(word) for word in text.split()])
    return text


# Define the data structure for the API request
class EmailRequest(BaseModel):
    message: str


# Route to serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Route to handle the logic
@app.post("/predict")
async def predict(request: EmailRequest):
    # 1. Clean
    cleaned = clean_text(request.message)
    # 2. Vectorize
    vectorized = cv.transform([cleaned])
    # 3. Predict
    prediction = model.predict(vectorized)[0]

    result = "Spam" if prediction == 1 else "Ham"
    return {"prediction": result}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)