from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Initialize FastAPI app
app = FastAPI(title="Hate Speech Detection API")

model_path = "./fine_tuned_hate_speech_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define request model for FastAPI
class PredictRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: PredictRequest):
    text = request.text.strip()  # Remove extra whitespace
    print(f"Received text: {text}")  # Debug: see what text is coming in

    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    label_map = {0: "Hate Speech", 1: "Offensive", 2: "Neutral"}
    prediction = label_map[predicted_class]
    
    print(f"Prediction: {prediction}, Probabilities: {probabilities.tolist()}")  # Debugging
    return {"prediction": prediction}


# To run the API, use: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
