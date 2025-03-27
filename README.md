# Hate Speech Detection API

This project is a Hate Speech Detection system using a fine-tuned DistilBERT model. It provides a **FastAPI**-based API to classify text as *Hate Speech, Offensive, or Neutral*. The API can be integrated into Discord bots, Chrome extensions, and other community platforms.

## 🚀 Features

- **Machine Learning Model**: Fine-tuned DistilBERT for hate speech detection.
- **FastAPI Integration**: Provides a RESTful API for predictions.
- **Discord Bot Integration**: Automates message moderation.
- **Chrome Extension Support**: Extend functionality to browsers.

## 📂 Project Structure

```
├── datasets/                     # Dataset used for training
│   ├── labeled_data.csv
├── fine_tuned_hate_speech_model/ # Pre-trained model files
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── vocab.txt
├── main.py                       # API implementation using FastAPI
├── train.py                      # Training script for the model
├── requirements.txt              # Dependencies list
├── .gitignore                    # Files to ignore in Git
└── README.md                     # Project documentation
```

## ⚙️ Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/gdg-hackathon-project.git
   cd gdg-hackathon-project
   ```
2. **Set up a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Set up Git LFS (for large model files):**
   ```sh
   git lfs install
   ```

## 🚀 Running the API

Start the FastAPI server:

```sh
uvicorn main:app --reload
```

Access the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## 📡 API Endpoints

- `` - Predicts if the given text is Hate Speech, Offensive, or Neutral.
  ```json
  {
    "text": "Your input text here"
  }
  ```
  **Response:**
  ```json
  {
    "prediction": "Hate Speech"
  }
  ```

## 🤖 Discord Bot Integration

You can integrate this API into a Discord bot by calling the API inside an event handler:

```python
response = requests.post("http://localhost:8000/predict", json={"text": message.content})
result = response.json()
await message.channel.send(f"Prediction: {result['prediction']}")
```

## 🛠️ Training Your Own Model

To retrain the model with new data:

```sh
python train.py
```

## 📜 License

This project is open-source and available under the **MIT License**.

## 💡 Contributors

- **Shivam Rajput** ([@shira0123](https://github.com/shira0123))

Feel free to contribute by submitting issues or pull requests! 🚀

