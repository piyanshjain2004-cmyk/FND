# TruthLens

TruthLens is a Flask-based fake news detection web application that analyzes submitted news text and returns a credibility-focused prediction. The app uses NLP preprocessing, TF-IDF vectorization, and a Passive Aggressive classifier trained on labeled news data.

## Features

- Real/Fake news prediction
- Prediction confidence indicator
- Reason and evidence-style signals
- Clean multi-page interface
- Separate route and service structure for easier maintenance

## Project Structure

```text
FND/
  app.py
  requirements.txt
  routes/
  services/
  templates/
  datasets/
```

## Setup

```bash
pip install -r requirements.txt
python app.py
```

## Deploy Free on Render

This project is configured for Render with [`render.yaml`](./render.yaml).

1. Push this repository to GitHub.
2. Go to Render dashboard and choose **New +** -> **Blueprint**.
3. Connect your GitHub repo and select this project.
4. Render will detect `render.yaml` and create a free Python web service.
5. After build and deploy complete, you will get a public URL like:
   `https://truthlens-fnd.onrender.com`

### Important

- Free services can sleep when inactive, so first request after idle time can be slow.
- This app downloads required NLTK resources during build.
- Start command on Render:
  `gunicorn app:app --bind 0.0.0.0:$PORT`

## Notes

- The model is loaded from saved files if available.
- If the model files are missing, the app trains a new model using the dataset files in `datasets/`.
- NLTK resources are downloaded automatically on first run.

## Main Pages

- `/` Home
- `/predict` Prediction
- `/how-it-works` How It Works
- `/about` About
