# Leather Defect Detection — Project Overview

## What this project does

This repository implements an automated leather quality-control tool that detects surface defects from close-up texture photos. It provides a small web UI for uploading multiple images, runs a trained CNN to classify each image into defect categories, and marks a bag as `REJECT` if any uploaded image is classified as defective.

## How it works (high level)

- A ResNet50-based classifier (saved as `leather_defect_model.h5`) predicts one of several defect classes (see `class_names.txt`).
- The web UI is implemented in `app.py` (Flask). Users upload photos; each image is center-cropped, resized to 224×224, and passed through the model. The app displays per-image predictions and an overall accept/reject verdict.
- `train_model.py` builds and trains the model on the `dataset/` folders (train/ and valid/). After training it saves `leather_defect_model.h5` and writes the label order to `class_names.txt`.

## Key files

- `app.py` — Flask web application that loads the model, serves the upload UI, performs predictions, and renders results.
- `train_model.py` — Script to create and train the ResNet50-based model, save it, and produce `class_names.txt`.
- `leather_defect_model.h5` — The trained Keras model (used by `app.py`).
- `class_names.txt` — Newline-separated list of class labels used by the model (e.g. `non defective`, `pinhole`, etc.).
- `dataset/` — Structured image folders used for training: `train/`, `valid/`, and `test/` with subfolders per class.

## Run locally

1. Create and activate a Python 3.11 or 3.12 virtual environment (recommended).

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Train the model from your dataset:

```bash
python train_model.py
```

4. Run the web app:

```bash
python app.py
```

Open the app in your browser at `http://127.0.0.1:5000/` and upload close-up texture photos.

## Notes & tips

- `train_model.py` uses `ResNet50` with a `Lambda(preprocess_input)` layer so the final saved model includes preprocessing.
- The app center-crops images before resizing to focus on texture and reduce background influence.
- The app considers any class other than `non defective` to be a defect; if any uploaded image is defective the whole bag is flagged as rejected.
- `README_SETUP.md` notes Python version compatibility (use Python 3.11/3.12). Follow that if you hit dependency issues.

If you want, I can also: add a short troubleshooting section, fix the `README_SETUP.md` Streamlit/Flask wording, or create a short demo GIF for the UI.
