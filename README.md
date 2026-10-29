# GAD — Generative Agricultural Diagnostics

## Overview

GAD (Generative Agricultural Diagnostics) is an AI-based plant disease diagnosis project designed to help identify plant diseases from leaf images.

The project explores the use of synthetic image generation to address the limited availability of real images for certain plant diseases. It combines deep learning-based image classification with a web interface for submitting leaf images and viewing predictions.

## Objectives

- Identify plant diseases from uploaded leaf images.
- Explore synthetic image generation for data augmentation.
- Train and evaluate deep learning image classification models.
- Provide an accessible web interface for disease prediction.
- Investigate the effect of synthetic data on classification performance.

## Technologies Used

- **Frontend:** React, JavaScript, CSS, Vite
- **Backend:** Python, FastAPI
- **Deep Learning:** PyTorch, ResNet-50
- **Generative AI:** Stable Diffusion and LoRA
- **Data Analysis:** Python and relevant machine learning libraries

## Project Structure

```text
GAD/
├── backend/
│   ├── main.py
│   └── disease_info.json
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.js
├── .gitignore
└── README.md
```

## How It Works

1. A user selects a plant and uploads a leaf image.
2. The frontend sends the image to the backend.
3. The backend processes the image using the configured disease classification model.
4. The application displays the predicted disease and available diagnostic information.

The exact plant and disease coverage depends on the models configured in the application.

## Research Approach

The research investigates whether synthetic disease images can supplement limited real training data.

The intended comparison is between models trained using real images only and models trained using real images combined with synthetic images.

## Running the Project

### 1. Backend

Activate your existing Python environment, then install the dependencies listed in your backend requirements file.

Start the FastAPI application using the entry point and command configured in your project.

### 2. Frontend

Open a terminal in the frontend directory and run:

```bash
npm install
npm run dev
```

Use the local URL displayed by Vite to open the frontend.

**Note:** Backend dependency installation and startup commands should be confirmed against the actual project configuration before running the application on another computer.

## Current Status

The project source code is available in this repository. Model checkpoints and datasets may be maintained separately and are not necessarily included in the repository.

## Future Scope

- Expand and validate plant disease coverage.
- Improve model evaluation across different image conditions.
- Refine synthetic image quality and filtering.
- Deploy the frontend and backend for wider access.
- Provide clearer setup and model download instructions.

## Disclaimer

GAD is an academic project intended to assist with plant disease identification. Predictions may be incorrect and should not replace expert agricultural advice.

## Author

**Kaivalya Kaushik**

GitHub: [kaivalyakaushik](https://github.com/kaivalyakaushik)
