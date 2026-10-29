import base64
import io
import json
import os

import numpy as np

import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as F


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="GAD - Generative Agricultural Diagnostics",
    version="1.1.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
INFO_PATH = os.path.join(BASE_DIR, "disease_info.json")

TOMATO_MULTICLASS_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "tomato_multiclass_real_only_best.pth",
)

APPLE_MULTICLASS_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "apple_multiclass_real_only_best.pth",
)

GRAPE_MULTICLASS_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "grape_multiclass_v8_best.pth",
)

# Retained as fallbacks to the completed binary research experiments.
TOMATO_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "model_B2_resnet50.pth",
)

APPLE_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "apple_black_rot_model_B_real_plus_synthetic_final.pth",
)

GRAPE_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "grape_black_rot_model_B_real_plus_synthetic_best.pth",
)

PEACH_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "peach_bacterial_leaf_spot_model_B_real_plus_synthetic.pth",
)

# New GAD checkpoints trained for Cherry, Peach, and Strawberry.
CHERRY_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "cherry_resnet50_best.pth",
)

PEACH_NEW_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "peach_resnet50_best.pth",
)

STRAWBERRY_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "strawberry_resnet50_best.pth",
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# GLOBAL STATE
# ============================================================

MODELS = {}
DISEASE_INFO = {}

# Built-in information for the newly connected Peach model.
# This keeps the result page populated even when the local
# disease_info.json does not yet contain these Peach entries.
BUILTIN_DISEASE_INFO = {
    "Peach_Bacterial Leaf Spot": {
        "symptoms": (
            "Small dark brown to reddish-brown leaf spots may develop, "
            "with affected tissue sometimes drying and dropping out. "
            "Severe infections can cause premature leaf drop."
        ),
        "possible_cause": (
            "Bacterial spot of peach is caused by "
            "Xanthomonas arboricola pv. pruni. Warm, humid, rainy and "
            "windy conditions can favor infection."
        ),
        "treatment": (
            "Management depends on cultivar, disease pressure and local "
            "recommendations. Copper-based bactericide programs may be "
            "used where locally registered and appropriate; always follow "
            "the current product label and local agricultural guidance."
        ),
        "prevention": (
            "Use less-susceptible cultivars where available, reduce the "
            "amount of initial inoculum, prune out bacterial spot cankers "
            "where appropriate, and maintain a management program suited "
            "to local weather and disease pressure."
        ),
    },
    "Peach_Healthy": {
        "symptoms": (
            "No major disease symptoms were identified by the connected "
            "Peach classifier."
        ),
        "possible_cause": (
            "The image was classified as healthy by the Peach model."
        ),
        "treatment": (
            "No disease treatment is indicated from this classification. "
            "Continue normal orchard care and routine monitoring."
        ),
        "prevention": (
            "Continue regular scouting, maintain good orchard hygiene, "
            "and use locally appropriate preventive practices."
        ),
    },
}


# ============================================================
# DISEASE INFORMATION
# ============================================================

def load_disease_info():
    global DISEASE_INFO

    if not os.path.exists(INFO_PATH):
        print("⚠️ disease_info.json not found.")
        DISEASE_INFO = {}
        return

    try:
        with open(INFO_PATH, "r", encoding="utf-8") as f:
            DISEASE_INFO = json.load(f)

        print(
            f"✅ Disease information loaded: {len(DISEASE_INFO)} entries"
        )
    except Exception as exc:
        print(f"⚠️ Failed to load disease_info.json: {exc}")
        DISEASE_INFO = {}


def get_disease_info(class_name):
    candidates = [
        class_name,
        class_name.replace("_", " "),
        class_name.replace(" ", "_"),
    ]

    # Check built-in information first.
    for key in candidates:
        if key in BUILTIN_DISEASE_INFO:
            return BUILTIN_DISEASE_INFO[key]

    # Then check the existing disease_info.json.
    for key in candidates:
        if key in DISEASE_INFO and isinstance(DISEASE_INFO[key], dict):
            return DISEASE_INFO[key]

    return {
        "symptoms": "Information not yet added.",
        "possible_cause": "Information not yet added.",
        "treatment": "Information not yet added.",
        "prevention": "Information not yet added.",
    }


# ============================================================
# MODEL LOADING
# ============================================================

def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        return checkpoint

    raise RuntimeError("Unsupported model checkpoint format.")


def create_resnet50(num_classes):
    net = models.resnet50(weights=None)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def load_multiclass_checkpoint(model_path, model_key):
    """Load a checkpoint that stores both model weights and class names."""
    if not os.path.exists(model_path):
        print(f"⚠️ Multiclass model not found: {model_path}")
        return False

    try:
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        )

        if not isinstance(checkpoint, dict) or not checkpoint.get("class_names"):
            raise RuntimeError("Checkpoint is missing its class_names list.")

        class_names = checkpoint["class_names"]
        state_dict = extract_state_dict(checkpoint)
        net = create_resnet50(len(class_names))
        net.load_state_dict(state_dict)
        net = net.to(DEVICE)
        net.eval()

        MODELS[model_key] = {
            "model": net,
            "class_names": class_names,
            "model_type": "Real-only multiclass baseline",
            "is_multiclass": True,
        }

        print(
            f"✅ {model_key} multiclass model loaded: "
            f"{len(class_names)} classes"
        )
        print(f"   Best validation accuracy: {checkpoint.get('best_val_accuracy', 'unknown')}")
        return True

    except Exception as exc:
        print(f"❌ Failed to load {model_key} multiclass model: {exc}")
        return False


def load_binary_model(model_path, class_names, model_key):
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found: {model_path}")
        return False

    try:
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=False,
        )

        state_dict = extract_state_dict(checkpoint)

        net = create_resnet50(len(class_names))
        net.load_state_dict(state_dict)
        net = net.to(DEVICE)
        net.eval()

        MODELS[model_key] = {
            "model": net,
            "class_names": class_names,
            "model_type": "Model B - Real + Synthetic",
            "is_multiclass": False,
        }

        print(
            f"✅ {model_key} model loaded: "
            f"{len(class_names)} classes"
        )
        print(f"   Classes: {class_names}")
        return True

    except Exception as exc:
        print(f"❌ Failed to load {model_key} model: {exc}")
        return False


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("GAD BACKEND STARTING")
    print("=" * 60)
    print("Device:", DEVICE)

    load_disease_info()

    # Tomato: use the new 12-class real-only multiclass model when present.
    # If it is missing or fails to load, retain compatibility with the old binary model.
    if not load_multiclass_checkpoint(TOMATO_MULTICLASS_MODEL_PATH, "Tomato"):
        load_binary_model(
            TOMATO_MODEL_PATH,
            ["Tomato_Early Blight", "Tomato_Healthy"],
            "Tomato",
        )

    # Apple: use the new 9-class real-only multiclass model when present.
    # Keep the old binary Apple research model as a fallback if the new
    # checkpoint has not been copied into backend/models yet.
    if not load_multiclass_checkpoint(APPLE_MULTICLASS_MODEL_PATH, "Apple"):
        load_binary_model(
            APPLE_MODEL_PATH,
            ["Apple_Black Rot", "Apple_Healthy"],
            "Apple",
        )

    # Grape: use the new 5-class real-only multiclass model when present.
    # Keep the previous binary Grape research model as a fallback.
    if not load_multiclass_checkpoint(GRAPE_MULTICLASS_MODEL_PATH, "Grape"):
        load_binary_model(
            GRAPE_MODEL_PATH,
            ["Grape_Black Rot", "Grape_Healthy"],
            "Grape",
        )

    # Cherry: new 2-class ResNet-50 checkpoint.
    load_multiclass_checkpoint(
        CHERRY_MODEL_PATH,
        "Cherry",
    )

    # Peach: prefer the new checkpoint. If it fails to load, retain
    # compatibility with the existing Peach Model B checkpoint.
    if not load_multiclass_checkpoint(
        PEACH_NEW_MODEL_PATH,
        "Peach",
    ):
        load_binary_model(
            PEACH_MODEL_PATH,
            ["Peach_Bacterial Leaf Spot", "Peach_Healthy"],
            "Peach",
        )

    # Strawberry: new 4-class ResNet-50 checkpoint.
    load_multiclass_checkpoint(
        STRAWBERRY_MODEL_PATH,
        "Strawberry",
    )

    print("\nSupported plant models:", list(MODELS.keys()))
    for plant, bundle in MODELS.items():
        print(
            f"   {plant}: {len(bundle['class_names'])} classes "
            f"({bundle.get('model_type', 'unknown model type')})"
        )
    print("=" * 60)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "project": "GAD - Generative Agricultural Diagnostics",
        "status": "running",
        "supported_plants": list(MODELS.keys()),
        "device": str(DEVICE),
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": list(MODELS.keys()),
        "model_loaded": bool(MODELS),
        "device": str(DEVICE),
    }


# ============================================================
# CLASSES / SUPPORTED PLANTS
# ============================================================

@app.get("/classes")
def get_classes():
    classes = []

    for plant, bundle in MODELS.items():
        for class_name in bundle["class_names"]:
            classes.append(class_name)

    return {
        "model_type": "Plant-specific classifiers; Tomato, Apple, Grape, Cherry, Peach, and Strawberry",
        "count": len(classes),
        "classes": classes,
        "supported_plants": list(MODELS.keys()),
    }


# ============================================================
# EXPLAINABLE AI: GRAD-CAM
# ============================================================

def make_gradcam_image(model, input_tensor, original_image, target_index):
    """Return a PNG data URL showing the regions influencing a class score."""
    captured = {}

    def capture_layer4(_module, _inputs, output):
        captured["features"] = output

    hook = model.layer4.register_forward_hook(capture_layer4)

    try:
        model.zero_grad(set_to_none=True)

        # Gradients are required for Grad-CAM, so do not wrap this in no_grad().
        with torch.enable_grad():
            outputs = model(input_tensor)
            features = captured.get("features")
            if features is None:
                raise RuntimeError("Could not capture ResNet-50 layer4 features.")

            gradients = torch.autograd.grad(
                outputs[0, target_index],
                features,
                retain_graph=False,
                create_graph=False,
            )[0]

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * features.detach()).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam = cam.detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Create the heatmap in the model-input coordinate system first.
        # The classifier input is resized to 224x224, but the uploaded image
        # may not be square. Resize the heatmap back to the source image's
        # dimensions before blending so the displayed overlay stays aligned
        # with the original image instead of being shown as a square.
        red = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
        green = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
        blue = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
        heatmap = np.stack([red, green, blue], axis=-1)
        heatmap = (heatmap * 255).astype(np.uint8)

        base_image = original_image.convert("RGB")
        heatmap_image = Image.fromarray(heatmap, mode="RGB").resize(
            base_image.size,
            resample=Image.Resampling.BILINEAR,
        )
        base_array = np.asarray(base_image).astype(np.float32)
        heatmap_array = np.asarray(heatmap_image).astype(np.float32)
        blended = (0.58 * base_array + 0.42 * heatmap_array)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        overlay = Image.fromarray(blended, mode="RGB")

        output_buffer = io.BytesIO()
        overlay.save(output_buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(output_buffer.getvalue()).decode("ascii")
        return "data:image/png;base64," + encoded

    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    plant: str = Form(...),
):
    # --------------------------------------------------------
    # MODEL CHECK
    # --------------------------------------------------------

    if plant not in MODELS:
        supported = ", ".join(MODELS.keys())
        raise HTTPException(
            status_code=400,
            detail=(
                f"A trained model is not connected for {plant} yet. "
                f"Currently supported: {supported}."
            ),
        )

    # --------------------------------------------------------
    # FILE TYPE CHECK
    # --------------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/tiff",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, JPEG, PNG, WEBP, BMP, or TIFF image.",
        )

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        image = Image.open(io.BytesIO(contents)).convert("RGB")

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read the uploaded image: {exc}",
        )

    # --------------------------------------------------------
    # SELECT MODEL
    # --------------------------------------------------------

    bundle = MODELS[plant]
    model = bundle["model"]
    class_names = bundle["class_names"]

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    explanation_image = None

    try:
        # Gradient-based forward pass is needed to create the Grad-CAM map.
        with torch.enable_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs.detach(), dim=1)
            confidence, prediction_index = probabilities.max(dim=1)

        predicted_index = prediction_index.item()
        predicted_class = class_names[predicted_index]
        confidence_percent = confidence.item() * 100

        # Show alternative candidates as well as the top-1 prediction.
        # These are softmax scores, not calibrated probabilities.
        top_k = min(3, len(class_names))
        top_scores, top_indices = torch.topk(
            probabilities[0], k=top_k
        )
        top_predictions = [
            {
                "class_name": class_names[int(class_index)],
                "confidence": round(float(score) * 100, 2),
            }
            for score, class_index in zip(top_scores, top_indices)
        ]

        # A simple initial warning threshold; it should be calibrated and
        # validated on a representative field-image set before production use.
        low_confidence_threshold = 70.0
        needs_review = confidence_percent < low_confidence_threshold

        try:
            explanation_image = make_gradcam_image(
                model=model,
                input_tensor=input_tensor,
                original_image=image,
                target_index=predicted_index,
            )
        except Exception as cam_exc:
            # Do not fail the diagnosis if only the explanation visualization fails.
            print(f"⚠️ Grad-CAM generation failed for {plant}: {cam_exc}")

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )

    # --------------------------------------------------------
    # DISEASE INFORMATION
    # --------------------------------------------------------

    info = get_disease_info(predicted_class)

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "success": True,
        "model_type": bundle.get("model_type", "Plant-specific classifier"),
        "prediction": predicted_class,
        "explanation_method": "Grad-CAM" if explanation_image else None,
        "explanation_image": explanation_image,
        "confidence": round(confidence_percent, 2),
        "top_predictions": top_predictions,
        "needs_review": needs_review,
        "confidence_threshold": low_confidence_threshold,
        "confidence_warning": (
            "The model is uncertain about this image. Review the alternative "
            "classes and do not use this result alone to choose treatment."
            if needs_review else None
        ),
        "plant": plant,
        "selected_plant": plant,
        "filename": file.filename,
        "symptoms": info.get(
            "symptoms",
            "Information not yet added.",
        ),
        "possible_cause": info.get(
            "possible_cause",
            "Information not yet added.",
        ),
        "treatment": info.get(
            "treatment",
            "Information not yet added.",
        ),
        "prevention": info.get(
            "prevention",
            "Information not yet added.",
        ),
    }
