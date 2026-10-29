import { useEffect, useState } from "react";
import "./App.css";
const API_URL = "http://127.0.0.1:8000";
const plantCategories = {
  Fruits: [
    { name: "Apple", status: "connected" },
    { name: "Grape", status: "connected" },
    { name: "Peach", status: "connected" },
    { name: "Tomato", status: "connected" },
    { name: "Cherry", status: "connected" },
    { name: "Strawberry", status: "connected" },
  ],
};
const statusText = {
  connected: "Model connected",
  training: "Training in progress",
  soon: "Future fruit plant",
};
function App() {
  const selectedCategory = "Fruits";
  const [selectedPlant, setSelectedPlant] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [topPredictions, setTopPredictions] = useState([]);
  const [confidenceWarning, setConfidenceWarning] = useState("");
  const [resultPlant, setResultPlant] = useState(null);
  const [symptoms, setSymptoms] = useState("");
  const [possibleCause, setPossibleCause] = useState("");
  const [treatment, setTreatment] = useState("");
  const [prevention, setPrevention] = useState("");
  const [explanationImage, setExplanationImage] = useState(null);
  const [modelType, setModelType] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);
  const resetResult = () => {
    setPrediction(null);
    setConfidence(null);
    setTopPredictions([]);
    setConfidenceWarning("");
    setResultPlant(null);
    setSymptoms("");
    setPossibleCause("");
    setTreatment("");
    setPrevention("");
    setExplanationImage(null);
    setModelType("");
    setErrorMessage("");
  };
  const handlePlantSelect = (plant) => {
    setSelectedPlant(plant);
    setSelectedImage(null);
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setPreview(null);
    resetResult();
  };
  const handleImageChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    if (!file.type.startsWith("image/")) {
      setErrorMessage("Please choose a valid image file.");
      return;
    }
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setSelectedImage(file);
    setPreview(URL.createObjectURL(file));
    resetResult();
  };
  const formatPrediction = (value, plant) => {
    if (!value) {
      return "Unknown";
    }

    const cleaned = String(value).replace(/_/g, " ").replace(/\s+/g, " ").trim();
    const plantName = String(plant || "").replace(/_/g, " ").trim();

    if (
      plantName &&
      cleaned.toLowerCase().startsWith(plantName.toLowerCase() + " ")
    ) {
      return cleaned.slice(plantName.length + 1).trim();
    }

    return cleaned;
  };

  const handleAnalyze = async () => {
    if (!selectedPlant || !selectedImage) {
      setErrorMessage(
        "Select a fruit plant and upload its leaf image before analysis."
      );
      return;
    }
    setIsAnalyzing(true);
    setErrorMessage("");
    resetResult();
    try {
      const formData = new FormData();
      formData.append("file", selectedImage);
      formData.append("plant", selectedPlant);
      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          body: formData,
        }
      );
      if (!response.ok) {
        let message = "Analysis request failed.";
        try {
          const errorData = await response.json();
          if (errorData.detail) {
            message = errorData.detail;
          }
        } catch {
          // Keep default error message.
        }
        throw new Error(message);
      }
      const data = await response.json();
      setPrediction(formatPrediction(data.prediction, selectedPlant));
      setConfidence(data.confidence ?? 0);
      setTopPredictions(
        (Array.isArray(data.top_predictions) ? data.top_predictions : []).map(
          (item) => ({
            ...item,
            displayName: formatPrediction(item.class_name, selectedPlant),
          })
        )
      );
      setConfidenceWarning(
        data.confidence_warning ||
          (Number(data.confidence ?? 0) < 70
            ? "The model is uncertain about this image. Review the alternatives and do not use this result alone to choose treatment."
            : "")
      );
      setResultPlant(data.plant || selectedPlant);
      setExplanationImage(data.explanation_image || null);
      setModelType(data.model_type || "Plant-specific classifier");
      setSymptoms(
        data.symptoms || "Information not available."
      );
      setPossibleCause(
        data.possible_cause || "Information not available."
      );
      setTreatment(
        data.treatment || "Information not available."
      );
      setPrevention(
        data.prevention || "Information not available."
      );
      setTimeout(() => {
        document
          .getElementById("diagnosis-report")
          ?.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
      }, 100);
    } catch (error) {
      console.error("Prediction error:", error);
      setErrorMessage(
        error.message ||
          "Unable to connect to the diagnostic backend."
      );
    } finally {
      setIsAnalyzing(false);
    }
  };
  const selectedPlantObject =
    selectedCategory && selectedPlant
      ? plantCategories[selectedCategory].find(
          (plant) => plant.name === selectedPlant
        )
      : null;
  const isModelAvailable =
    selectedPlantObject?.status === "connected";
  return (
    <div className="site">
      {/* HEADER */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">GAD</div>
          <div>
            <div className="brand-name">
              Generative Agricultural Diagnostics
            </div>
            <div className="brand-subtitle">
              Fruit Plant Disease Detection
            </div>
          </div>
        </div>
        <nav className="navigation">
          <a href="#diagnosis">Diagnosis</a>
          <a href="#workflow">Workflow</a>
          <a href="#about">About</a>
        </nav>
      </header>
      <main>
        {/* HERO */}
        <section className="hero" id="diagnosis">
          <div className="hero-left">
            <div className="eyebrow">
              FRUIT PLANT DISEASE DETECTION
            </div>
            <h1>
              Fruit plant health,
              <br />
              examined from a
              <br />
              <span>single leaf.</span>
            </h1>
            <p className="hero-description">
              Select a fruit plant, upload a leaf photograph,
              and receive a model-based disease prediction,
              confidence information, and disease-management
              guidance. More fruit plants can be added as models become available.
            </p>
            <a
              className="hero-action"
              href="#plant-selection"
            >
              Begin diagnosis
            </a>
          </div>
          <div className="hero-panel">
            <div className="panel-heading">
              CURRENT SYSTEM
            </div>
            <div className="system-row">
              <span>Image classifier</span>
              <strong>ResNet-50</strong>
            </div>
            <div className="system-row">
              <span>Generative model</span>
              <strong>Stable Diffusion + LoRA</strong>
            </div>
            <div className="system-row">
              <span>Connected plants</span>
              <strong>Apple / Grape / Peach / Tomato / Cherry / Strawberry</strong>
            </div>
            <div className="system-note">
              Six plant-specific classifiers are connected for image-based
              disease prediction.
            </div>
          </div>
          <div className="hero-visual" aria-hidden="true">
            <img
              src="https://images.pexels.com/photos/15203364/pexels-photo-15203364.jpeg?auto=compress&cs=tinysrgb&w=1800"
              alt="Agricultural field with healthy green leaves"
            />
            <div className="hero-visual-label">
              Field environment / plant observation
            </div>
          </div>
        </section>
        {/* PLANT SELECTION */}
        <section
          className="section"
          id="plant-selection"
        >
          <div className="section-intro">
            <div>
              <div className="eyebrow">
                01 / PLANT SELECTION
              </div>
              <h2>
                Choose a fruit plant
              </h2>
            </div>
            <p>
              Choose a fruit plant before uploading an image.
              The selected plant determines which
              diagnostic model is used.
            </p>
          </div>
          <div className="fruit-selection-heading">
            <h3>Available Fruit Plants</h3>
            <p>
              Select one of the supported fruit plants to check its leaf for possible diseases.
              Select any of the six connected plant models to analyze its leaf image.
              </p>
              </div>

          <div className="plant-grid">
            {plantCategories[selectedCategory].map(
              (plant) => (
                <button
                  key={plant.name}
                  className={
                    selectedPlant === plant.name
                      ? "plant-card selected"
                      : "plant-card"
                  }
                  onClick={() =>
                    handlePlantSelect(plant.name)
                  }
                >
                  <div className="plant-card-top">
                    <span className="plant-number">
                      {String(
                        plantCategories[
                          selectedCategory
                        ].indexOf(plant) + 1
                      ).padStart(2, "0")}
                    </span>
                    <span
                      className={`status-dot ${plant.status}`}
                    ></span>
                  </div>
                  <div className="plant-name">
                    {plant.name}
                  </div>
                  <div className="plant-status">
                    {statusText[plant.status]}
                  </div>
                </button>
              )
            )}
          </div>
        </section>
        {/* UPLOAD WORKSPACE */}
        {selectedPlant && (
          <section className="section workspace-section">
            <div className="section-intro">
              <div>
                <div className="eyebrow">
                  02 / IMAGE ANALYSIS
                </div>
                <h2>
                  Examine {selectedPlant}
                </h2>
              </div>
              <p>
                Upload a clear photograph of the selected fruit plant leaf,
                showing the leaf surface and any visible symptoms.
              </p>
            </div>
            <div className="workspace">
              <div className="preview-panel">
                <div className="preview-kicker">LIVE IMAGE VIEW</div>
                {preview ? (
                  <img
                    src={preview}
                    alt="Selected plant leaf"
                    className="leaf-preview"
                  />
                ) : (
                  <div className="empty-preview">
                    <div className="empty-preview-title">
                      Image preview
                    </div>
                    <div className="empty-preview-text">
                      Your selected fruit plant leaf image will appear
                      here.
                    </div>
                  </div>
                )}
              </div>
              <div className="upload-panel">
                <div className="upload-index">
                  SELECTED PLANT
                </div>
                <div className="selected-plant-name">
                  {selectedPlant}
                </div>
                <div className="upload-divider"></div>
                <div className="upload-title">
                  Upload a fruit plant leaf photograph
                </div>
                <div className="upload-description">
                  JPG, PNG, WEBP, BMP or TIFF
                </div>
                <label className="upload-control">
                  {selectedImage
                    ? "Replace image"
                    : "Choose image"}
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageChange}
                    hidden
                  />
                </label>
                {selectedImage && (
                  <div className="file-name">
                    {selectedImage.name}
                  </div>
                )}
                <div className="model-status">
                  <span
                    className={`status-dot ${
                      isModelAvailable
                        ? "connected"
                        : "soon"
                    }`}
                  ></span>
                  <span>
                    {isModelAvailable
                      ? "Diagnostic model available"
                      : "Diagnostic model not connected"}
                  </span>
                </div>
                {isModelAvailable && preview && (
                  <button
                    className="analyze-control"
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                  >
                    {isAnalyzing
                      ? "Analyzing image..."
                      : "Analyze image"}
                  </button>
                )}
                {!isModelAvailable && (
                  <div className="unavailable-note">
                    A trained model is not connected for
                    {` ${selectedPlant}`} yet. The plant
                    remains visible as part of the planned
                    multi-plant system.
                  </div>
                )}
              </div>
            </div>
            {errorMessage && (
              <div className="error-box">
                {errorMessage}
              </div>
            )}
          </section>
        )}
        {/* RESULT */}
        {prediction && (
          <section
            className="section report-section"
            id="diagnosis-report"
          >
            <div className="report-heading">
              <div>
                <div className="eyebrow">
                  03 / DIAGNOSTIC REPORT
                </div>
                <h2>
                  Analysis complete
                </h2>
              </div>
              <div className="report-model">
                {modelType || "Plant-specific classifier"}
              </div>
            </div>
            <div className="diagnosis-summary">
              <div className="summary-plant">
                <div className="summary-label">
                  PLANT
                </div>
                <div className="summary-value">
                  {resultPlant || selectedPlant}
                </div>
              </div>
              <div className="summary-diagnosis">
                <div className="summary-label">
                  DETECTED CONDITION
                </div>
                <div className="diagnosis-name">
                  {prediction}
                </div>
              </div>
              <div className="summary-confidence">
                <div className="summary-label">
                  CONFIDENCE
                </div>
                <div className="confidence-number">
                  {confidence}%
                </div>
                <div className="confidence-track">
                  <div
                    className="confidence-fill"
                    style={{
                      width: `${Math.min(
                        Number(confidence) || 0,
                        100
                      )}%`,
                    }}
                  ></div>
                </div>
              </div>
            </div>
            {confidenceWarning && (
              <div
                role="alert"
                style={{
                  marginTop: "18px",
                  padding: "16px 18px",
                  border: "1px solid #d8b36a",
                  borderRadius: "10px",
                  background: "#fff8e8",
                  color: "#684b13",
                  lineHeight: 1.55,
                }}
              >
                <strong>Low-confidence result — review before acting</strong>
                <p style={{ margin: "6px 0 0" }}>{confidenceWarning}</p>
                <p style={{ margin: "6px 0 0", fontSize: "0.9em" }}>
                  The displayed score is the model's softmax score; it is not a calibrated guarantee of correctness.
                </p>
              </div>
            )}
            {topPredictions.length > 1 && (
              <section
                aria-label="Top candidate predictions"
                style={{
                  marginTop: "18px",
                  padding: "20px 22px",
                  border: "1px solid #d9e1dc",
                  borderRadius: "12px",
                  background: "#ffffff",
                }}
              >
                <div className="eyebrow">ALTERNATIVE CANDIDATES</div>
                <h3 style={{ marginTop: "8px", marginBottom: "8px" }}>
                  Other classes the model considered
                </h3>
                <p style={{ marginTop: 0, lineHeight: 1.5, color: "#647067" }}>
                  These are the model's top three scores, not confirmed diagnoses.
                </p>
                <div style={{ display: "grid", gap: "12px" }}>
                  {topPredictions.map((item, index) => (
                    <div key={`${item.class_name}-${index}`}>
                      <div style={{ display: "flex", justifyContent: "space-between", gap: "12px", marginBottom: "5px" }}>
                        <span>{index + 1}. {item.displayName}</span>
                        <strong>{Number(item.confidence).toFixed(2)}%</strong>
                      </div>
                      <div style={{ height: "7px", background: "#e6e9e5", borderRadius: "999px", overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${Math.min(Number(item.confidence) || 0, 100)}%`, background: index === 0 ? "#315b43" : "#8caa96", borderRadius: "999px" }} />
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}
            {explanationImage && (
              <section
                aria-label="Grad-CAM visual explanation"
                style={{
                  marginTop: "28px",
                  padding: "24px",
                  border: "1px solid #d9e1dc",
                  borderRadius: "12px",
                  background: "#f8faf8",
                }}
              >
                <h3 style={{ marginTop: "8px", marginBottom: "8px" }}>
                  Why the model predicted this
                  </h3>
                  <p style={{ marginTop: 0, marginBottom: "18px", lineHeight: 1.6 }}>
                    Grad-CAM highlights image regions that influenced the model's selected class.
                    It is an explanation of model behavior, not proof that the diagnosis is correct.
                    </p>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
                    gap: "20px",
                  }}
                >
                  <div>
                    <p style={{ fontWeight: 600, marginTop: 0 }}>Uploaded leaf image</p>
                    <img
                      src={preview}
                      alt="Original uploaded leaf"
                      style={{
                        display: "block",
                        width: "100%",
                        maxHeight: "360px",
                        objectFit: "contain",
                        background: "#ffffff",
                        borderRadius: "8px",
                      }}
                    />
                  </div>
                  <div>
                    <p style={{ fontWeight: 600, marginTop: 0 }}>Grad-CAM attention overlay</p>
                    <img
                      src={explanationImage}
                      alt="Grad-CAM heatmap overlay showing image regions that influenced the prediction"
                      style={{
                        display: "block",
                        width: "100%",
                        maxHeight: "360px",
                        objectFit: "contain",
                        background: "#ffffff",
                        borderRadius: "8px",
                      }}
                    />
                  </div>
                </div>
              </section>
            )}
            <div className="information-grid">
              <article className="information-card">
                <div className="info-index">
                  01
                </div>
                <h3>
                  Symptoms
                </h3>
                <p>
                  {symptoms}
                </p>
              </article>
              <article className="information-card">
                <div className="info-index">
                  02
                </div>
                <h3>
                  Possible cause
                </h3>
                <p>
                  {possibleCause}
                </p>
              </article>
              <article className="information-card">
                <div className="info-index">
                  03
                </div>
                <h3>
                  Treatment
                </h3>
                <p>
                  {treatment}
                </p>
              </article>
              <article className="information-card">
                <div className="info-index">
                  04
                </div>
                <h3>
                  Prevention
                </h3>
                <p>
                  {prevention}
                </p>
              </article>
            </div>
            <div className="new-analysis">
              <button
                onClick={() => {
                  setSelectedImage(null);
                  if (preview) {
                    URL.revokeObjectURL(preview);
                  }
                  setPreview(null);
                  resetResult();
                  window.scrollTo({
                    top: document.getElementById(
                      "plant-selection"
                    )?.offsetTop || 0,
                    behavior: "smooth",
                  });
                }}
              >
                Start another analysis
              </button>
            </div>
          </section>
        )}
        {/* WORKFLOW */}
        <section
          className="section workflow-section"
          id="workflow"
        >
          <div className="section-intro">
            <div>
              <div className="eyebrow">
                THE WORKFLOW
              </div>
              <h2>
                From image to diagnosis
              </h2>
            </div>
            <p>
              The application classifies leaf images from supported fruit plants
              and explores synthetic-data augmentation when real disease
              images are limited.
            </p>
          </div>
          <div className="workflow-grid">
            <div className="workflow-step">
              <span>01</span>
              <h3>Select fruit plant</h3>
              <p>
                Choose the fruit plant you want to examine.
              </p>
            </div>
            <div className="workflow-step">
              <span>02</span>
              <h3>Upload leaf</h3>
              <p>
                Provide a clear image of the selected fruit plant leaf.
              </p>
            </div>
            <div className="workflow-step">
              <span>03</span>
              <h3>Model analysis</h3>
              <p>
                The connected ResNet-50 model evaluates
                the image.
              </p>
            </div>
            <div className="workflow-step">
              <span>04</span>
              <h3>View guidance</h3>
              <p>
                Review confidence, symptoms, cause,
                treatment and prevention information.
              </p>
            </div>
          </div>
        </section>
        {/* ABOUT */}
        <section
          className="section about-section"
          id="about"
        >
          <div className="about-copy">
            <div className="eyebrow">
              ABOUT THE PROJECT
            </div>
            <h2>
              Generative Agricultural Diagnostics
            </h2>
            <p>
              GAD investigates whether synthetic disease images of fruit plants,
              generated using Stable Diffusion and adapted with LoRA,
              can help improve classification when real disease images
              are limited.
            </p>
          </div>
          <div className="about-visual">
            <img
              src="https://images.pexels.com/photos/6064376/pexels-photo-6064376.jpeg?auto=compress&cs=tinysrgb&w=1600"
              alt="Close-up green leaf texture"
              loading="lazy"
            />
            <span>Natural leaf texture</span>
          </div>
          <div className="about-meta">
            <div>
              <span>Research focus</span>
              <strong>
                Synthetic data augmentation
              </strong>
            </div>
            <div>
              <span>Classifier</span>
              <strong>
                ResNet-50
              </strong>
            </div>
            <div>
              <span>Application</span>
              <strong>
                Fruit plant disease detection
              </strong>
            </div>
          </div>
        </section>
      </main>
      {/* FOOTER */}
      <footer className="footer">
        <div>
          GAD — Generative Agricultural Diagnostics
        </div>
        <div>
          Fruit plant disease detection and research prototype
        </div>
      </footer>
    </div>
  );
}
export default App;
