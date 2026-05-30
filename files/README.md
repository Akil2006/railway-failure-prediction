# Railway Equipment Failure Prediction
### Final Year Deep Learning Project — LSTM × GRU

---

## Project Structure

```
railway_failure_prediction/
│
├── app.py                   ← Flask backend (main server)
├── requirements.txt         ← Python dependencies
│
├── models/                  ← Put your saved model files here
│   ├── lstm_model.h5
│   ├── gru_model.h5
│   └── scaler.pkl
│
├── templates/
│   └── index.html           ← Frontend HTML page
│
└── static/
    ├── style.css            ← All styling
    └── app.js               ← JavaScript (form → API → results)
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your model files
Copy `lstm_model.h5`, `gru_model.h5`, and `scaler.pkl` into the `models/` folder.

### 3. Start the server
```bash
python app.py
```

### 4. Open the app
Visit `http://localhost:5000` in your browser.

---

## Code Explanation

### `app.py` — Flask Backend

#### Model Loading (lines 30–55)
```python
lstm_model = load_model(LSTM_PATH)
gru_model  = load_model(GRU_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
```
- Models are loaded **once at startup**, not on every request — this is much faster.
- `load_model()` is Keras's built-in function for restoring `.h5` files.
- The scaler was fit on training data; we must use the same one for inference.

#### `preprocess(sensor_dict)` function
```python
raw      = np.array([[float(sensor_dict[f]) for f in feature_order]])
raw      = scaler.transform(raw)           # same scaling as training
sequence = raw.reshape(1, 1, n_features)   # shape: (samples, timesteps, features)
```
- LSTM/GRU models expect a **3-D input**: `(batch_size, timesteps, features)`.
- Since we have one reading (not a time series of multiple steps), `timesteps = 1`.
- If your training used sequences of e.g. 10 time steps, set `SEQUENCE_LENGTH = 10`
  and send 10 consecutive readings from the frontend.

#### `/predict` endpoint
```
POST /predict
Content-Type: application/json

{ "air_temperature": 298.1, "process_temperature": 308.6, ... }
```
- Validates fields → preprocesses → runs LSTM → runs GRU → averages probabilities.
- Returns a JSON object with per-model results AND a combined `final` decision.

---

### `templates/index.html` — Frontend

The HTML page has two main sections:

| Section | Purpose |
|---|---|
| **Panel 01 — Sensor Readings** | Input form for all 5 sensor values |
| **Panel 02 — Prediction Results** | Shows per-model breakdown + final verdict |

We use a `<div>` form (not `<form>`) so JavaScript controls submission without page reload.

---

### `static/app.js` — JavaScript (the connector)

#### How the frontend talks to the backend

```
User fills form  →  clicks "Run Prediction"
      ↓
validateInputs()  — checks all 5 fields
      ↓
fetch("/predict", { method: "POST", body: JSON.stringify(payload) })
      ↓
Flask receives JSON  →  runs models  →  returns JSON
      ↓
displayResults(data)  — updates the DOM
```

Key JavaScript concepts used:
- **`fetch()`** — modern browser API for HTTP requests (replaces XMLHttpRequest)
- **`async/await`** — makes async code readable; errors caught with `try/catch`
- **`JSON.stringify()`** — converts JS object to JSON string for the request body
- **`response.json()`** — parses the Flask response back into a JS object

#### Input validation (`validateInputs()`)
Before sending anything to the server:
- Checks each field is not empty
- Checks the value is a number
- Checks it falls within a sensible physiological range
- Adds CSS class `invalid` to highlight broken fields

---

### `static/style.css` — Styling

| CSS Feature | Where used |
|---|---|
| CSS Custom Properties (`--amber`, `--bg`, …) | Consistent theming across all elements |
| CSS Grid (`display: grid`) | Two-column layout, model card grid |
| CSS transitions | Smooth hover states, spinner animation, bar fill |
| `@keyframes fadeUp` | Smooth entrance animation for results |
| `@media (max-width: 820px)` | Responsive single-column on mobile |

---

## Adjusting for Your Training Setup

### If your model uses sequences > 1 timestep

In `app.py`, change:
```python
SEQUENCE_LENGTH = 10   # match your training window
```
Then on the frontend you would send an array of 10 readings instead of one.

### If you have different features

Update `feature_order` in `preprocess()` in `app.py` to match the column order
you used when fitting your scaler and training your models:
```python
feature_order = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]
```

### If your model outputs multi-class probabilities

Replace:
```python
prob = float(model.predict(sequence, verbose=0)[0][0])
```
with:
```python
probs = model.predict(sequence, verbose=0)[0]
label = int(np.argmax(probs))
prob  = float(probs[1])   # probability of class 1 (Failure)
```

---

## API Reference

### `POST /predict`

**Request body** (JSON):
```json
{
  "air_temperature": 298.1,
  "process_temperature": 308.6,
  "rotational_speed": 1551,
  "torque": 42.8,
  "tool_wear": 108
}
```

**Response** (JSON):
```json
{
  "lstm": {
    "label": 0,
    "probability": 0.12,
    "confidence": 88.0
  },
  "gru": {
    "label": 0,
    "probability": 0.09,
    "confidence": 91.0
  },
  "final": {
    "label": 0,
    "decision": "No Failure",
    "avg_probability": 0.105
  }
}
```

### `GET /health`
Returns whether each model and the scaler loaded successfully.
```json
{ "lstm_loaded": true, "gru_loaded": true, "scaler_loaded": true }
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `lstm_model failed to load` | Make sure `models/lstm_model.h5` exists and the TF version matches training |
| `Scaler failed to load` | Verify `models/scaler.pkl` was saved with `pickle.dump()` |
| `Missing fields` error | Frontend sent wrong field names — check `FIELDS` array in `app.js` |
| Results always 0 or 1 | Check that `SEQUENCE_LENGTH` in `app.py` matches your training configuration |
| CORS error in browser | Add `flask-cors` and `CORS(app)` if serving frontend from a different origin |
