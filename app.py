import json
import math

from flask import Flask, jsonify, render_template, request

from ml_pipeline import INPUT_COLS, MODEL_PATH, NUMERIC_COLS, DemandPredictor, train

app = Flask(__name__)
_predictor = None


def get_predictor() -> DemandPredictor:
    global _predictor
    if _predictor is None:
        if not MODEL_PATH.exists():
            train()
        _predictor = DemandPredictor(MODEL_PATH)
    return _predictor


def parse_record(payload: dict) -> dict:
    missing = [c for c in INPUT_COLS if c not in payload]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    try:
        record = {c: float(payload[c]) for c in NUMERIC_COLS}
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError("All inputs must be valid numbers.") from e
    if not all(math.isfinite(value) for value in record.values()):
        raise ValueError("All inputs must be finite numbers.")
    if record["surge"] < 1:
        raise ValueError("Surge multiplier must be at least 1.")
    if record["available_driver_count"] < 0:
        raise ValueError("Available drivers cannot be negative.")
    if not record["available_driver_count"].is_integer():
        raise ValueError("Available driver count must be a whole number.")
    return record


def get_json_payload() -> dict | None:
    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        return payload

    raw = request.get_data(cache=True, as_text=True)
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None

    return parsed if isinstance(parsed, dict) else None


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok", model_loaded=_predictor is not None)


@app.post("/predict")
@app.post("/forecast")
@app.post("/forecasting")
@app.post("/forcasting")
def predict():
    payload = get_json_payload()
    if not isinstance(payload, dict):
        return jsonify(error="Request body must be a JSON object"), 400
    try:
        record = parse_record(payload)
        result = get_predictor().forecasting(record)
    except (ValueError, TypeError) as e:
        return jsonify(error=str(e)), 400
    return jsonify(result)


if __name__ == "__main__":
    get_predictor()
    app.run(host="0.0.0.0", port=5000, debug=False)