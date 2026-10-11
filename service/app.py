"""
service/app.py — FastAPI Web Service & API for Returns Risk
============================================================
Provides real-time risk scoring, explanations, health check, and interactive web UI.
Uses FastAPI lifespan handler.

Data location:
    products.csv is resolved via KESTREL_DATA_DIR env var or data/ directory.
    See kestrel/data_paths.py and README.md.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from kestrel.data_paths import find_data_file
from kestrel.features import (
    ALL_FEATURES,
    build_single_order_features,
    explain_prediction,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
# products.csv resolved via KESTREL_DATA_DIR env var or data/ directory
PRODUCTS_CSV_NAME = "products.csv"

# Global model state
state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and metadata at startup
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(
            f"Model artifacts not found in '{ARTIFACTS_DIR}'.\n"
            "Run 'py train.py' first to train the model."
        )

    state["pipeline"] = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        state["metadata"] = json.load(f)

    # Load products table for SKU metadata lookup.
    # Raises FileNotFoundError clearly if products.csv is missing — do NOT silently
    # substitute an empty or invented table, as that would change model scores.
    products_path = find_data_file(PRODUCTS_CSV_NAME)
    state["products"] = pd.read_csv(products_path)

    yield
    state.clear()

app = FastAPI(
    title="Kestrel Home — Returns Risk Scoring API",
    description="Risk assessment service for D2C order dispatch decision support.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Request / Response Schemas ────────────────────────────────────────────────
class OrderRiskRequest(BaseModel):
    order_id: str = Field(..., json_schema_extra={"example": "KO2610504"})
    sku: str = Field("KH-AF-01", json_schema_extra={"example": "KH-AF-01"})
    sales_channel: str = Field("app", json_schema_extra={"example": "app"})
    payment_mode: str = Field("cod", json_schema_extra={"example": "cod"})
    discount_pct: float = Field(10.0, ge=0, le=100, json_schema_extra={"example": 15.0})
    qty: int = Field(1, ge=1, json_schema_extra={"example": 1})
    order_value_inr: float = Field(4500.0, ge=0, json_schema_extra={"example": 4200.0})
    promised_delivery_days: int = Field(5, ge=1, json_schema_extra={"example": 5})
    is_gift: str = Field("N", json_schema_extra={"example": "N"})
    customer_prior_orders: int = Field(0, ge=0, json_schema_extra={"example": 2})
    customer_prior_returns: int = Field(0, ge=0, json_schema_extra={"example": 1})
    delivery_pincode: Optional[int] = Field(None, json_schema_extra={"example": 110001})

class RiskReason(BaseModel):
    feature: str
    impact: str
    description: str

class OrderRiskResponse(BaseModel):
    order_id: str
    return_risk_score: float
    risk_band: str
    threshold: float
    reasons: List[RiskReason]
    note: str = "Risk estimate for human review, not an automated decision."

# ── Helper for Feature Preparation & Reasons ──────────────────────────────────
def prepare_features(req: OrderRiskRequest) -> pd.DataFrame:
    req_dict = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    df, _ = build_single_order_features(req_dict, state.get("products"))
    return df

def generate_reasons(df: pd.DataFrame, pipe: Optional[Any] = None) -> List[RiskReason]:
    raw_reasons = explain_prediction(df, pipe, top_n=3)
    return [
        RiskReason(
            feature=r["feature"],
            impact=r["impact"],
            description=r["description"],
        )
        for r in raw_reasons
    ]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/healthz")
def health_check():
    return {"status": "ok", "model_loaded": "pipeline" in state}

@app.post("/predict", response_model=OrderRiskResponse)
def predict_risk(req: OrderRiskRequest):
    if "pipeline" not in state:
        raise HTTPException(status_code=500, detail="Model pipeline not initialized.")
    
    df = prepare_features(req)
    pipe = state["pipeline"]
    meta = state.get("metadata", {})
    
    score = float(pipe.predict_proba(df)[0, 1])
    threshold = float(meta.get("threshold", 0.6485))
    
    if score >= threshold:
        risk_band = "HIGH"
    elif score >= threshold * 0.7:
        risk_band = "MEDIUM"
    else:
        risk_band = "LOW"

    reasons = generate_reasons(df, pipe)

    return OrderRiskResponse(
        order_id=req.order_id,
        return_risk_score=round(score, 4),
        risk_band=risk_band,
        threshold=round(threshold, 4),
        reasons=reasons,
    )

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    meta = state.get("metadata", {})
    thresh = meta.get("threshold", 0.6485)
    model_name = meta.get("model", "LogisticRegression")
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kestrel Home — Returns Risk Evaluator</title>
  <style>
    :root {{
      --bg-dark: #0f172a;
      --card-bg: #1e293b;
      --accent-blue: #38bdf8;
      --accent-red: #f43f5e;
      --accent-green: #10b981;
      --text-light: #f8fafc;
      --text-muted: #94a3b8;
    }}
    body {{
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background-color: var(--bg-dark);
      color: var(--text-light);
      margin: 0;
      padding: 40px 20px;
      display: flex;
      justify-content: center;
    }}
    .container {{
      max-width: 850px;
      width: 100%;
    }}
    .header {{
      text-align: center;
      margin-bottom: 30px;
    }}
    .header h1 {{
      font-size: 2rem;
      color: var(--accent-blue);
      margin: 0 0 8px 0;
    }}
    .header p {{
      color: var(--text-muted);
      margin: 0;
    }}
    .card {{
      background-color: var(--card-bg);
      border-radius: 12px;
      padding: 28px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.4);
      border: 1px solid #334155;
      margin-bottom: 24px;
    }}
    .form-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 18px;
    }}
    .form-group {{
      display: flex;
      flex-direction: column;
    }}
    label {{
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-bottom: 6px;
      font-weight: 600;
    }}
    input, select {{
      background: #0f172a;
      border: 1px solid #475569;
      color: var(--text-light);
      padding: 10px 14px;
      border-radius: 6px;
      font-size: 0.95rem;
    }}
    input:focus, select:focus {{
      outline: none;
      border-color: var(--accent-blue);
    }}
    button {{
      background: linear-gradient(135deg, #0284c7, #2563eb);
      color: white;
      border: none;
      padding: 14px;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      width: 100%;
      margin-top: 20px;
      transition: opacity 0.2s;
    }}
    button:hover {{
      opacity: 0.9;
    }}
    .result-box {{
      display: none;
      margin-top: 24px;
      padding: 20px;
      border-radius: 10px;
      background: #0f172a;
      border: 1px solid #334155;
    }}
    .score-badge {{
      display: inline-block;
      padding: 6px 14px;
      border-radius: 20px;
      font-weight: bold;
      font-size: 1.1rem;

    }}
    .HIGH {{ background: rgba(244, 63, 94, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }}
    .MEDIUM {{ background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }}
    .LOW {{ background: rgba(16, 185, 129, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }}
    
    .reasons-list {{
      margin-top: 16px;
      padding-left: 20px;
    }}
    .reasons-list li {{
      margin-bottom: 8px;
      color: #cbd5e1;
    }}
    .disclaimer {{
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 14px;
      font-style: italic;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Kestrel Home — Returns Risk Assessor</h1>
      <p>Pre-dispatch order evaluation tool (Model: {model_name} | Target Capacity Threshold: {thresh})</p>
    </div>

    <div class="card">
      <form id="risk-form">
        <div class="form-grid">
          <div class="form-group">
            <label for="order_id">Order ID</label>
            <input type="text" id="order_id" value="KO2610504" required>
          </div>
          <div class="form-group">
            <label for="sku">SKU Code</label>
            <select id="sku">
              <option value="KH-AF-01">KH-AF-01 (Air Fryer Lite)</option>
              <option value="KH-AF-02">KH-AF-02 (Air Fryer Pro)</option>
              <option value="KH-WP-01">KH-WP-01 (Water Purifier Lite)</option>
              <option value="KH-RV-01">KH-RV-01 (Robot Vacuum Lite)</option>
              <option value="KH-MG-01">KH-MG-01 (Mixer Grinder Lite)</option>
            </select>
          </div>
          <div class="form-group">
            <label for="sales_channel">Sales Channel</label>
            <select id="sales_channel">
              <option value="app">App</option>
              <option value="web">Web</option>
              <option value="marketplace">Marketplace</option>
              <option value="partner_outlet">Partner Outlet</option>
            </select>
          </div>
          <div class="form-group">
            <label for="payment_mode">Payment Mode</label>
            <select id="payment_mode">
              <option value="cod">Cash on Delivery (COD)</option>
              <option value="prepaid_upi">Prepaid UPI</option>
              <option value="prepaid_card">Prepaid Card</option>
              <option value="emi">EMI</option>
            </select>
          </div>
          <div class="form-group">
            <label for="discount_pct">Discount %</label>
            <input type="number" id="discount_pct" value="15" min="0" max="100">
          </div>
          <div class="form-group">
            <label for="order_value_inr">Order Value (₹)</label>
            <input type="number" id="order_value_inr" value="4500" min="1">
          </div>
          <div class="form-group">
            <label for="customer_prior_orders">Customer Prior Orders</label>
            <input type="number" id="customer_prior_orders" value="1" min="0">
          </div>
          <div class="form-group">
            <label for="customer_prior_returns">Customer Prior Returns</label>
            <input type="number" id="customer_prior_returns" value="1" min="0">
          </div>
          <div class="form-group">
            <label for="is_gift">Is Gift Order?</label>
            <select id="is_gift">
              <option value="N">No</option>
              <option value="Y">Yes</option>
            </select>
          </div>
          <div class="form-group">
            <label for="promised_delivery_days">Promised Delivery Days</label>
            <input type="number" id="promised_delivery_days" value="5" min="1">
          </div>
        </div>
        <button type="submit">Evaluate Risk Score</button>
      </form>

      <div id="result" class="result-box">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div>
            <h3 style="margin:0; font-size:1.3rem;">Order: <span id="res-id"></span></h3>
            <p style="margin:4px 0 0 0; color:var(--text-muted);">Calculated Risk Score: <strong id="res-score"></strong></p>
          </div>
          <span id="res-band" class="score-badge"></span>
        </div>
        
        <h4 style="margin-top:20px; margin-bottom:8px;">Key Risk Drivers:</h4>
        <ul id="res-reasons" class="reasons-list"></ul>
        
        <p class="disclaimer">⚠️ <span id="res-note">Risk estimate for human review, not an automated decision.</span></p>
      </div>
    </div>
  </div>

  <script>
    document.getElementById('risk-form').addEventListener('submit', async (e) => {{
      e.preventDefault();
      const body = {{
        order_id: document.getElementById('order_id').value,
        sku: document.getElementById('sku').value,
        sales_channel: document.getElementById('sales_channel').value,
        payment_mode: document.getElementById('payment_mode').value,
        discount_pct: parseFloat(document.getElementById('discount_pct').value),
        qty: 1,
        order_value_inr: parseFloat(document.getElementById('order_value_inr').value),
        promised_delivery_days: parseInt(document.getElementById('promised_delivery_days').value),
        is_gift: document.getElementById('is_gift').value,
        customer_prior_orders: parseInt(document.getElementById('customer_prior_orders').value),
        customer_prior_returns: parseInt(document.getElementById('customer_prior_returns').value),
      }};

      const res = await fetch('/predict', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(body)
      }});

      if (res.ok) {{
        const data = await res.json();
        document.getElementById('res-id').innerText = data.order_id;
        document.getElementById('res-score').innerText = (data.return_risk_score * 100).toFixed(1) + '%';
        const bandEl = document.getElementById('res-band');
        bandEl.innerText = data.risk_band + ' RISK';
        bandEl.className = 'score-badge ' + data.risk_band;
        
        const ul = document.getElementById('res-reasons');
        ul.innerHTML = '';
        data.reasons.forEach(r => {{
          const li = document.createElement('li');
          li.innerHTML = `<strong>${{r.feature}}</strong> (${{r.impact}}): ${{r.description}}`;
          ul.appendChild(li);
        }});

        document.getElementById('res-note').innerText = data.note;
        document.getElementById('result').style.display = 'block';
      }} else {{
        alert('Error evaluating score');
      }}
    }});
  </script>
</body>
</html>"""
