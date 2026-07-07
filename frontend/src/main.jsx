import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  BadgeCheck,
  BarChart3,
  ChevronRight,
  CreditCard,
  Download,
  Loader2,
  ShieldCheck,
  Upload,
} from "lucide-react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const repaymentOptions = [
  { value: -2, label: "No bill / no card use" },
  { value: -1, label: "Paid in full" },
  { value: 0, label: "Revolving / minimum paid" },
  { value: 1, label: "1 month late" },
  { value: 2, label: "2 months late" },
  { value: 3, label: "3 months late" },
  { value: 4, label: "4 months late" },
  { value: 5, label: "5 months late" },
  { value: 6, label: "6 months late" },
  { value: 7, label: "7 months late" },
  { value: 8, label: "8+ months late" },
];

const initialCustomer = {
  LIMIT_BAL: 200000,
  SEX: 2,
  EDUCATION: 2,
  MARRIAGE: 2,
  AGE: 35,
  PAY_0: 0,
  PAY_2: 0,
  PAY_3: 0,
  PAY_4: 0,
  PAY_5: 0,
  PAY_6: 0,
  BILL_AMT1: 50000,
  BILL_AMT2: 48000,
  BILL_AMT3: 46000,
  BILL_AMT4: 44000,
  BILL_AMT5: 42000,
  BILL_AMT6: 40000,
  PAY_AMT1: 3000,
  PAY_AMT2: 3000,
  PAY_AMT3: 3000,
  PAY_AMT4: 3000,
  PAY_AMT5: 3000,
  PAY_AMT6: 3000,
};

const monthFields = [
  { pay: "PAY_0", bill: "BILL_AMT1", paid: "PAY_AMT1", label: "September 2005" },
  { pay: "PAY_2", bill: "BILL_AMT2", paid: "PAY_AMT2", label: "August 2005" },
  { pay: "PAY_3", bill: "BILL_AMT3", paid: "PAY_AMT3", label: "July 2005" },
  { pay: "PAY_4", bill: "BILL_AMT4", paid: "PAY_AMT4", label: "June 2005" },
  { pay: "PAY_5", bill: "BILL_AMT5", paid: "PAY_AMT5", label: "May 2005" },
  { pay: "PAY_6", bill: "BILL_AMT6", paid: "PAY_AMT6", label: "April 2005" },
];

function numberValue(value) {
  return Number.isFinite(Number(value)) ? Number(value) : 0;
}

function estimatePreview(customer) {
  const delays = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"].map((key) =>
    numberValue(customer[key])
  );
  const payments = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3"].map((key) => numberValue(customer[key]));
  const bills = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"].map((key) => numberValue(customer[key]));
  const maxDelay = Math.max(...delays);
  const utilization = Math.max(0, numberValue(customer.BILL_AMT1) / Math.max(1, numberValue(customer.LIMIT_BAL)));
  const recentPaymentRatio =
    payments.reduce((sum, value) => sum + value, 0) /
    Math.max(1, bills.reduce((sum, value) => sum + Math.max(0, value), 0));
  const raw = -2.4 + maxDelay * 0.65 + utilization * 1.2 - recentPaymentRatio * 1.8;
  const probability = 1 / (1 + Math.exp(-raw));
  return Math.max(0.02, Math.min(0.92, probability));
}

function riskSegment(score) {
  if (score <= 30) return "Low Risk";
  if (score <= 60) return "Medium Risk";
  return "High Risk";
}

function App() {
  const [customer, setCustomer] = useState(initialCustomer);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState("");

  const previewProbability = useMemo(() => estimatePreview(customer), [customer]);
  const previewScore = Math.round(previewProbability * 100);

  const updateValue = (field, value) => {
    setCustomer((current) => ({ ...current, [field]: numberValue(value) }));
  };

  const submitPrediction = async () => {
    setLoading(true);
    setApiError("");
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(customer),
      });
      if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
      }
      const data = await response.json();
      setPrediction(data);
      await loadHistory();
    } catch (error) {
      setApiError("Backend is not running yet. Showing an interface preview instead.");
      const probability = previewProbability;
      const riskScore = Math.round(probability * 100);
      setPrediction({
        default_probability: probability,
        risk_score: riskScore,
        risk_segment: riskSegment(riskScore),
        predicted_default: probability >= 0.3 ? 1 : 0,
        mode: "Preview only",
      });
    } finally {
      setLoading(false);
    }
  };

  const loadHistory = async () => {
    try {
      const response = await fetch(`${API_URL}/assessments?limit=8`);
      if (!response.ok) return;
      const data = await response.json();
      setHistory(data.items || []);
    } catch {
      setHistory([]);
    }
  };

  const activePrediction = prediction || {
    default_probability: previewProbability,
    risk_score: previewScore,
    risk_segment: riskSegment(previewScore),
    predicted_default: previewProbability >= 0.3 ? 1 : 0,
    mode: "Live preview",
  };

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <ShieldCheck size={24} />
          </div>
          <div>
            <p className="eyebrow">Explainable ML</p>
            <h1>RiskNot</h1>
          </div>
        </div>

        <nav className="nav-list" aria-label="Dashboard navigation">
          <a className="nav-item active" href="#assessment">
            <CreditCard size={18} />
            Customer Assessment
          </a>
          <a className="nav-item" href="#performance">
            <BarChart3 size={18} />
            Model Summary
          </a>
          <a className="nav-item" href="#batch">
            <Upload size={18} />
            Batch Upload
          </a>
        </nav>

        <section className="sidebar-panel" id="performance">
          <p className="panel-label">Current Model</p>
          <strong>LightGBM tuned</strong>
          <span>Threshold 0.30, recall-focused credit screening.</span>
        </section>
      </aside>

      <section className="workspace" id="assessment">
        <header className="topbar">
          <div>
            <p className="eyebrow">Single Customer Risk Prediction</p>
            <h2>Credit default risk assessment</h2>
          </div>
          <button className="secondary-button" type="button">
            <Download size={17} />
            Export
          </button>
        </header>

        <div className="content-grid">
          <form className="form-surface" onSubmit={(event) => event.preventDefault()}>
            <section className="section-block">
              <div className="section-heading">
                <h3>Customer Profile</h3>
                <span>Readable inputs mapped to model codes.</span>
              </div>
              <div className="profile-grid">
                <label>
                  Credit limit
                  <input
                    type="number"
                    min="0"
                    value={customer.LIMIT_BAL}
                    onChange={(event) => updateValue("LIMIT_BAL", event.target.value)}
                  />
                </label>
                <label>
                  Age
                  <input
                    type="number"
                    min="18"
                    max="100"
                    value={customer.AGE}
                    onChange={(event) => updateValue("AGE", event.target.value)}
                  />
                </label>
                <label>
                  Gender
                  <select value={customer.SEX} onChange={(event) => updateValue("SEX", event.target.value)}>
                    <option value={1}>Male</option>
                    <option value={2}>Female</option>
                  </select>
                </label>
                <label>
                  Education
                  <select
                    value={customer.EDUCATION}
                    onChange={(event) => updateValue("EDUCATION", event.target.value)}
                  >
                    <option value={1}>Graduate school</option>
                    <option value={2}>University</option>
                    <option value={3}>High school</option>
                    <option value={4}>Other / Unknown</option>
                  </select>
                </label>
                <label>
                  Marital status
                  <select
                    value={customer.MARRIAGE}
                    onChange={(event) => updateValue("MARRIAGE", event.target.value)}
                  >
                    <option value={1}>Married</option>
                    <option value={2}>Single</option>
                    <option value={3}>Other / Unknown</option>
                  </select>
                </label>
              </div>
            </section>

            <section className="section-block">
              <div className="section-heading">
                <h3>Repayment History</h3>
                <span>Month-by-month status, bill amount, and paid amount.</span>
              </div>
              <div className="month-list">
                {monthFields.map((month) => (
                  <div className="month-row" key={month.pay}>
                    <div className="month-title">{month.label}</div>
                    <label>
                      Repayment status
                      <select
                        value={customer[month.pay]}
                        onChange={(event) => updateValue(month.pay, event.target.value)}
                      >
                        {repaymentOptions.map((option) => (
                          <option value={option.value} key={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Bill statement
                      <input
                        type="number"
                        value={customer[month.bill]}
                        onChange={(event) => updateValue(month.bill, event.target.value)}
                      />
                    </label>
                    <label>
                      Amount paid
                      <input
                        type="number"
                        min="0"
                        value={customer[month.paid]}
                        onChange={(event) => updateValue(month.paid, event.target.value)}
                      />
                    </label>
                  </div>
                ))}
              </div>
            </section>
          </form>

          <aside className="result-panel">
            <div className={`risk-card ${activePrediction.risk_segment.toLowerCase().replaceAll(" ", "-")}`}>
              <p className="panel-label">{activePrediction.mode || "Model output"}</p>
              <div className="score-ring" style={{ "--score": activePrediction.risk_score }}>
                <span>{activePrediction.risk_score}</span>
                <small>/100</small>
              </div>
              <h3>{activePrediction.risk_segment}</h3>
              <p>
                Default probability{" "}
                <strong>{(activePrediction.default_probability * 100).toFixed(1)}%</strong>
              </p>
              <div className="decision">
                {activePrediction.predicted_default ? <AlertTriangle size={18} /> : <BadgeCheck size={18} />}
                {activePrediction.predicted_default ? "Flag for review" : "No default flag"}
              </div>
            </div>

            <button className="primary-button" type="button" onClick={submitPrediction} disabled={loading}>
              {loading ? <Loader2 className="spin" size={18} /> : <ChevronRight size={18} />}
              Predict with model
            </button>
            {apiError && <p className="warning-text">{apiError}</p>}

            <div className="explain-box">
              <h4>Top expected drivers</h4>
              <ul>
                <li>Recent repayment delay</li>
                <li>Credit utilization ratio</li>
                <li>Payment-to-bill behavior</li>
                <li>Credit limit and bill volatility</li>
              </ul>
            </div>

            <div className="explain-box">
              <h4>Saved assessments</h4>
              <button className="secondary-button full-width" type="button" onClick={loadHistory}>
                Refresh history
              </button>
              <div className="history-list">
                {history.length === 0 ? (
                  <p className="muted-text">No saved predictions loaded yet.</p>
                ) : (
                  history.map((item) => (
                    <div className="history-item" key={item.id}>
                      <span>{item.risk_segment}</span>
                      <strong>{item.risk_score}/100</strong>
                    </div>
                  ))
                )}
              </div>
            </div>
          </aside>
        </div>

        <section className="batch-strip" id="batch">
          <div>
            <h3>Batch CSV Prediction</h3>
            <p>Upload support will call the same API and return scored customers as a downloadable CSV.</p>
          </div>
          <button className="secondary-button" type="button">
            <Upload size={17} />
            Upload CSV
          </button>
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
