import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  BadgeCheck,
  ChevronRight,
  Loader2,
  ShieldCheck,
} from "lucide-react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const repaymentOptions = [
  { value: -2, label: "No statement issued" },
  { value: -1, label: "Paid in full" },
  { value: 0, label: "Current / minimum paid" },
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

function monthLabel(monthOffset) {
  const date = new Date();
  date.setMonth(date.getMonth() - monthOffset);
  return new Intl.DateTimeFormat("en", {
    month: "long",
    year: "numeric",
  }).format(date);
}

const monthFields = [
  { pay: "PAY_0", bill: "BILL_AMT1", paid: "PAY_AMT1", monthOffset: 0 },
  { pay: "PAY_2", bill: "BILL_AMT2", paid: "PAY_AMT2", monthOffset: 1 },
  { pay: "PAY_3", bill: "BILL_AMT3", paid: "PAY_AMT3", monthOffset: 2 },
  { pay: "PAY_4", bill: "BILL_AMT4", paid: "PAY_AMT4", monthOffset: 3 },
  { pay: "PAY_5", bill: "BILL_AMT5", paid: "PAY_AMT5", monthOffset: 4 },
  { pay: "PAY_6", bill: "BILL_AMT6", paid: "PAY_AMT6", monthOffset: 5 },
];

function numberValue(value) {
  if (value === "" || value === null || value === undefined) return 0;
  return Number.isFinite(Number(value)) ? Number(value) : 0;
}

function normalizeCustomer(customer) {
  return {
    LIMIT_BAL: numberValue(customer.LIMIT_BAL),
    SEX: numberValue(customer.SEX),
    EDUCATION: numberValue(customer.EDUCATION),
    MARRIAGE: numberValue(customer.MARRIAGE),
    AGE: numberValue(customer.AGE),
    PAY_0: numberValue(customer.PAY_0),
    PAY_2: numberValue(customer.PAY_2),
    PAY_3: numberValue(customer.PAY_3),
    PAY_4: numberValue(customer.PAY_4),
    PAY_5: numberValue(customer.PAY_5),
    PAY_6: numberValue(customer.PAY_6),
    BILL_AMT1: numberValue(customer.BILL_AMT1),
    BILL_AMT2: numberValue(customer.BILL_AMT2),
    BILL_AMT3: numberValue(customer.BILL_AMT3),
    BILL_AMT4: numberValue(customer.BILL_AMT4),
    BILL_AMT5: numberValue(customer.BILL_AMT5),
    BILL_AMT6: numberValue(customer.BILL_AMT6),
    PAY_AMT1: numberValue(customer.PAY_AMT1),
    PAY_AMT2: numberValue(customer.PAY_AMT2),
    PAY_AMT3: numberValue(customer.PAY_AMT3),
    PAY_AMT4: numberValue(customer.PAY_AMT4),
    PAY_AMT5: numberValue(customer.PAY_AMT5),
    PAY_AMT6: numberValue(customer.PAY_AMT6),
  };
}

function formatContribution(value) {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}`;
}

function App() {
  const [customer, setCustomer] = useState(initialCustomer);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState("");

  const updateValue = (field, value) => {
    setCustomer((current) => ({ ...current, [field]: value === "" ? "" : numberValue(value) }));
  };

  const updateRepaymentStatus = (month, value) => {
    const status = numberValue(value);
    setCustomer((current) => {
      const next = { ...current, [month.pay]: status };
      if (status === -2) {
        next[month.bill] = 0;
        next[month.paid] = 0;
      }
      if (status === -1) {
        next[month.paid] = Math.max(0, numberValue(current[month.bill]));
      }
      return next;
    });
  };

  const updateStatementBalance = (month, value) => {
    const balance = value === "" ? "" : numberValue(value);
    setCustomer((current) => {
      const next = { ...current, [month.bill]: balance };
      if (numberValue(current[month.pay]) === -1) {
        next[month.paid] = balance === "" ? "" : Math.max(0, balance);
      }
      if (numberValue(current[month.pay]) === -2) {
        next[month.paid] = 0;
      }
      return next;
    });
  };

  const submitPrediction = async () => {
    setLoading(true);
    setApiError("");
    const payload = normalizeCustomer(customer);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        let message = `API returned ${response.status}`;
        try {
          const errorPayload = await response.json();
          message = errorPayload.detail || message;
        } catch {
          const text = await response.text();
          message = text || message;
        }
        throw new Error(message);
      }
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      setApiError(error.message || "Prediction failed. Restart the backend and try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand">
          <div className="brand-mark">
            <ShieldCheck size={24} />
          </div>
          <div>
            <p className="eyebrow">Credit Risk Suite</p>
            <h1>RiskNot</h1>
          </div>
        </div>
       
      </header>

      <section className="workspace" id="assessment">
        <div className="content-grid">
          <form className="form-surface" onSubmit={(event) => event.preventDefault()}>
            <section className="section-block">
              <div className="section-heading">
                <h3>Customer profile</h3>
                
              </div>
              <div className="profile-grid">
                <label>
                  Approved credit limit
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
                <h3>Last 6 months</h3>
                <span>Repayment status, statement balance, and payment amount.</span>
              </div>
              <div className="month-list">
                {monthFields.map((month) => (
                  <div className="month-row" key={month.pay}>
                    <div className="month-title">
                      <span>{monthLabel(month.monthOffset)}</span>
                      <small>{month.monthOffset === 0 ? "Current month" : `${month.monthOffset} month${month.monthOffset > 1 ? "s" : ""} ago`}</small>
                    </div>
                    <label>
                      Repayment status
                      <select
                        value={customer[month.pay]}
                        onChange={(event) => updateRepaymentStatus(month, event.target.value)}
                      >
                        {repaymentOptions.map((option) => (
                          <option value={option.value} key={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Statement balance
                      <input
                        type="number"
                        value={customer[month.bill]}
                        onChange={(event) => updateStatementBalance(month, event.target.value)}
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
            <div className={`risk-card ${prediction ? prediction.risk_segment.toLowerCase().replaceAll(" ", "-") : "empty-risk"}`}>
              <div className="score-ring" style={{ "--score": prediction?.risk_score || 0 }}>
                <span>{prediction ? prediction.risk_score : "--"}</span>
                <small>/100</small>
              </div>
              <h3>{prediction ? prediction.risk_segment : "Run assessment"}</h3>
              <p>
                Default probability{" "}
                <strong>{prediction ? `${(prediction.default_probability * 100).toFixed(1)}%` : "not calculated"}</strong>
              </p>
              {prediction ? (
                <>
                  <div className="decision">
                    {prediction.predicted_default ? <AlertTriangle size={18} /> : <BadgeCheck size={18} />}
                    {prediction.predicted_default ? "Flag for review" : "No default flag"}
                  </div>
                </>
              ) : (
                <p className="status-line muted-text">Fill the form, then run the model to save a real assessment.</p>
              )}
            </div>

            <button className="primary-button" type="button" onClick={submitPrediction} disabled={loading}>
              <span className="button-icon">
                {loading ? <Loader2 className="spin" size={18} /> : <ChevronRight size={18} />}
              </span>
              <span>Run risk assessment</span>
            </button>
            <p className={`api-message ${apiError ? "warning-text" : ""}`}>{apiError}</p>

            {prediction?.explanation && (
              <section className="explain-panel" aria-label="Prediction explanation">
                <div>
                  <h4>Raises risk</h4>
                  <ul>
                    {prediction.explanation.risk_increasing.slice(0, 4).map((item) => (
                      <li key={`up-${item.raw_feature}`}>
                        <span>{item.feature}</span>
                        <strong>{formatContribution(item.contribution)}</strong>
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4>Lowers risk</h4>
                  <ul>
                    {prediction.explanation.risk_decreasing.slice(0, 4).map((item) => (
                      <li key={`down-${item.raw_feature}`}>
                        <span>{item.feature}</span>
                        <strong>{formatContribution(item.contribution)}</strong>
                      </li>
                    ))}
                  </ul>
                </div>
              </section>
            )}
          </aside>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
