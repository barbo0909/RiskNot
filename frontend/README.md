# RiskNot Frontend

React + Vite product-style interface for single-customer credit risk scoring.

## Run

```bash
npm install
npm run dev
```

The app expects the FastAPI backend at:

```text
http://localhost:8000
```

You can override this with:

```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

Inputs are human-readable in the UI and mapped to the numeric model format before being sent to the API.
