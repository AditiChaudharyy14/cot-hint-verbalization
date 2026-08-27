# Socialyze — AI-Powered Instagram Analytics & Growth Platform

Socialyze is a full-stack web application that helps creators and businesses understand and grow their Instagram presence. Unlike traditional analytics dashboards that only report *what* happened, Socialyze uses analytics, machine learning, and a rule-based insights engine to explain *why* content performs and *what to do next* — turning raw metrics into data-driven decisions.

This project was developed as a final-year project (BSc Computing with Artificial Intelligence, London Metropolitan University).

---

## Overview

The platform connects to a real Instagram Business/Creator account through the Instagram Graph API, stores its performance data over time, and presents it through a multi-page dashboard. It provides live account analytics, per-post performance breakdowns, audience insights, optimal posting-time analysis, follower growth forecasting, and machine-learning-based engagement predictions with explainability.

---

## Key Features

- **Multi-user authentication** — secure sign up / log in with hashed passwords (bcrypt) and JWT session tokens, plus a password-reset flow.
- **Live Instagram integration** — pulls real profile and post data via the Instagram Graph API (v25.0, Instagram Login).
- **Overview dashboard** — headline KPIs (followers, reach, engagement) and a follower-growth chart.
- **Content performance** — a sortable table of posts ranked by real engagement, with format comparison.
- **Audience insights** — follower demographics with graceful handling of the Graph API's data-availability limitations.
- **Best time to post** — an engagement heatmap by day and hour, derived from the account's own post history.
- **Growth forecast** — a follower-count projection using an OLS trend model (statsmodels), with honest handling of small-sample data.
- **AI Insights** — a rule-based recommendation engine that analyses stored post metrics to produce specific, actionable growth advice grounded in real numbers.
- **ML engagement prediction + explainability** — a scikit-learn RandomForest model predicting post engagement from engineered features, with SHAP values explaining which factors drive engagement.
- **Snapshot collector** — a data-collection layer that periodically stores account snapshots, building the historical time series the Instagram API does not itself provide.
- **Automated testing** — 37 backend tests (pytest) and 3 frontend tests (Vitest), covering authentication, analytics, ML, error handling, and UI components.

---

## Technology Stack

**Backend**
- Python 3.12, FastAPI (REST API with auto-generated OpenAPI/Swagger docs)
- SQLAlchemy ORM, SQLite (development), PostgreSQL (production)
- JWT authentication (python-jose), password hashing (passlib / bcrypt)
- pandas, NumPy (data processing)
- scikit-learn (ML), SHAP (explainability), statsmodels (forecasting), joblib (model persistence)
- Instagram Graph API integration via `requests`

**Frontend**
- React 18 (Vite build tool)
- Tailwind CSS (dark navy + peach design system)
- Recharts (data visualisation)
- React Router (client-side routing), Context API (auth state)

**Testing & Tooling**
- pytest (backend), Vitest + React Testing Library (frontend)
- Git / GitHub (version control)

---

## Architecture

The application is split into two independently deployable services:

1. **FastAPI backend** — exposes REST endpoints for authentication, Instagram data retrieval, analytics computation, the ML pipeline, and per-user account connection. It reads the Instagram credentials for the connected account, stores snapshots and post metrics in the database, and serves computed analytics as JSON.

2. **React frontend** — a single-page application that consumes the backend API and renders the dashboard, styled with a custom dark theme.

A key design decision addresses a real Instagram Graph API constraint: the API only returns *current* values and provides no historical endpoint. Socialyze therefore includes a **snapshot collector** that records timestamped readings over time, making trend analysis and forecasting possible.

---

## Project Structure

```
socialyze/
├── backend/
│   ├── app/
│   │   ├── main.py              FastAPI entry point
│   │   ├── config.py            Environment configuration
│   │   ├── database.py          Database session
│   │   ├── models/              SQLAlchemy models (User, Snapshot, PostMetric)
│   │   ├── routers/             API endpoints (auth, instagram, analytics, insights, ml, connect)
│   │   ├── services/            Business logic (instagram_client, collector, insights, auth)
│   │   └── ml/                  Engagement model + SHAP explainability
│   ├── tests/                   pytest suite
│   └── requirements.txt
└── frontend/
    ├── src/                     App shell, routing, auth context
    ├── pages/                   Page components (Overview, Content, Audience, etc.)
    ├── components/              Reusable UI (KpiCard, Sidebar, charts)
    ├── lib/api.js               API client
    └── tests/                   Vitest suite
```

---

## Running Locally

**Prerequisites:** Python 3.12, Node.js, a Meta Developer app with an Instagram Business/Creator account connected.

**Backend**
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload  # runs at http://127.0.0.1:8000
```

**Frontend**
```bash
cd frontend
npm install
npm run dev                    # runs at http://localhost:5173
```

**Environment variables** (`backend/.env`, not committed):
```
IG_APP_ID=...
IG_APP_SECRET=...
IG_USER_ID=...
IG_ACCESS_TOKEN=...
JWT_SECRET=...
REDIRECT_URI=...
```

**Running the tests**
```bash
# Backend
cd backend && venv\Scripts\python -m pytest -v

# Frontend
cd frontend && npm test
```

---

## Scope & Limitations

This project is transparent about the boundaries of what the Instagram platform permits:

- **Public multi-user access** requires Instagram App Review (business verification and a formal approval process), which is outside the scope of a final-year project. The application is architected as a multi-user platform — users sign up and can connect their own Instagram Business account via OAuth — and is demonstrated using an authorised test account in Instagram Development Mode.
- **Audience demographics** are subject to the Graph API's own thresholds and are not returned for all accounts; the application handles this gracefully with a clear empty state rather than failing.
- **The forecasting and ML models** are demonstrated on a limited dataset. The value lies in the methodology — feature engineering, model training, honest evaluation (including reporting a low/negative R² where the sample is small), and SHAP-based explainability — which improves as more data is collected.
- **Password reset** implements the full token flow; email delivery is stubbed (the reset token is returned in the response) as email infrastructure is out of scope.

---

## Author

Aditi Chaudhary — BSc (Hons) Computing with Artificial Intelligence, London Metropolitan University.
