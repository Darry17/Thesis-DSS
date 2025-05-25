# Decision Support System - Time Series Forecasting

A web-based Decision Support System built with React and FastAPI for time series forecasting using DHR, ESN, and DHR-ESN models.

## Features

- User authentication and authorization
- CSV file upload and processing
- Time series forecasting using multiple models
- Interactive data visualization
- Forecast history logging
- Secure API endpoints

## Technology Stack

### Frontend

- React 18
- Vite for build tooling
- TailwindCSS for styling
- React Router for navigation
- Axios for API calls
- React Dropzone for file uploads
- Papa Parse for CSV handling

### Backend

- FastAPI framework
- SQLAlchemy ORM
- MySQL database
- JWT authentication
- Scientific computing:
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib

## Quick Start

1. **Clone the repository:**

```bash
git clone https://github.com/Darry17/Thesis-DSS.git
cd Thesis-DSS
```

2. **Install Dependencies:**

Frontend:

```bash
npm install @tailwindcss/vite axios comlink fs papaparse react react-dom react-dropzone react-router-dom @eslint/js @types/react @types/react-dom @vitejs/plugin-react eslint eslint-plugin-react eslint-plugin-react-hooks eslint-plugin-react-refresh globals vite worker-loader unocss react-chartjs-2
```

Backend:

```bash
pip install -r requirements.txt
```

3. **Configure Environment:**

- Copy `.env.example` to `.env`
- Update database and JWT settings

4. **Start Development Servers:**

Frontend:

```bash
npm run dev
```

Backend:

```bash
cd src/backend
uvicorn main:app --reload
```

Access the application at `http://localhost:3000`

## Project Structure

```
├── public/               # Static assets
├── src/
│   ├── backend/         # FastAPI backend
│   │   ├── forecasts/   # Forecast results
│   │   ├── models/      # ML models (DHR, ESN, DHR-ESN)
│   │   ├── temp/        # Temporary CSV storage
│   │   ├── auth.py      # Authentication logic
│   │   ├── main.py      # FastAPI application
│   │   └── ...
│   └── components/      # React components
├── .env                 # Environment variables
├── requirements.txt     # Python dependencies
└── package.json        # NPM dependencies
```

## API Documentation

Once the backend server is running, access the API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Available Scripts

- `npm run dev` - Start frontend development server
- `npm run build` - Build frontend for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
