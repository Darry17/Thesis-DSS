# Getting Started

This project was bootstrapped with [Vite](https://vitejs.dev/) and React for the frontend, and [FastAPI](https://fastapi.tiangolo.com/) for the backend.

## Installation

To install all required dependencies, run:

```bash
npm install @tailwindcss/vite axios comlink csv-parse csv-parser fs react react-dom react-dropzone react-router-dom tailwindcss @eslint/js @types/react @types/react-dom @vitejs/plugin-react eslint eslint-plugin-react eslint-plugin-react-hooks eslint-plugin-react-refresh globals vite worker-loader && npm audit fix --force
```

## Backend (FastAPI + SQLAlchemy + MySQL + Authentication)

To set up the backend environment, run:

```bash
# Install FastAPI and Uvicorn server
pip install fastapi uvicorn

# Install SQLAlchemy (ORM for database operations)
pip install sqlalchemy

# Install dotenv (for environment variable management)
pip install python-dotenv

# Install MySQL connectors
pip install mysql-connector-python
pip install PyMySQL

# Install authentication libraries
pip install PyJWT
pip install passlib[bcrypt]
```

## Available Scripts

In the project directory, you can run:

Start Frontend Server (React)

### `npm run dev`

Start Backend Server (FastAPI)

### `cd src/backend`

Then run:

### `uvicorn main:app --reload`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm run lint`

Launches the ESLint linter to check for code quality issues.

### `npm run build`

Builds the app for production to the `dist` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

### `npm run preview`

Locally preview the production build.\
This command will serve the built files from the `dist` folder.

## Project Dependencies

This project uses several key dependencies:

Frontend:

- React 18
- React Router DOM for navigation
- TailwindCSS for styling
- Axios for HTTP requests
- React Dropzone for file uploads
- CSV parsing utilities
- ESLint for code quality

Backend:

- FastAPI for backend APIs
- Uvicorn as ASGI server
- SQLAlchemy for ORM (Object Relational Mapping)
- dotenv for environment variable management
- MySQL Connector / PyMySQL for database connection
- PyJWT for authentication (JWT tokens)
- Passlib with bcrypt for password hashing

For a complete list of dependencies, see the `package.json` file.
For a complete list of dependencies, see the `package.json` file.

## Project Structure

```
├── public/               # Static images/assets
├── src/
│   ├── backend/           # FastAPI backend code
│   │   ├── auth.py
│   │   ├── configurations.py
│   │   ├── forecast.py
│   │   ├── history_logs.py
│   │   ├── storage.py
│   │   └── main.py
│   └── components/        # Frontend React components
├── storage/               # Data storage (daily, hourly, etc.)
├── testenv/               # Testing environment
├── .env                   # Environment Variables
├── requirements.txt       # Python dependencies
├── package.json           # NPM project configuration
├── vite.config.js         # Vite frontend configuration
├── README.md              # Project instructions (this file)
```
