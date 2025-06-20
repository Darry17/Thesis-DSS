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
- React Chart.js 2 for visualizations

### Backend

- FastAPI framework
- SQLAlchemy ORM
- MySQL database
- JWT authentication
- Scientific computing: NumPy, Pandas, Scikit-learn, Matplotlib

## Prerequisites

- [Node.js](https://nodejs.org/) (v16 or higher) and npm
- [Python](https://www.python.org/) (v3.8 or higher) and pip
- [XAMPP](https://www.apachefriends.org/) or a standalone MySQL server
- Git for cloning the repository

## Quick Start

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Darry17/Thesis-DSS.git
   cd Thesis-DSS
   ```

2. **Install Dependencies**:

   - Frontend:
     ```bash
     npm install
     ```
   - Backend:
     ```bash
     pip install -r requirements.txt
     ```

3. **Configure Environment**:

   - Copy `.env.example` to `.env` in the project root.
   - Update the `.env` file with your settings, e.g.:
     ```env
     DATABASE_URL=mysql://root:@localhost/thesis
     JWT_SECRET=your_secret_key_here
     ```
   - Ensure the `DATABASE_URL` matches your MySQL configuration (username, password, host, and database name).

4. **Database Setup**:
   This project uses a MySQL database. The schema and initial data are in `sql/thesis.sql`. **The MySQL database must be running, and the schema must be imported before starting the FastAPI backend, or it will fail to run or throw errors.**

   ### Steps

   - **Start MySQL Server**:
     - For XAMPP: Open the XAMPP Control Panel and start the MySQL module.
     - For standalone MySQL: Ensure the server is running (e.g., `sudo systemctl start mysql` on Linux or equivalent).
   - **Create the Database**:
     - Open a MySQL client:
       - **phpMyAdmin**: Go to `http://localhost/phpmyadmin` (XAMPP must be running).
       - **Command Line**: Run `mysql -u root -p` (default XAMPP credentials: username `root`, password empty).
     - Create the database:
       ```sql
       CREATE DATABASE thesis;
       ```
   - **Import the Schema**:
     - In phpMyAdmin:
       1. Select the `thesis` database.
       2. Click the "Import" tab.
       3. Choose `sql/thesis.sql` from the project directory.
       4. Click "Go" to import.
     - In the command line:
       ```bash
       mysql -u root -p thesis < sql/thesis.sql
       ```
       Enter your password when prompted.
   - **Verify**: Check tables in phpMyAdmin or run `SHOW TABLES;` in the command line.
   - **Admin Accounts**: After importing `sql/thesis.sql`, the following admin accounts are available for testing:
     | Username | Password |
     |----------|-----------|
     | admin1 | admin001 |
     | admin2 | admin002 |
     | admin3 | admin003 |
     **Note**: These accounts are for development/testing only. Change passwords for production use.

5. **Start Development Servers**:

   - **Ensure the MySQL database is running** to avoid errors in the FastAPI backend.
   - Frontend (runs on `http://localhost:3000`):
     ```bash
     npm run dev
     ```
   - Backend (runs on `http://localhost:8000`):
     ```bash
     cd src/backend
     uvicorn main:app --reload
     ```

6. **Access the Application**:
   - Open `http://localhost:3000` in your browser and log in using the admin accounts above.

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
├── sql/                 # Database schema (thesis.sql)
├── .env                 # Environment variables
├── requirements.txt     # Python dependencies
├── package.json         # NPM dependencies
```

## API Documentation

Once the backend is running, access the API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Available Scripts

- `npm run dev` - Start frontend development server
- `npm run build` - Build frontend for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Troubleshooting

- **MySQL fails to start**: Ensure XAMPP’s MySQL module is running or no other MySQL instance is using port 3306.
- **Database already exists**: Drop it with `DROP DATABASE thesis;` before recreating (use cautiously).
- **FastAPI errors**: Ensure the MySQL database is running and `sql/thesis.sql` is imported correctly, as the backend requires a valid database connection.
- **Login issues**: Use the admin accounts listed above for testing. Verify the database import includes these accounts.
- **Port conflicts**: If ports `3000` or `8000` are in use, update them in `.env` or stop conflicting processes.
- **Dependency issues**: Ensure Node.js (v16+) and Python (v3.8+) are installed. Re-run `npm install` or `pip install -r requirements.txt` if errors occur.
- **Import errors**: Verify `sql/thesis.sql` exists and has valid SQL syntax.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/Darry17/Thesis-DSS).
