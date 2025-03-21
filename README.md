# Getting Started

This project was bootstrapped with [Vite](https://vitejs.dev/) and React.

## Installation

To install all required dependencies, run:

```bash
npm install @tailwindcss/vite axios comlink csv-parse csv-parser fs react react-dom react-dropzone react-router-dom tailwindcss @eslint/js @types/react @types/react-dom @vitejs/plugin-react eslint eslint-plugin-react eslint-plugin-react-hooks eslint-plugin-react-refresh globals vite worker-loader && npm audit fix --force
```

## Available Scripts

In the project directory, you can run:

### `npm run dev`

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

- React 18
- React Router DOM for navigation
- TailwindCSS for styling
- Axios for HTTP requests
- React Dropzone for file uploads
- CSV parsing utilities
- ESLint for code quality

For a complete list of dependencies, see the `package.json` file.

## Project Structure

```
├── src/           # Source files
├── public/        # Static assets
├── storage/       # Storage directory
├── vite.config.js # Vite configuration
└── eslint.config.js # ESLint configuration
```
