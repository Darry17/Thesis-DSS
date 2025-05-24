import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const Login = () => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  // Check if the user is already logged in on component mount
  useEffect(() => {
    const token = localStorage.getItem("token")?.trim();
    if (token) {
      // Optionally validate the token with the backend
      axios
        .get("http://localhost:8000/api/validate-token", {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        })
        .then((response) => {
          // If token is valid, redirect to dashboard
          navigate("/history");
        })
        .catch((err) => {
          console.error("Token validation failed:", err);
          // If token is invalid, clear it
          localStorage.removeItem("token");
        });
    }
  }, [navigate]);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      // Send login request to FastAPI backend
      const response = await axios.post("http://localhost:8000/login", {
        username,
        password,
      });

      // Store the token in localStorage
      localStorage.setItem("token", response.data.access_token);

      // Redirect to dashboard
      navigate("/history");
    } catch (error) {
      console.error("Login failed:", error);
      if (error.response && error.response.status === 401) {
        setError("Invalid username or password. Please try again.");
      } else {
        setError("An error occurred. Please try again later.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center">
      <div className="bg-white rounded-lg shadow-md p-10 w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-1">Log In to System</h1>
          <p className="text-sm text-gray-600">DHR-ESN MODEL SYSTEM</p>
        </div>

        {error && (
          <div className="mb-4 text-red-500 text-center text-sm">{error}</div>
        )}

        <form onSubmit={handleLogin} className="space-y-4">
          <div>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="username"
              className="w-full px-4 py-2 rounded-full border border-gray-300 focus:outline-none focus:ring-2 focus:ring-green-500"
              required
              disabled={isLoading}
            />
          </div>

          <div>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="password"
              className="w-full px-4 py-2 rounded-full border border-gray-300 focus:outline-none focus:ring-2 focus:ring-green-500"
              required
              disabled={isLoading}
            />
          </div>

          <div className="text-center pt-2">
            <button
              type="submit"
              className={`bg-green-700 hover:bg-green-800 text-white font-medium py-2 px-6 rounded-full ${
                isLoading ? "opacity-50 cursor-not-allowed" : ""
              }`}
              disabled={isLoading}>
              {isLoading ? "Logging in..." : "Enter"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Login;
