import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

const Accounts = () => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [accessControl, setAccessControl] = useState("ADMIN");
  const [editingId, setEditingId] = useState(null);
  const [accounts, setAccounts] = useState([]);
  const navigate = useNavigate();

  // Fetch users from backend on component mount
  useEffect(() => {
    const fetchAccounts = async () => {
      try {
        const token = localStorage.getItem("token");

        if (!token) {
          console.error("No token found, redirecting to login...");
          navigate("/");
          return;
        }

        const response = await fetch("http://localhost:8000/users/", {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        });

        if (response.status === 401) {
          console.error("Unauthorized: Invalid or expired token");
          localStorage.removeItem("token");
          navigate("/login");
          return;
        }

        if (!response.ok) {
          throw new Error("Failed to fetch accounts");
        }

        const data = await response.json();
        setAccounts(
          data.map((user) => ({
            ...user,
            accessControl: user.access_control,
            date: user.created_at,
          }))
        );
      } catch (error) {
        console.error("Error fetching accounts:", error);
      }
    };
    fetchAccounts();
  }, [navigate]);

  // Handle form submission for adding or updating users
  const handleAddAccount = async (e) => {
    e.preventDefault();
    if (!username) return;
    const token = localStorage.getItem("token");

    if (!token) {
      navigate("/login");
      return;
    }

    try {
      if (editingId) {
        // Update existing user
        const updateData = { username, access_control: accessControl };
        if (password) updateData.password = password;
        const response = await fetch(
          `http://localhost:8000/users/${editingId}`,
          {
            method: "PUT",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify(updateData),
          }
        );
        if (response.status === 401) {
          navigate("/login");
          return;
        }
        if (!response.ok) {
          throw new Error("Failed to update account");
        }
        const updatedUser = await response.json();
        setAccounts(
          accounts.map((account) =>
            account.id === editingId
              ? {
                  ...account,
                  username: updatedUser.username,
                  accessControl: updatedUser.access_control,
                }
              : account
          )
        );
        alert("Updated Account Successfully"); // Native alert
        setEditingId(null);
      } else {
        // Add new user
        if (!password) return;
        const newUserData = {
          username,
          password,
          access_control: accessControl,
        };
        const response = await fetch("http://localhost:8000/users", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify(newUserData),
        });
        if (response.status === 401) {
          navigate("/login");
          return;
        }
        if (!response.ok) {
          throw new Error("Failed to add account");
        }
        const newUser = await response.json();
        setAccounts([
          ...accounts,
          {
            id: newUser.id,
            username: newUser.username,
            accessControl: newUser.access_control,
            date: newUser.created_at,
          },
        ]);
        alert("Added Account Successfully"); // Native alert
      }
      // Reset form
      setUsername("");
      setPassword("");
      setAccessControl("ADMIN");
    } catch (error) {
      console.error("Error:", error);
      alert("Error: " + error.message); // Optional: Show error alert
    }
  };

  // Handle user deletion
  const handleDelete = async (accountId) => {
    const confirmDelete = window.confirm(
      "Are you sure you want to delete this account?"
    );
    if (confirmDelete) {
      try {
        const token = localStorage.getItem("token");
        if (!token) {
          navigate("/login");
          return;
        }
        const response = await fetch(
          `http://localhost:8000/users/${accountId}`,
          {
            method: "DELETE",
            headers: {
              Authorization: `Bearer ${token}`,
              "Content-Type": "application/json",
            },
          }
        );
        if (response.status === 401) {
          navigate("/login");
          return;
        }
        if (!response.ok) {
          throw new Error("Failed to delete account");
        }
        setAccounts(accounts.filter((account) => account.id !== accountId));
      } catch (error) {
        console.error("Error deleting account:", error);
        alert("Error deleting account: " + error.message); // Optional: Show error alert
      }
    }
  };

  // Set form fields for editing
  const handleEdit = (account) => {
    setEditingId(account.id);
    setUsername(account.username);
    setAccessControl(account.accessControl);
    setPassword("");
  };

  return (
    <div className="p-6 min-h-screen">
      <div className="max-w-5xl mx-auto">
        <h2 className="text-4xl font-bold mb-4">Accounts</h2>
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left">
                  <th className="pb-3">Username</th>
                  <th className="pb-3">Access Control</th>
                  <th className="pb-3"></th>
                  <th className="pb-3">Date</th>
                </tr>
              </thead>
              <tbody>
                {accounts.map((account) => (
                  <tr key={account.id} className="border-t">
                    <td className="py-3">{account.username}</td>
                    <td className="py-3 text-blue-600">
                      {account.accessControl}
                    </td>
                    <td className="py-3">
                      <button
                        onClick={() => handleEdit(account)}
                        className="bg-green-100 text-green-600 px-4 py-1 rounded mr-10 hover:bg-green-200 font-medium">
                        EDIT
                      </button>
                      <button
                        onClick={() => handleDelete(account.id)}
                        className="bg-red-100 text-red-600 px-4 py-1 rounded hover:bg-red-200 font-medium">
                        DELETE
                      </button>
                    </td>
                    <td className="py-3">
                      {new Date(account.date).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-2xl font-bold mb-4">
            {editingId ? "Edit Account" : "Add Account"}
          </h2>
          <form onSubmit={handleAddAccount} className="space-y-4">
            <div>
              <label className="block mb-1">Username</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full px-3 py-2 border rounded-md"
                placeholder="Enter username"
                required
              />
            </div>
            <div>
              <label className="block mb-1">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-3 py-2 border rounded-md"
                placeholder={
                  editingId
                    ? "Leave blank to keep current password"
                    : "Enter password"
                }
                required={!editingId}
              />
            </div>
            <div className="flex gap-2">
              <button
                type="submit"
                className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600">
                {editingId ? "Update Account" : "Add Account"}
              </button>
              {editingId && (
                <button
                  type="button"
                  onClick={() => {
                    setEditingId(null);
                    setUsername("");
                    setPassword("");
                    setAccessControl("ADMIN");
                  }}
                  className="bg-gray-500 text-white px-4 py-2 rounded-md hover:bg-gray-600">
                  Cancel
                </button>
              )}
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Accounts;
