import React, { useState } from "react";

const Settings = () => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [accessControl, setAccessControl] = useState("USER");
  const [editingId, setEditingId] = useState(null);
  const [accounts, setAccounts] = useState([
    {
      id: 1,
      username: "Darry Diaz",
      accessControl: "ADMIN",
      date: "2024/01/01",
    },
    {
      id: 2,
      username: "Mykel Santos",
      accessControl: "ADMIN",
      date: "2024/01/01",
    },
    {
      id: 3,
      username: "Rian Buhay",
      accessControl: "ADMIN",
      date: "2024/01/01",
    },
    {
      id: 4,
      username: "Juan Dela Cruz",
      accessControl: "USER",
      date: "2024/01/01",
    },
    {
      id: 5,
      username: "Isabella Rodriguez",
      accessControl: "USER",
      date: "2024/01/02",
    },
    {
      id: 6,
      username: "Justine Campos",
      accessControl: "USER",
      date: "2024/01/02",
    },
  ]);

  const handleAddAccount = (e) => {
    e.preventDefault();
    if (!username || !password) return;

    const currentDate = new Date()
      .toISOString()
      .split("T")[0]
      .replace(/-/g, "/");

    if (editingId) {
      // Update existing account
      setAccounts(
        accounts.map((account) =>
          account.id === editingId
            ? { ...account, username, accessControl, date: currentDate }
            : account
        )
      );
      setEditingId(null);
    } else {
      // Add new account
      const newAccount = {
        id: accounts.length + 1,
        username,
        accessControl,
        date: currentDate,
      };
      setAccounts([...accounts, newAccount]);
    }

    // Reset form
    setUsername("");
    setPassword("");
    setAccessControl("USER");
  };

  const handleEdit = (account) => {
    setEditingId(account.id);
    setUsername(account.username);
    setAccessControl(account.accessControl);
    setPassword(""); // Clear password field for security
  };

  const handleDelete = (accountId) => {
    if (window.confirm("Are you sure you want to delete this account?")) {
      setAccounts(accounts.filter((account) => account.id !== accountId));
    }
  };

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-2xl font-bold mb-4">Accounts</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left">
                  <th className="pb-3">Username</th>
                  <th className="pb-3">Access Control</th>
                  <th className="pb-3">Actions</th>
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
                        className="bg-green-100 text-green-600 px-4 py-1 rounded mr-2 hover:bg-green-200 font-medium">
                        EDIT
                      </button>
                      <button
                        onClick={() => handleDelete(account.id)}
                        className="bg-red-100 text-red-600 px-4 py-1 rounded hover:bg-red-200 font-medium">
                        DELETE
                      </button>
                    </td>
                    <td className="py-3">{account.date}</td>
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
            <div>
              <label className="block mb-1">Access Control</label>
              <select
                value={accessControl}
                onChange={(e) => setAccessControl(e.target.value)}
                className="w-full px-3 py-2 border rounded-md"
                required>
                <option value="ADMIN">ADMIN</option>
                <option value="USER">USER</option>
              </select>
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
                    setAccessControl("USER");
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

export default Settings;
