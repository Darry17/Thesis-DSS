import { useNavigate } from "react-router-dom";

const Dashboard = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex overflow-y-hidden">
      {/* Background Images and Overlay */}
      <div
        className="fixed inset-0"
        style={{
          backgroundImage: `url(/wind-img.png), url(/solar-img.png)`,
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundBlendMode: "overlay",
          zIndex: -1,
        }}
      />
      <div className="fixed inset-0 bg-black/60" style={{ zIndex: -1 }} />

      {/* Content */}
      <div className="relative z-10 flex-1 flex items-center justify-center p-8 text-white">
        <div className="max-w-2xl text-center">
          <h1 className="text-6xl md:text-7xl font-bold mb-4">DHR - ESN</h1>
          <p className="text-5xl md:text-6xl font-bold mb-4 whitespace-nowrap">
            Forecasting Model
          </p>
          <div className="flex justify-center mt-6">
            <button
              onClick={() => navigate("/forecast")}
              className="bg-white w-[300px] h-[70px] text-black rounded-4xl px-4 py-2 font-bold text-2xl hover:bg-gray-100 transition">
              Forecast
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
