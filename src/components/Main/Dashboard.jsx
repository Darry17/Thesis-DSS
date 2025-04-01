import { useNavigate } from "react-router-dom";

const Dashboard = () => {
  const navigate = useNavigate();

  return (
    <div className="relative min-h-screen flex">
      <div
        className="fixed inset-0 overflow-hidden"
        style={{
          backgroundImage: `url(/wind-img.png), url(/solar-img.png)`,
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundBlendMode: "overlay",
          zIndex: -1,
        }}
      />
      <div className="fixed inset-0 bg-black/60" style={{ zIndex: -1 }} />
      <div className="relative z-10 flex-1 flex items-center justify-center p-8 text-white">
        <div className="max-w-2xl">
          <h1 className="text-[80px] font-bold mb-4 flex justify-center">
            DHR - ESN
          </h1>
          <p className="text-[80px] font-bold mb-4 text-center whitespace-nowrap">
            Forecasting Model
          </p>
          <div className="flex justify-center mt-8">
            <button
              onClick={() => navigate("/forecast")}
              className="bg-white w-[300px] h-[70px] text-black rounded-4xl px-4 py-2 font-bold text-2xl">
              Forecast
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
