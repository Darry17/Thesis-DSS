const Dashboard = () => {
  return (
    <div className="relative min-h-screen flex">
      {/* Dual Background with Blend Mode */}
      <div
        className="absolute inset-0"
        style={{
          backgroundImage: `
        url(/wind-img.png),
        url(/solar-img.png)
      `,
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundBlendMode: "overlay",
        }}
      />

      {/* Dark Overlay */}
      <div className="absolute inset-0 bg-black/60" />

      {/* Content */}
      <div className="relative z-10 flex-1 flex items-center justify-center p-8 text-white">
        <div className="bg-black/30 p-8 rounded-xl backdrop-blur-sm max-w-2xl">
          <h1 className="text-3xl font-bold mb-4">Dashboard</h1>
          <p className="text-lg">Your content here...</p>
        </div>
      </div>
    </div>
  );
};
export default Dashboard;
