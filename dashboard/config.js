// Chooses the ML API for the environment. Edit if your hostnames differ.
window.CNR_ML_CONFIG = {
  apiBase: ["localhost", "127.0.0.1"].includes(location.hostname)
    ? "http://localhost:8000"
    : "https://ml.chakudya.xyz",
};
