// Registers the service worker and shows an "Install app" button when the browser allows it.
(() => {
  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch(() => {}));
  }
  const btn = document.getElementById("install");
  if (!btn) return;
  let deferred = null;
  window.addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); deferred = e; btn.hidden = false; });
  btn.addEventListener("click", async () => {
    if (!deferred) return;
    deferred.prompt();
    await deferred.userChoice.catch(() => {});
    deferred = null; btn.hidden = true;
  });
  window.addEventListener("appinstalled", () => { deferred = null; btn.hidden = true; });
})();
