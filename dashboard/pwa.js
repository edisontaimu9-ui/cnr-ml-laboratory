// Registers the service worker and shows a prominent install banner when installing is possible.
(() => {
  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch(() => {}));
  }
  const bar = document.getElementById("install-bar");
  const btn = document.getElementById("install");
  const text = document.getElementById("install-text");
  const dismiss = document.getElementById("install-dismiss");
  if (!bar || !btn) return;

  const standalone = matchMedia("(display-mode: standalone)").matches || navigator.standalone === true;
  const KEY = "cnr-ml-install-dismissed";
  const dismissed = () => { try { return sessionStorage.getItem(KEY) === "1"; } catch { return false; } };
  const offset = () => document.documentElement.style.setProperty("--bar-h", bar.hidden ? "0px" : bar.offsetHeight + "px");
  const show = () => { if (standalone || dismissed()) return; bar.hidden = false; offset(); };
  const hide = () => { bar.hidden = true; offset(); };
  window.addEventListener("resize", offset);

  let deferred = null;
  window.addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); deferred = e; show(); });
  window.addEventListener("appinstalled", () => { deferred = null; hide(); });

  btn.addEventListener("click", async () => {
    if (!deferred) return;
    deferred.prompt();
    await deferred.userChoice.catch(() => {});
    deferred = null; hide();
  });
  dismiss.addEventListener("click", () => { try { sessionStorage.setItem(KEY, "1"); } catch {} hide(); });

  // iPhone/iPad Safari has no install prompt: show the manual steps instead.
  if (/iphone|ipad|ipod/i.test(navigator.userAgent) && !standalone) {
    text.textContent = "To install: tap Share, then Add to Home Screen.";
    btn.hidden = true; show();
  }
})();
