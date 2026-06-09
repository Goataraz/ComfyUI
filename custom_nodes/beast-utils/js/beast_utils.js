import { app } from "../../scripts/app.js";

const STYLE_ID = "beast-utils-styles";

function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = [
        "#beast-utils-toast {",
        "  position: fixed; top: 48px; left: 50%;",
        "  transform: translateX(-50%);",
        "  background: var(--comfy-menu-bg, #1a1a2e);",
        "  color: var(--input-text, #fff);",
        "  border: 1px solid var(--border-color, #444);",
        "  border-radius: 6px; padding: 8px 20px;",
        "  font-size: 14px; z-index: 99999;",
        "  pointer-events: none;",
        "  opacity: 1; transition: opacity 0.4s ease;",
        "}",
        "#beast-utils-toast.fade-out { opacity: 0; }",
        "#beast-utils-overlay {",
        "  position: fixed; inset: 0;",
        "  background: rgba(0,0,0,0.75);",
        "  display: flex; flex-direction: column;",
        "  align-items: center; justify-content: center;",
        "  z-index: 99999; color: #fff;",
        "  font-size: 18px; gap: 20px;",
        "}",
        ".beast-utils-spinner {",
        "  width: 48px; height: 48px;",
        "  border: 5px solid rgba(255,255,255,0.2);",
        "  border-top-color: #fff; border-radius: 50%;",
        "  animation: beast-spin 0.8s linear infinite;",
        "}",
        "@keyframes beast-spin { to { transform: rotate(360deg); } }",
    ].join("\n");
    document.head.appendChild(style);
}

function showToast(message, durationMs = 2000) {
    document.getElementById("beast-utils-toast")?.remove();
    const toast = document.createElement("div");
    toast.id = "beast-utils-toast";
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.classList.add("fade-out");
        setTimeout(() => toast.remove(), 400);
    }, durationMs);
}

function showRestartOverlay() {
    const overlay = document.createElement("div");
    overlay.id = "beast-utils-overlay";

    const spinner = document.createElement("div");
    spinner.className = "beast-utils-spinner";

    const label = document.createElement("div");
    label.textContent = "Restarting ComfyUI…";

    const sub = document.createElement("div");
    sub.style.cssText = "font-size:13px;opacity:0.6";
    sub.textContent = "Will reconnect automatically";

    overlay.appendChild(spinner);
    overlay.appendChild(label);
    overlay.appendChild(sub);
    document.body.appendChild(overlay);
}

function removeOverlay() {
    document.getElementById("beast-utils-overlay")?.remove();
}

async function pollUntilAlive() {
    while (true) {
        await new Promise(r => setTimeout(r, 2000));
        try {
            const res = await fetch("/system_stats", { signal: AbortSignal.timeout(3000) });
            if (res.ok) return;
        } catch {
            // still restarting — keep polling
        }
    }
}

async function onFreeMemory() {
    if (!confirm("Unload all models and free VRAM/RAM?\n\nThis clears everything currently in memory.")) return;
    try {
        const res = await fetch("/beast/free-memory", { method: "POST" });
        const data = await res.json();
        if (data.success) {
            showToast("Memory freed");
        } else {
            alert("Free memory failed: " + (data.error ?? "unknown error"));
        }
    } catch (err) {
        alert("Free memory request failed: " + err.message);
    }
}

async function onRestart() {
    if (!confirm("Restart ComfyUI?\n\nThe page will reconnect automatically (~30 seconds).")) return;
    try {
        const res = await fetch("/beast/restart", { method: "POST" });
        const data = await res.json();
        if (!data.success) {
            alert("Restart failed: " + (data.error ?? "unknown error"));
            return;
        }
    } catch (err) {
        alert("Restart request failed: " + err.message);
        return;
    }
    showRestartOverlay();
    await pollUntilAlive();
    removeOverlay();
    window.location.reload();
}

app.registerExtension({
    name: "BeastUtils.ActionBar",
    async setup() {
        injectStyles();
    },
    actionBarButtons: [
        {
            icon: "icon-[mdi--memory] size-4",
            tooltip: "Free VRAM & RAM (unload all models)",
            onClick: onFreeMemory,
        },
        {
            icon: "icon-[mdi--restart] size-4",
            tooltip: "Restart ComfyUI",
            onClick: onRestart,
        },
    ],
});
