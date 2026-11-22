// Theme preference (per viewer). Light is the default look; "system" follows the operating system.
// The first paint is handled by the small inline script in index.html, which uses the same key and rule.
export type Pref = "light" | "dark" | "system";
const KEY = "finsight-theme";
const osDark = window.matchMedia("(prefers-color-scheme: dark)");

function stored(): Pref {
  try {
    const t = localStorage.getItem(KEY);
    if (t === "dark" || t === "system") return t;
  } catch { /* storage may be unavailable */ }
  return "light";
}

let pref: Pref = stored();

function apply() {
  document.documentElement.dataset.theme = pref === "dark" || (pref === "system" && osDark.matches) ? "dark" : "light";
}

export const getTheme = () => pref;

export function setTheme(next: Pref) {
  pref = next;
  try { localStorage.setItem(KEY, next); } catch { /* storage may be unavailable */ }
  apply();
}

export function initTheme() {
  apply();
  osDark.addEventListener("change", apply);
}
