const $ = id => document.getElementById(id);
const LABELS = {safe: "Safe", suspicious: "Suspicious", likely_phishing: "Likely Phishing"};

$("form").addEventListener("submit", async e => {
  e.preventDefault();
  $("error").hidden = true;
  const btn = e.target.querySelector("button");
  btn.disabled = true;
  try {
    const r = await fetch("/predict", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({url: $("url").value.trim()})});
    if (!r.ok) throw new Error((await r.json()).detail || "Request failed");
    const d = await r.json();
    $("badge").textContent = LABELS[d.verdict];
    $("badge").className = d.verdict;
    $("score").textContent = d.score;
    $("reasons").replaceChildren(...d.top_features.map(f => {
      const li = document.createElement("li");
      li.textContent = (f.direction === "phishing" ? "⚠ " : "✓ ") + f.explanation;
      return li;
    }));
    $("result").hidden = false;
  } catch (err) {
    $("error").textContent = String(err.message || err);
    $("error").hidden = false;
  } finally { btn.disabled = false; }
});

document.querySelectorAll(".examples a").forEach(a => a.addEventListener("click", e => {
  e.preventDefault();
  $("url").value = a.dataset.url;
  $("form").requestSubmit();
}));

fetch("/info").then(r => r.json()).then(i => {
  const m = i.test_metrics && i.test_metrics.stacked_platt;
  if (!m) return;
  $("footer").textContent = `Character n-gram + gradient boosting | ${i.n_total.toLocaleString()} URLs | held-out domains: AUC ${m.roc_auc.toFixed(3)}, ${(m.recall*100).toFixed(0)}% of phishing caught at ${(m.fpr*100).toFixed(0)}% false alarms | built by Kaushik Vootukur`;
}).catch(() => {});
