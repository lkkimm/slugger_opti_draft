import os, io, base64, sys, logging
from typing import Dict, Tuple
from flask import Flask, request, jsonify, render_template_string, send_file
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

# -------------------------------------------------------
# CONFIG: batters and which side they're hitting from
# (add more entries + file paths as you get real data)
# -------------------------------------------------------
BATTERS: Dict[str, Dict] = {
    "dickerson_L": {
        "label": "Corey Dickerson (L)",
        "batter_name": "Corey Dickerson",
        "batter_hand": "L",
        # for now, same file for all pitcher hands; you can split later
        "file": "Dickerson_L.csv"
    },
    "dickerson_R": {
        "label": "Corey Dickerson (R)",
        "batter_name": "Corey Dickerson",
        "batter_hand": "R",
        "file": "Dickerson_R.csv"
    },
    # example Turner entries – point these to real files when you have them
    # "turner_L": {
    #     "label": "Trea Turner (L)",
    #     "batter_name": "Trea Turner",
    #     "batter_hand": "L",
    #     "file": "Turner_L.csv"
    # },
    # "turner_R": {
    #     "label": "Trea Turner (R)",
    #     "batter_name": "Trea Turner",
    #     "batter_hand": "R",
    #     "file": "Turner_R.csv"
    # },
}

STADIUM_IMAGE = "stadium.png"  # optional background image
LAST_CSV_PATH = "optimized_positions.csv"

# -------------------------------------------------------
# HTML UI – simple but close to final product flow
# -------------------------------------------------------
HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Outfield Positioning Optimizer</title>
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
           margin: 2rem; background: #111; color: #fff; }
    h1 { margin-bottom: 1rem; }
    .layout { display: flex; gap: 2rem; }
    .panel { background:#1f1f1f; padding:1rem 1.25rem; border-radius:12px; width:260px; }
    label { font-size:0.9rem; font-weight:600; display:block; margin-top:0.75rem; }
    select, button { width:100%; padding:0.45rem; margin-top:0.25rem; border-radius:6px; border:1px solid #444; }
    button { background:#0077ff; color:#fff; cursor:pointer; font-weight:600; border:none; }
    button:hover { background:#005fd6; }
    #screen { flex:1; background:#000; border-radius:12px; padding:0.5rem; display:flex; justify-content:center; align-items:center; }
    img.tv { max-width:100%; border-radius:12px; }
    .caption { margin-top:0.5rem; font-size:1.1rem; text-align:center; }
    .caption span.name { color:#4aa3ff; font-weight:700; }
    .coords { margin-top:0.5rem; font-size:0.9rem; text-align:center; }
  </style>
</head>
<body>
  <h1>Outfield Positioning Optimizer</h1>
  <div class="layout">
    <div class="panel">
      <label>Pitcher Handedness</label>
      <select id="pitcher_hand">
        <option value="RHP">RHP</option>
        <option value="LHP">LHP</option>
      </select>

      <label>Batter &amp; Side</label>
      <select id="batter_id">
        {% for bid, meta in batters.items() %}
          <option value="{{ bid }}">{{ meta.label }}</option>
        {% endfor %}
      </select>

      <button type="button" onclick="run()">Generate Positions</button>
      <p style="margin-top:0.75rem;font-size:0.8rem;color:#bbb;">
        Demo currently uses sample data / simulated spray charts. Trackman integration to come.
      </p>
    </div>

    <div id="screen">
      <div style="text-align:center;color:#888;">
        Select pitcher handedness and batter to generate optimized outfield positioning.
      </div>
    </div>
  </div>

<script>
async function run() {
  const pitcher_hand = document.getElementById("pitcher_hand").value;
  const batter_id = document.getElementById("batter_id").value;
  const res = await fetch("/api/compute", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ batter_id, pitcher_hand })
  });
  const data = await res.json();
  if (!data.ok) {
    alert("Error: " + data.error);
    return;
  }
  const img = `<img class="tv" src="data:image/png;base64,${data.image_base64}" />`;
  const coords = Object.entries(data.positions)
      .map(([f,[x,y]]) => `${f}: X=${x.toFixed(1)}, Y=${y.toFixed(1)}`)
      .join(" • ");
  const caption = `
    <div class="caption">
      <span class="name">${data.batter_label}</span>
      &nbsp;&nbsp;vs.&nbsp;${data.pitcher_hand}
    </div>
    <div class="coords">${coords}</div>`;
  document.getElementById("screen").innerHTML = img + caption;
}
</script>
</body>
</html>
"""

# -------------------------------------------------------
# Data loading
# -------------------------------------------------------
def load_batted_balls(batter_id: str, pitcher_hand: str) -> pd.DataFrame:
    """
    For now, we ignore pitcher_hand when picking files (demo mode).
    Later you can have separate files per pitcher hand, e.g.
    Dickerson_L_vs_RHP.csv vs Dickerson_L_vs_LHP.csv.
    """
    meta = BATTERS.get(batter_id)
    if meta is None:
        raise ValueError(f"Unknown batter_id: {batter_id}")

    path = meta.get("file")
    if path and os.path.exists(path):
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            # for .xlsm etc
            df = pd.read_excel(path, engine="openpyxl")

        # normalize to x,y
        df.columns = [c.lower().strip() for c in df.columns]
        if not {"x","y"}.issubset(df.columns):
            num = df.select_dtypes(include="number").columns.tolist()
            if len(num) < 2:
                raise ValueError(f"{path} has no numeric x,y columns")
            df = df.rename(columns={num[0]:"x", num[1]:"y"})
        df = df[["x","y"]].dropna()
        # clip to outfield-ish region
        df = df[(df["x"].between(40,260)) & (df["y"].between(180,420))]
        if len(df) > 0:
            return df.reset_index(drop=True)

    # Fallback: generate synthetic spray if file missing
    rng = np.random.default_rng(0 if meta["batter_hand"]=="L" else 1)
    n = 150
    x = rng.uniform(50, 250, n)
    y = rng.uniform(220, 380, n)
    return pd.DataFrame({"x": x, "y": y})

# -------------------------------------------------------
# Optimization
# -------------------------------------------------------
def optimize_outfield(df: pd.DataFrame) -> Dict[str, Tuple[float,float]]:
    """
    3-layer loop over LF, CF, RF positions using a simple distance-based
    reward/penalty: closer to batted balls is better.
    """
    lf_grid = [(x,y) for x in range(70,120,10)  for y in range(260,330,10)]
    cf_grid = [(x,y) for x in range(120,180,10) for y in range(310,380,10)]
    rf_grid = [(x,y) for x in range(180,230,10) for y in range(260,330,10)]

    bx = df["x"].to_numpy()
    by = df["y"].to_numpy()

    best_score = float("inf")
    best = {}

    for lf in lf_grid:
        dlf = np.hypot(bx - lf[0], by - lf[1])
        for cf in cf_grid:
            dcf = np.hypot(bx - cf[0], by - cf[1])
            for rf in rf_grid:
                drf = np.hypot(bx - rf[0], by - rf[1])
                # assign each ball to closest fielder; penalty = negative distance
                dist_min = np.minimum(np.minimum(dlf, dcf), drf)
                total_penalty = -dist_min.sum()
                if total_penalty < best_score:
                    best_score = total_penalty
                    best = {"LF": lf, "CF": cf, "RF": rf}
    return best

# -------------------------------------------------------
# Plotting (stadium-style overlay)
# -------------------------------------------------------
def make_stadium_plot(df: pd.DataFrame, positions: Dict[str, Tuple[float,float]],
                      batter_label: str, pitcher_hand: str) -> str:
    fig, ax = plt.subplots(figsize=(9,5))

    # background stadium image if available
    if os.path.exists(STADIUM_IMAGE):
        img = mpimg.imread(STADIUM_IMAGE)
        ax.imshow(img, extent=[0,300,0,420], zorder=0)
    else:
        ax.set_facecolor("darkgreen")

    ax.scatter(df["x"], df["y"], c="orange", s=25, alpha=0.6,
               label="Batted balls", zorder=2)

    for pos_name, (x,y) in positions.items():
        rect = Rectangle((x-5, y-5), 10, 10,
                         linewidth=2, edgecolor="red",
                         facecolor="none", zorder=3)
        ax.add_patch(rect)
        ax.text(x+6, y+6, pos_name, color="red", fontsize=10,
                weight="bold", zorder=4)

    ax.set_xlim(40,260)
    ax.set_ylim(200,420)
    ax.axis("off")
    ax.set_title(f"{batter_label} vs. {pitcher_hand}", color="white")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# -------------------------------------------------------
# Routes
# -------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(HTML, batters=BATTERS)

@app.route("/api/compute", methods=["POST"])
def api_compute():
    try:
        payload = request.get_json(force=True)
        batter_id = payload.get("batter_id")
        pitcher_hand = payload.get("pitcher_hand", "RHP")

        if batter_id not in BATTERS:
            return jsonify({"ok": False, "error": "Unknown batter"}), 400

        meta = BATTERS[batter_id]
        df = load_batted_balls(batter_id, pitcher_hand)
        positions = optimize_outfield(df)

        # save printable CSV
        pd.DataFrame.from_dict(positions, orient="index", columns=["X","Y"]).to_csv(LAST_CSV_PATH)

        img_b64 = make_stadium_plot(df, positions, meta["label"], pitcher_hand)

        return jsonify({
            "ok": True,
            "batter_id": batter_id,
            "batter_label": meta["label"],
            "batter_hand": meta["batter_hand"],
            "pitcher_hand": pitcher_hand,
            "positions": positions,
            "image_base64": img_b64,
            "download_url": "/download"
        })
    except Exception as e:
        log.exception("api_compute failed")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/download")
def download():
    if not os.path.exists(LAST_CSV_PATH):
        return "Run an optimization first.", 404
    return send_file(LAST_CSV_PATH, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
