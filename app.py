# app.py  —  Outfield Positioning Optimizer Demo (drawn field)

import io
import base64
import sys
import logging
from typing import Dict, Tuple

from flask import Flask, request, jsonify, render_template_string, send_file
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

# -------------------------------------------------------
# CONFIG: batter options (name + side)
# -------------------------------------------------------
BATTERS: Dict[str, Dict] = {
    "dickerson_L": {
        "label": "Corey Dickerson (L)",
        "batter_name": "Corey Dickerson",
        "batter_hand": "L",
    },
    "dickerson_R": {
        "label": "Corey Dickerson (R)",
        "batter_name": "Corey Dickerson",
        "batter_hand": "R",
    },
    # You can add more entries later, e.g.:
    # "turner_L": {...}, "turner_R": {...}
}

LAST_CSV_PATH = "optimized_positions.csv"

# -------------------------------------------------------
# SIMPLE UI (inline HTML)
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
    select, button { width:100%; padding:0.45rem; margin-top:0.25rem;
                     border-radius:6px; border:1px solid #444; }
    button { background:#0077ff; color:#fff; cursor:pointer; font-weight:600; border:none; }
    button:hover { background:#005fd6; }
    #screen { flex:1; background:#000; border-radius:12px; padding:0.5rem;
              display:flex; flex-direction:column; justify-content:center; align-items:center; }
    img.tv { max-width:100%; border-radius:12px; }
    .caption { margin-top:0.5rem; font-size:1.1rem; text-align:center; }
    .caption span.name { color:#4aa3ff; font-weight:700; }
    .coords { margin-top:0.5rem; font-size:0.9rem; text-align:center; color:#ddd; }
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
        Demo uses synthetic spray charts per batter/handedness matchup.
      </p>
      <p style="margin-top:0.25rem;font-size:0.8rem;color:#bbb;">
        <a href="/download" style="color:#4aa3ff;">Download latest LF/CF/RF CSV</a>
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
# SPRAY GENERATION (synthetic, by batter & pitcher hand)
# -------------------------------------------------------
def generate_spray(batter_id: str, pitcher_hand: str) -> pd.DataFrame:
    """
    Synthetic spray pattern that depends on batter side + pitcher hand.
    This keeps the demo self-contained and consistent.
    """
    meta = BATTERS[batter_id]
    bhand = meta["batter_hand"]

    seed = abs(hash(batter_id + "_" + pitcher_hand)) % (2**32)
    rng = np.random.default_rng(seed)
    n = 180

    # directional bias
    if bhand == "L" and pitcher_hand == "RHP":
        x = rng.normal(210, 25, n)  # pull to RF
    elif bhand == "L" and pitcher_hand == "LHP":
        x = rng.normal(150, 25, n)  # more middle
    elif bhand == "R" and pitcher_hand == "LHP":
        x = rng.normal(90, 25, n)   # pull to LF
    else:  # R vs RHP
        x = rng.normal(150, 25, n)

    y = rng.normal(310, 35, n)

    x = np.clip(x, 50, 250)
    y = np.clip(y, 230, 400)

    return pd.DataFrame({"x": x, "y": y})

# -------------------------------------------------------
# OPTIMIZATION (3-layer LF/CF/RF search)
# -------------------------------------------------------
def optimize_outfield(df: pd.DataFrame) -> Dict[str, Tuple[float,float]]:
    """
    3-layer brute-force over LF, CF, RF.
    Minimizes total distance from batted balls to nearest fielder.
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
                dist_min = np.minimum(np.minimum(dlf, dcf), drf)
                total_distance = dist_min.sum()
                if total_distance < best_score:
                    best_score = total_distance
                    best = {"LF": lf, "CF": cf, "RF": rf}
    return best

# -------------------------------------------------------
# PLOTTING (drawn baseball field + color-coded spray)
# -------------------------------------------------------
from matplotlib.patches import Polygon, Rectangle, Circle, Arc

def make_plot(df: pd.DataFrame,
              positions: Dict[str, Tuple[float, float]],
              batter_label: str,
              pitcher_hand: str) -> str:
    """
    Realistic drawn baseball field:
      - curved outfield fence
      - infield dirt arc
      - basepaths
      - mound / home plate
      - green grass
      - spray dots color-coded by outcome
      - optimized LF/CF/RF positions with small red boxes
    """

    # ---------- outcome → color ----------
    outcome_col = None
    for c in df.columns:
        if c.lower() in ("result", "outcome", "event"):
            outcome_col = c
            break

    if outcome_col is None:
        rng = np.random.default_rng(0)
        labels = np.array(["1B", "2B", "3B", "OUT"])
        df["outcome"] = rng.choice(labels, size=len(df),
                                   p=[0.55, 0.25, 0.03, 0.17])
        outcome_col = "outcome"

    color_map = {
        "1B": "#42a5f5",
        "2B": "#66bb6a",
        "3B": "#ffa726",
        "OUT": "#bdbdbd"
    }
    spray_colors = df[outcome_col].map(lambda v: color_map.get(str(v).upper(), "white"))

    # ---------- figure settings ----------
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#144d14")   # deep green grass

    # MLB-ish dimensions in your coordinate system
    # scale everything into the existing df coordinate system
    home = (150, 200)
    left_line = (60, 250)
    right_line = (240, 250)

    # Outfield fence arc
    fence_radius = 180
    fence_arc = Arc(home, width=fence_radius*2, height=fence_radius*2,
                    theta1=22, theta2=158, edgecolor="white",
                    linewidth=2, zorder=1)

    ax.add_patch(fence_arc)

    # Outfield grass fan polygon
    fence_points = []
    for angle in np.linspace(22, 158, 30):
        rad = np.radians(angle)
        px = home[0] + fence_radius * np.cos(rad)
        py = home[1] + fence_radius * np.sin(rad)
        fence_points.append((px, py))

    outfield_poly = Polygon(
        [left_line] + fence_points + [right_line],
        closed=True,
        facecolor="#1c6b1c",
        edgecolor="none",
        zorder=0
    )
    ax.add_patch(outfield_poly)

    # Infield dirt arc
    dirt_radius = 95
    dirt_arc = Arc(home, width=dirt_radius*2, height=dirt_radius*2,
                   theta1=22, theta2=158,
                   edgecolor="#c49a6c", linewidth=25, zorder=2)
    ax.add_patch(dirt_arc)

    # Basepaths (square rotated)
    basepath = Polygon(
        [
            (150, 200),       # home
            (170, 220),       # 1B-ish
            (150, 240),       # 2B-ish
            (130, 220)        # 3B-ish
        ],
        closed=True,
        facecolor="#c49a6c",
        edgecolor="white",
        linewidth=2,
        zorder=3
    )
    ax.add_patch(basepath)

    # Foul lines
    ax.plot([home[0], left_line[0]], [home[1], left_line[1]],
            color="white", linewidth=2, zorder=4)
    ax.plot([home[0], right_line[0]], [home[1], right_line[1]],
            color="white", linewidth=2, zorder=4)

    # Centerline (optional)
    ax.plot([150, 150], [250, 380],
            color="white", linestyle="--", linewidth=1.2, alpha=0.6, zorder=4)

    # ---------- spray dots ----------
    ax.scatter(df["x"], df["y"],
               c=spray_colors, s=30, alpha=0.8,
               edgecolor="none", zorder=5)

    # ---------- LF / CF / RF red boxes ----------
    box_w, box_h = 12, 12

    for name, (cx, cy) in positions.items():
        rect = Rectangle((cx - box_w/2, cy - box_h/2),
                         box_w, box_h,
                         linewidth=2, edgecolor="red",
                         facecolor="none", zorder=7)
        ax.add_patch(rect)

        ax.scatter(cx, cy, c="red", s=70, zorder=8)

        ax.text(cx, cy + box_h + 3,
                name, color="red", fontsize=10,
                ha="center", va="bottom",
                weight="bold", zorder=9)

    # ---------- cosmetics ----------
    ax.set_xlim(40, 260)
    ax.set_ylim(200, 420)
    ax.set_xticks([]); ax.set_yticks([])
    ax.axis("off")

    ax.set_title(f"{batter_label} vs {pitcher_hand}",
                 color="white", fontsize=16, pad=12)

    # ---------- export ----------
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# -------------------------------------------------------
# ROUTES
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

        df = generate_spray(batter_id, pitcher_hand)
        positions = optimize_outfield(df)

        # save printable CSV
        pd.DataFrame.from_dict(positions, orient="index", columns=["X","Y"]).to_csv(LAST_CSV_PATH)

        img_b64 = make_plot(df, positions, meta["label"], pitcher_hand)

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
    if not pd.io.common.file_exists(LAST_CSV_PATH):
        return "Run an optimization first.", 404
    return send_file(LAST_CSV_PATH, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
