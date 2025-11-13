import io, base64, sys, logging
from typing import Dict, Tuple
from flask import Flask, request, jsonify, render_template_string, send_file
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

# -------------------------------------------------------
# CONFIG: batter options (batter name + side)
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
    # add more if you want:
    # "turner_L": {...}, "turner_R": {...}
}

LAST_CSV_PATH = "optimized_positions.csv"

# -------------------------------------------------------
# SIMPLE UI
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
# SPRAY DATA (synthetic but different for each scenario)
# -------------------------------------------------------
def generate_spray(batter_id: str, pitcher_hand: str) -> pd.DataFrame:
    """
    Generate a synthetic spray chart based on batter + pitcher handedness.
    This guarantees we always have data and the hitters look different.
    """
    seed = abs(hash(batter_id + "_" + pitcher_hand)) % (2**32)
    rng = np.random.default_rng(seed)
    n = 160

    # Slight directional bias based on batter handedness and pitcher
    meta = BATTERS[batter_id]
    bhand = meta["batter_hand"]

    if bhand == "L" and pitcher_hand == "RHP":
        # pull-side to right field
        x = rng.normal(210, 25, n)
    elif bhand == "L" and pitcher_hand == "LHP":
        # more opposite-field
        x = rng.normal(140, 30, n)
    elif bhand == "R" and pitcher_hand == "LHP":
        x = rng.normal(90, 25, n)
    else:  # R vs RHP
        x = rng.normal(160, 30, n)

    y = rng.normal(300, 35, n)
    x = np.clip(x, 50, 250)
    y = np.clip(y, 220, 400)
    return pd.DataFrame({"x": x, "y": y})

# -------------------------------------------------------
# OPTIMIZATION (3-layer loop)
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

    best_score = float("inf")   # we want the *smallest* total distance
    best = {}

    for lf in lf_grid:
        dlf = np.hypot(bx - lf[0], by - lf[1])
        for cf in cf_grid:
            dcf = np.hypot(bx - cf[0], by - cf[1])
            for rf in rf_grid:
                drf = np.hypot(bx - rf[0], by - rf[1])

                # distance to closest fielder for each ball
                dist_min = np.minimum(np.minimum(dlf, dcf), drf)

                # objective: *total* distance (smaller is better)
                total_distance = dist_min.sum()

                if total_distance < best_score:
                    best_score = total_distance
                    best = {"LF": lf, "CF": cf, "RF": rf}

    return best

# -------------------------------------------------------
# PLOTTING (make sure players show!)
# -------------------------------------------------------
def make_plot(df: pd.DataFrame, positions: Dict[str, Tuple[float,float]],
              batter_label: str, pitcher_hand: str) -> str:
    fig, ax = plt.subplots(figsize=(9,5))

    # simple "stadium": green field + faint fence
    ax.set_facecolor("#084408")
    ax.axhline(210, color="#dddddd", linewidth=3)  # infield/fence line

    # spray
    ax.scatter(df["x"], df["y"], c="#ff9933", s=35, alpha=0.7,
               edgecolor="none", label="Batted balls", zorder=2)

    # outfielders as red rectangles
    for pos_name, (x,y) in positions.items():
        rect = Rectangle((x-6, y-6), 12, 12,
                         linewidth=2, edgecolor="red",
                         facecolor="none", zorder=4)
        ax.add_patch(rect)
        ax.scatter(x, y, c="red", s=80, zorder=5)
        ax.text(x+8, y+8, pos_name, color="red",
                fontsize=10, weight="bold", zorder=6)

    ax.set_xlim(40,260)
    ax.set_ylim(200,420)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{batter_label} vs. {pitcher_hand}",
                 color="white", fontsize=14)

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

        # Save printable CSV
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
    if not os.path.exists(LAST_CSV_PATH):
        return "Run an optimization first.", 404
    return send_file(LAST_CSV_PATH, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
