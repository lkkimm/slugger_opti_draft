# app.py
import os, io, base64, glob, sys, logging
from typing import Dict, List, Tuple
from flask import Flask, jsonify, request, send_file, render_template
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")

# -----------------------------
# Helpers: data discovery/load
# -----------------------------
def discover_players() -> Dict[str, List[str]]:
    """
    Scan repo for files named <Player>_L.(csv|xlsm) or <Player>_R.(csv|xlsm).
    Returns: { "Dickerson": ["L","R"], "Turner": ["L"], ... }
    """
    files = glob.glob("*_[LR].csv") + glob.glob("*_[LR].xlsm")
    found = {}
    for f in files:
        base = os.path.basename(f)
        try:
            name, hand_with_ext = base.split("_", 1)
            hand = hand_with_ext[0].upper()  # 'L' or 'R'
            if hand in ("L", "R"):
                found.setdefault(name, [])
                if hand not in found[name]:
                    found[name].append(hand)
        except Exception:
            continue
    return found

def load_sample(player: str, hand: str) -> pd.DataFrame:
    """
    Load CSV/XLSM for a player/hand. Normalizes to columns ['x','y'].
    """
    candidates = [f"{player}_{hand}.csv", f"{player}_{hand}.xlsm"]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"Missing sample file for {player} {hand}: tried {', '.join(candidates)}")

    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, engine="openpyxl")

    # Normalize columns
    lower = {c.lower().strip(): c for c in df.columns}
    # common aliases
    for x_alias, y_alias in [("x","y"), ("hc_x","hc_y"), ("spray_x","spray_y"), ("px","py")]:
        if x_alias in lower and y_alias in lower:
            df = df[[lower[x_alias], lower[y_alias]]].rename(columns={lower[x_alias]:"x", lower[y_alias]:"y"})
            break
    else:
        # fallback: first two numeric columns
        num = df.select_dtypes(include="number").columns.tolist()
        if len(num) < 2:
            raise ValueError(f"{path} does not contain usable numeric x/y columns")
        df = df[[num[0], num[1]]].rename(columns={num[0]:"x", num[1]:"y"})

    # bounds clamp (just in case)
    df = df[(df["x"].between(0, 300)) & (df["y"].between(0, 420))].dropna()
    if df.empty:
        raise ValueError(f"{path} produced zero rows after cleaning")
    return df.reset_index(drop=True)

# -----------------------------
# Core: optimization + plotting
# -----------------------------
def optimize_positions(df: pd.DataFrame) -> Dict[str, Tuple[float,float]]:
    # coarser grids for speed on small dynos; tweak step if you want finer
    lf_grid = [(x, y) for x in range(60, 120, 15) for y in range(250, 350, 15)]
    cf_grid = [(x, y) for x in range(120, 180, 15) for y in range(300, 400, 15)]
    rf_grid = [(x, y) for x in range(180, 240, 15) for y in range(250, 350, 15)]

    best_score = float("inf")
    best = {}

    bx = df["x"].to_numpy()
    by = df["y"].to_numpy()

    for lf in lf_grid:
        dlf = np.hypot(bx - lf[0], by - lf[1])
        for cf in cf_grid:
            dcf = np.hypot(bx - cf[0], by - cf[1])
            for rf in rf_grid:
                drf = np.hypot(bx - rf[0], by - rf[1])
                # penalty is negative distance to closest fielder
                total_penalty = -(np.minimum(np.minimum(dlf, dcf), drf)).sum()
                if total_penalty < best_score:
                    best_score = total_penalty
                    best = {"LF": lf, "CF": cf, "RF": rf}
    return best

def average_positions(pL: Dict[str, Tuple[float,float]], pR: Dict[str, Tuple[float,float]]):
    return {k: ((pL[k][0]+pR[k][0])/2.0, (pL[k][1]+pR[k][1])/2.0) for k in pL}

def make_plot_png_b64(df: pd.DataFrame, positions: Dict[str, Tuple[float,float]]) -> str:
    fig, ax = plt.subplots(figsize=(6.5,6.5))
    ax.scatter(df["x"], df["y"], alpha=0.45, s=15, label="Batted balls")
    for name, (x, y) in positions.items():
        ax.scatter(x, y, s=120, label=name)
        ax.text(x+4, y+4, name, color="tab:blue", fontsize=9)
    ax.set_xlim(0, 300); ax.set_ylim(0, 420)
    ax.set_xlabel("Horizontal (ft)"); ax.set_ylabel("Depth (ft)")
    ax.set_title("Optimized Outfield Placement")
    ax.legend(loc="upper right")
    buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

LAST_CSV_PATH = "optimized_positions.csv"

# -----------------------------
# Routes for your UI
# -----------------------------
@app.route("/")
def home():
    # serve your existing frontend (keep your index.html if you have it)
    try:
        return render_template("index.html")
    except Exception:
        # fallback simple note if template missing
        return "<h2>Backend up. Frontend expects /templates/index.html</h2>"

@app.route("/api/players")
def api_players():
    """
    Returns players your UI can show in the left list.
    Shape your JS is expecting: { players: [{name:'Dickerson', hands:['L','R']}, ...] }
    """
    players = discover_players()
    payload = {"players": [{"name": p, "hands": sorted(players[p])} for p in sorted(players.keys())]}
    return jsonify(payload)

@app.route("/api/compute", methods=["POST"])
def api_compute():
    """
    Body example your UI can send:
    {
      "players": ["Dickerson"],     // first batter, or multiple later
      "handedness": "L" | "R" | "B",
      "start": "2021-01-01",        // ignored for sample mode
      "end": "2021-12-31"           // ignored for sample mode
    }
    """
    try:
        req = request.get_json(force=True) or {}
        players = req.get("players") or []
        hand = (req.get("handedness") or "L").upper()

        if not players:
            return jsonify({"error": "No players selected"}), 400

        player = players[0]  # first batter, as per your workflow

        if hand == "B":
            dfL = load_sample(player, "L")
            dfR = load_sample(player, "R")
            posL = optimize_positions(dfL)
            posR = optimize_positions(dfR)
            positions = average_positions(posL, posR)
            df = pd.concat([dfL, dfR], ignore_index=True)
        else:
            df = load_sample(player, hand)
            positions = optimize_positions(df)

        # Save printable CSV
        pd.DataFrame.from_dict(positions, orient="index", columns=["X","Y"]).to_csv(LAST_CSV_PATH)

        # Plot
        png_b64 = make_plot_png_b64(df, positions)

        return jsonify({
            "ok": True,
            "player": player,
            "handedness": hand,
            "positions": positions,                 # {"LF":[x,y],...}
            "plot_png_base64": png_b64,             # frontend can <img src="data:image/png;base64, ...">
            "download_url": "/download"
        })
    except Exception as e:
        log.exception("api_compute failed")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/download")
def download():
    if not os.path.exists(LAST_CSV_PATH):
        return "Run a computation first.", 404
    return send_file(LAST_CSV_PATH, as_attachment=True)

if __name__ == "__main__":
    # local test
    app.run(host="0.0.0.0", port=8080, debug=True)
