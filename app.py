# app.py — demo-ready backend for your existing Spray Chart Optimizer UI

import os, io, base64, glob, sys, logging
from typing import Dict, List, Tuple, Any
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
# DEMO CONFIG (hardcode who appears in the Players list)
# Change this to match files you’ve committed.
# Clean filenames expected: <Player>_L.csv / <Player>_R.csv (or .xlsm)
# -----------------------------
DEMO_MODE = True
DEMO_PLAYERS = {
    "Dickerson": ["L", "R"],
    # Uncomment if you actually have the file(s):
    # "Turner": ["L"]
}

LAST_CSV_PATH = "optimized_positions.csv"

# -----------------------------
# Data discovery / loading
# -----------------------------
def discover_players() -> Dict[str, List[str]]:
    files = glob.glob("*_[LR].csv") + glob.glob("*_[LR].xlsm")
    found: Dict[str, List[str]] = {}
    for f in files:
        base = os.path.basename(f)
        try:
            name, rest = base.split("_", 1)
            hand = rest[0].upper()  # L or R
            if hand in ("L", "R"):
                found.setdefault(name, [])
                if hand not in found[name]:
                    found[name].append(hand)
        except Exception:
            continue
    return found

def load_sample(player: str, hand: str) -> pd.DataFrame:
    """
    Reads <Player>_<Hand>.csv or .xlsm and returns columns ['x','y'] clipped to field bounds.
    """
    candidates = [f"{player}_{hand}.csv", f"{player}_{hand}.xlsm"]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"Missing sample file for {player} {hand}: tried {', '.join(candidates)}")

    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, engine="openpyxl")

    # Normalize to x,y
    lower = {c.lower().strip(): c for c in df.columns}
    for xa, ya in [("x", "y"), ("hc_x", "hc_y"), ("spray_x", "spray_y"), ("px", "py")]:
        if xa in lower and ya in lower:
            df = df[[lower[xa], lower[ya]]].rename(columns={lower[xa]: "x", lower[ya]: "y"})
            break
    else:
        num = df.select_dtypes(include="number").columns.tolist()
        if len(num) < 2:
            raise ValueError(f"{path} has no usable numeric x/y columns")
        df = df[[num[0], num[1]]].rename(columns={num[0]: "x", num[1]: "y"})

    df = df[(df["x"].between(0, 300)) & (df["y"].between(0, 420))].dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{path} produced zero rows after cleaning")
    return df

# -----------------------------
# Optimization + plotting
# -----------------------------
def optimize_positions(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    # Coarse grid (fast on small dynos). Increase fidelity by reducing step.
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

def average_positions(pL: Dict[str, Tuple[float, float]], pR: Dict[str, Tuple[float, float]]):
    return {k: ((pL[k][0] + pR[k][0]) / 2.0, (pL[k][1] + pR[k][1]) / 2.0) for k in pL}

def make_plot_png_b64(df: pd.DataFrame, positions: Dict[str, Tuple[float, float]]) -> str:
    fig, ax = plt.subplots(figsize=(6.75, 6.75))
    ax.scatter(df["x"], df["y"], alpha=0.45, s=15, label="Batted balls")
    for name, (x, y) in positions.items():
        ax.scatter(x, y, s=120, label=name)
        ax.text(x + 4, y + 4, name, color="tab:blue", fontsize=9)
    ax.set_xlim(0, 300); ax.set_ylim(0, 420)
    ax.set_xlabel("Horizontal (ft)"); ax.set_ylabel("Depth (ft)")
    ax.set_title("Optimized Outfield Placement")
    ax.legend(loc="upper right")
    buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# -----------------------------
# Routes (match your existing UI)
# -----------------------------
@app.route("/")
def home():
    # serve your current frontend (templates/index.html)
    try:
        return render_template("index.html")
    except Exception:
        return "<h2>Backend is running. Add templates/index.html for the UI.</h2>"

@app.route("/api/players")
def api_players():
    """
    Return exactly what your left panel expects: a plain array of names.
    """
    if DEMO_MODE:
        names = sorted(DEMO_PLAYERS.keys())
    else:
        names = sorted(discover_players().keys())
    return jsonify(names)

def _coerce_players(payload: Any) -> List[str]:
    if isinstance(payload, dict) and "players" in payload:
        payload = payload["players"]
    if isinstance(payload, list):
        return [str(x).strip() for x in payload if str(x).strip()]
    if isinstance(payload, str):
        chunks = [t.strip() for part in payload.split("\n") for t in part.split(",")]
        return [c for c in chunks if c]
    return []

@app.route("/api/compute", methods=["POST"])
def api_compute():
    """
    Accepts:
      {"players":["Dickerson"], "handedness":"L|R|B|any"}
      {"players":"Dickerson", "handedness":"L"}
      or "Dickerson"
    Returns positions + base64 image + /download link.
    """
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            data = {"players": request.form.get("players", ""),
                    "handedness": request.form.get("handedness", "any")}

        players = _coerce_players(data)
        if not players:
            return jsonify({"ok": False, "error": "No players provided"}), 400

        player = players[0]
        hand = str((data.get("handedness") if isinstance(data, dict) else "any") or "any").upper()

        if DEMO_MODE:
            available = DEMO_PLAYERS.get(player, [])
        else:
            available = discover_players().get(player, [])

        if not available:
            return jsonify({"ok": False, "error": f"No sample files found for {player}."}), 400

        # Resolve ANY → B if both exist; else the one available
        if hand in ("ANY", "(ANY)"):
            hand = "B" if set(available) == {"L", "R"} else available[0]

        if hand == "B":
            if not {"L", "R"}.issubset(set(available)):
                return jsonify({"ok": False, "error": f"{player} only has {available}. Choose L or R."}), 400
            dfL = load_sample(player, "L")
            dfR = load_sample(player, "R")
            posL = optimize_positions(dfL)
            posR = optimize_positions(dfR)
            positions = average_positions(posL, posR)
            df = pd.concat([dfL, dfR], ignore_index=True)
        else:
            if hand not in available:
                hand = available[0]  # graceful fallback
            df = load_sample(player, hand)
            positions = optimize_positions(df)

        # Save printable CSV and make plot
        pd.DataFrame.from_dict(positions, orient="index", columns=["X", "Y"]).to_csv(LAST_CSV_PATH)
        img_b64 = make_plot_png_b64(df, positions)

        return jsonify({
            "ok": True,
            "player": player,
            "handedness": hand,
            "positions": positions,      # {"LF":[x,y], "CF":[x,y], "RF":[x,y]}
            "image_base64": img_b64,     # <img src="data:image/png;base64, ...">
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
    app.run(host="0.0.0.0", port=8080)
