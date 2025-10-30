import os, io, base64, logging, sys, glob
from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Logging to stdout (so Railway "Logs" shows stacktraces)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")

# -----------------------
# Helpers
# -----------------------
def find_available_players():
    """
    Auto-detect players from files present in the repo.
    Expected patterns: <Player>_L.csv, <Player>_R.csv, <Player>_L.xlsm, <Player>_R.xlsm
    """
    hits = glob.glob("*_[LR].csv") + glob.glob("*_[LR].xlsm")
    players = sorted(set(os.path.basename(f).split("_")[0] for f in hits))
    return players

def load_sample(player: str, hand: str) -> pd.DataFrame:
    """
    Load sample data for a player & handedness.
    Looks for both CSV and XLSM.
    Normalizes to columns ['x','y'].
    """
    candidates = [
        f"{player}_{hand}.csv",
        f"{player}_{hand}.xlsm"
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"No sample file for {player} {hand}. "
                                f"Tried: {', '.join(candidates)}")

    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        # needs openpyxl in requirements
        df = pd.read_excel(path, engine="openpyxl")

    # Robust column normalization
    rename_map = {c.lower().strip(): c for c in df.columns}
    cols_lower = [c.lower().strip() for c in df.columns]

    # Prefer common names first
    candidates_xy = [
        ("x","y"),
        ("hc_x","hc_y"),
        ("coordx","coordy"),
        ("spray_x","spray_y"),
        ("px","py")
    ]
    xcol = ycol = None
    for cx, cy in candidates_xy:
        if cx in cols_lower and cy in cols_lower:
            xcol = rename_map[cx]
            ycol = rename_map[cy]
            break

    # Fallback: first two numeric columns
    if xcol is None or ycol is None:
        numeric = df.select_dtypes(include="number").columns.tolist()
        if len(numeric) >= 2:
            xcol, ycol = numeric[0], numeric[1]
        else:
            raise ValueError("Could not find x/y columns in data.")

    out = df[[xcol, ycol]].dropna().copy()
    out.columns = ["x", "y"]

    # Filter out obvious grounders / HRs if present (optional soft filters)
    # Keep this ultra-light; your CSVs may not have these flags.
    for ground_tag in ("type","batted_type","bb_type"):
        if ground_tag in [c.lower() for c in df.columns]:
            col = df.columns[[c.lower()==ground_tag for c in df.columns][0]]
            mask = ~df[col].astype(str).str.lower().isin(["groundball","grounder","hr","homerun"])
            out = out.loc[mask.values[:len(out)]]

    # Clamp field bounds to something reasonable
    out = out[(out["x"].between(0, 300)) & (out["y"].between(0, 420))]
    return out

def penalty_to_fielder(ball, pos):
    # simple distance penalty (closer is better)
    return -np.hypot(ball["x"] - pos[0], ball["y"] - pos[1])

def optimize_positions(df: pd.DataFrame):
    """
    Brute-force 3-layer loop (LF/CF/RF) over reasonable areas.
    Coarser grid for speed on Railway free dynos.
    """
    lf_grid = [(x, y) for x in range(60, 120, 15) for y in range(250, 350, 15)]
    cf_grid = [(x, y) for x in range(120, 180, 15) for y in range(300, 400, 15)]
    rf_grid = [(x, y) for x in range(180, 240, 15) for y in range(250, 350, 15)]

    best = None
    best_score = float("inf")

    # vectorize a bit: pre-store balls
    bx = df["x"].to_numpy()
    by = df["y"].to_numpy()

    for lf in lf_grid:
        for cf in cf_grid:
            for rf in rf_grid:
                # compute min distance per ball to any fielder
                d_lf = np.hypot(bx - lf[0], by - lf[1])
                d_cf = np.hypot(bx - cf[0], by - cf[1])
                d_rf = np.hypot(bx - rf[0], by - rf[1])
                total = -(np.minimum(np.minimum(d_lf, d_cf), d_rf)).sum()  # negative distances = penalties
                if total < best_score:
                    best_score = total
                    best = {"LF": lf, "CF": cf, "RF": rf}
    return best

def average_pos(p1: dict, p2: dict):
    return {k: ((p1[k][0]+p2[k][0])/2.0, (p1[k][1]+p2[k][1])/2.0) for k in p1.keys()}

def make_plot(df, positions):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(df["x"], df["y"], alpha=0.45, s=15, label="Batted balls")
    for name, (x, y) in positions.items():
        ax.scatter(x, y, s=120, label=name)
        ax.text(x+4, y+4, name, color="tab:blue", fontsize=9)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 420)
    ax.set_title("Optimized Outfield Placement")
    ax.set_xlabel("Horizontal (ft)")
    ax.set_ylabel("Depth (ft)")
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    players = find_available_players()  # from files you actually uploaded
    error = None
    result = None
    plot_data = None

    # sensible default if repo has only Dickerson_*.csv
    default_player = players[0] if players else "Dickerson"

    if request.method == "POST":
        player = request.form.get("player", default_player)
        hand   = request.form.get("hand", "L")  # L / R / B
        try:
            if hand == "B":
                # require both files to exist
                dfL = load_sample(player, "L")
                dfR = load_sample(player, "R")
                posL = optimize_positions(dfL)
                posR = optimize_positions(dfR)
                positions = average_pos(posL, posR)
                df = pd.concat([dfL, dfR], ignore_index=True)
            else:
                df = load_sample(player, hand)
                positions = optimize_positions(df)

            # save printable CSV
            pd.DataFrame.from_dict(positions, orient="index", columns=["X","Y"])\
              .to_csv("optimized_positions.csv")

            plot_data = make_plot(df, positions)
            result = positions

        except Exception as e:
            error = str(e)
            log.exception("Error during optimization")

    return render_template(
        "index.html",
        players=players,
        default_player=default_player,
        error=error,
        result=result,
        plot_data=plot_data
    )

@app.route("/download")
def download():
    if not os.path.exists("optimized_positions.csv"):
        return "No file yet. Run an optimization first.", 404
    return send_file("optimized_positions.csv", as_attachment=True)

if __name__ == "__main__":
    # local dev
    app.run(host="0.0.0.0", port=8080, debug=True)
