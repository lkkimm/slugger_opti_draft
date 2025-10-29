import io, os
from datetime import date, timedelta
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adapter import fetch_batted_balls, fetch_players
from mapper import to_dataframe

app = Flask(__name__)

# --- physics & geometry ---
MPH_TO_MS = 0.44704
G = 9.81
PLAYER_SPEED_MS = 7.0
FIELD_RADIUS_M = 140.0     # ~460 ft
GRID_STEP_M = 5.0          # coarse grid

def est_hangtime(ev_mph, la_deg):
    v = np.maximum(ev_mph, 0.0) * MPH_TO_MS
    theta = np.radians(np.maximum(la_deg, 0.0))
    return np.maximum(2.0 * v * np.sin(theta) / G, 0.0)

def range_no_drag(ev_mph, la_deg):
    v = np.maximum(ev_mph, 0.0) * MPH_TO_MS
    theta = np.radians(np.maximum(la_deg, 0.0))
    R = (v**2) * np.sin(2.0 * theta) / G
    return np.clip(R, 0.0, FIELD_RADIUS_M)

def xy_from_range(R, spray_deg):
    phi = np.radians(spray_deg)
    x = R * np.sin(phi)   # +x RF, -x LF
    y = R * np.cos(phi)   # +y CF
    return x, y

def compute_xy(df):
    if "hangtime_s" not in df or df["hangtime_s"].isna().all():
        df["hangtime_s"] = est_hangtime(df["ev_mph"], df["la_deg"])
    R = range_no_drag(df["ev_mph"], df["la_deg"])
    df["x_m"], df["y_m"] = xy_from_range(R, df["spray_deg"])
    return df

def catch_flag(xs, ys, hts, fx, fy, speed=PLAYER_SPEED_MS):
    dist = np.sqrt((xs - fx)**2 + (ys - fy)**2)
    return (dist / speed <= hts).astype(int)

def optimize_three_fielders(df):
    # zone split by spray angle
    lf = df[df["spray_deg"] < -10]
    cf = df[(df["spray_deg"] >= -10) & (df["spray_deg"] <= 10)]
    rf = df[df["spray_deg"] > 10]

    grid = np.arange(0, FIELD_RADIUS_M + 1e-6, GRID_STEP_M)

    def best_for(sub):
        if sub.empty:
            return {"pos": (float("nan"), float("nan")), "caught": 0, "total": 0}
        xs, ys, hts = sub["x_m"].values, sub["y_m"].values, sub["hangtime_s"].values
        best_caught, best_xy = -1, (0.0, 0.0)
        for gx in grid:
            for gy in grid:
                if gy < 0:
                    continue
                caught = catch_flag(xs, ys, hts, gx, gy).sum()
                if caught > best_caught:
                    best_caught, best_xy = caught, (gx, gy)
        return {"pos": best_xy, "caught": int(best_caught), "total": int(len(sub))}

    out = {"LF": best_for(lf), "CF": best_for(cf), "RF": best_for(rf)}
    caught_total = out["LF"]["caught"] + out["CF"]["caught"] + out["RF"]["caught"]
    balls_total = int(len(df))
    out["summary"] = {
        "caught_total": caught_total,
        "balls_total": balls_total,
        "catch_rate": (caught_total / balls_total) if balls_total else 0.0
    }
    return out

def plot_spray(df, best):
    fig, ax = plt.subplots(figsize=(6,6))
    # outfield arc
    circle = plt.Circle((0,0), FIELD_RADIUS_M, fill=False, linewidth=1)
    ax.add_artist(circle)

    for hand, sub in df.groupby("handedness"):
        ax.scatter(sub["x_m"], sub["y_m"], s=12, alpha=0.7, label=f"{hand} ({len(sub)})")

    for zone in ("LF","CF","RF"):
        fx, fy = best[zone]["pos"]
        if not (np.isnan(fx) or np.isnan(fy)):
            ax.scatter([fx], [fy], marker="s", s=120, label=f"{zone} fielder")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-FIELD_RADIUS_M, FIELD_RADIUS_M)
    ax.set_ylim(0, FIELD_RADIUS_M)
    ax.set_title("Optimal Fielder Position")
    ax.set_xlabel("x (m)  ← LF        RF →")
    ax.set_ylabel("y (m) toward CF")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig

# ---------- Routes ----------
@app.route("/")
def index():
    today = date.today()
    start = (today - timedelta(days=180)).isoformat()
    return render_template("index.html", default_start=start, default_end=today.isoformat())

@app.route("/api/players")
def api_players():
    hand  = request.args.get("hand", "")
    start = request.args.get("start", "")
    end   = request.args.get("end", "")
    items = fetch_players(start_date=start or None, end_date=end or None, handedness=hand or None)
    return jsonify({"players": items})

@app.route("/api/compute")
def api_compute():
    players = request.args.get("players", "")  # comma-separated ids
    hand    = request.args.get("hand", "")
    start   = request.args.get("start", "")
    end     = request.args.get("end", "")

    player_ids = [p for p in players.split(",") if p] or None

    items = fetch_batted_balls(player_ids=player_ids, handedness=hand or None,
                               start_date=start or None, end_date=end or None, limit=5000)
    df = to_dataframe(items)
    if df.empty:
        return jsonify({
            "summary":{"caught_total":0,"balls_total":0,"catch_rate":0.0},
            "LF":{}, "CF":{}, "RF":{},
            "points": [],
            "selected_players": player_ids or [],
            "hand": hand
        })

    df = compute_xy(df)
    best = optimize_three_fielders(df)

    def pos(o):
        x,y = o["pos"]
        return {"x_m": float(x), "y_m": float(y), "caught": o["caught"], "total": o["total"]}

    payload = {
        "summary": best["summary"],
        "LF": pos(best["LF"]),
        "CF": pos(best["CF"]),
        "RF": pos(best["RF"]),
        "points": df[["x_m","y_m","handedness"]].to_dict(orient="records"),
        "selected_players": player_ids or [],
        "hand": hand
    }
    return jsonify(payload)

@app.route("/chart")
def chart_png():
    players = request.args.get("players", "")
    hand    = request.args.get("hand", "")
    start   = request.args.get("start", "")
    end     = request.args.get("end", "")
    player_ids = [p for p in players.split(",") if p] or None

    items = fetch_batted_balls(player_ids=player_ids, handedness=hand or None,
                               start_date=start or None, end_date=end or None, limit=5000)
    df = to_dataframe(items)
    if df.empty:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.text(0.5,0.5,"No data", ha="center", va="center")
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150); buf.seek(0); plt.close(fig)
        return send_file(buf, mimetype="image/png")

    df = compute_xy(df)
    best = optimize_three_fielders(df)
    fig = plot_spray(df, best)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
