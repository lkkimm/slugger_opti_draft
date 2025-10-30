from flask import Flask, render_template_string, request, send_file
import pandas as pd
import numpy as np
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# ------------------------
# HTML FRONTEND
# ------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Outfield Optimizer Demo</title>
    <style>
        body { font-family: sans-serif; background: #f5f7fa; margin: 2rem; }
        h1 { color: #002b5c; }
        form { background: white; padding: 1rem 2rem; border-radius: 10px; margin-bottom: 2rem; }
        select, button { margin-top: 1rem; padding: 0.4rem; font-size: 1rem; }
        .result { background: white; padding: 1rem; border-radius: 10px; }
        img { margin-top: 1rem; border-radius: 10px; max-width: 700px; }
    </style>
</head>
<body>
    <h1>⚾ Outfield Optimizer Widget (Demo)</h1>
    <p>Select a batter and pitcher handedness to generate optimal outfield placement using sample Trackman data.</p>

    <form method="POST" enctype="multipart/form-data">
        <label><b>Batter:</b></label><br>
        <select name="player">
            <option value="Dickerson">Dickerson</option>
            <option value="Turner">Turner</option>
        </select><br><br>
        <label><b>Pitcher Handedness:</b></label><br>
        <select name="pitcher_hand">
            <option value="L">Left-handed</option>
            <option value="R">Right-handed</option>
            <option value="B">Both (average)</option>
        </select><br><br>
        <button type="submit">Compute Optimal Placement</button>
    </form>

    {% if positions %}
    <div class="result">
        <h3>Optimal Outfield Positions:</h3>
        <ul>
            {% for fielder, coords in positions.items() %}
                <li><b>{{ fielder }}</b>: X={{ coords[0] | round(1) }}, Y={{ coords[1] | round(1) }}</li>
            {% endfor %}
        </ul>
        <a href="/download">📄 Download Printable CSV</a><br>
        <img src="data:image/png;base64,{{ plot_data }}" alt="Spray Chart">
    </div>
    {% endif %}
</body>
</html>
"""

# ------------------------
# CORE FUNCTIONS
# ------------------------
def load_sample_data(player, hand):
    """Load CSV/XLSM based on player & pitcher hand."""
    if player == "Dickerson":
        if hand == "L":
            df = pd.read_csv("Dickerson_L.csv")
        else:
            df = pd.read_csv("Dickerson_R (1).csv")
    elif player == "Turner":
        df = pd.read_excel("Turner- L 2021-06-20-2 (2).xlsm", engine="openpyxl")
        if "x" not in df.columns:
            df.rename(columns={df.columns[0]: "x", df.columns[1]: "y"}, inplace=True)
    else:
        raise ValueError("Unsupported player.")
    df = df.rename(columns=lambda c: c.strip().lower())
    if not {"x", "y"}.issubset(df.columns):
        df = df.rename(columns={df.columns[0]: "x", df.columns[1]: "y"})
    return df[["x", "y"]].dropna()

def reward(ball, fielder_pos):
    """Negative distance = penalty (closer = better)."""
    x, y = ball["x"], ball["y"]
    fx, fy = fielder_pos
    dist = np.sqrt((x - fx)**2 + (y - fy)**2)
    return -dist

def optimize(df):
    """Simple brute-force optimization across LF, CF, RF."""
    lf_range = [(x, y) for x in range(60, 120, 15) for y in range(250, 350, 15)]
    cf_range = [(x, y) for x in range(120, 180, 15) for y in range(300, 400, 15)]
    rf_range = [(x, y) for x in range(180, 240, 15) for y in range(250, 350, 15)]

    best_score = float("inf")
    best_positions = {}

    for lf in lf_range:
        for cf in cf_range:
            for rf in rf_range:
                total_penalty = 0
                for _, b in df.iterrows():
                    total_penalty += min(
                        reward(b, lf), reward(b, cf), reward(b, rf)
                    )
                if total_penalty < best_score:
                    best_score = total_penalty
                    best_positions = {"LF": lf, "CF": cf, "RF": rf}
    return best_positions

def average_positions(pos1, pos2):
    """Average coordinates from two dicts."""
    return {f: ((pos1[f][0] + pos2[f][0]) / 2, (pos1[f][1] + pos2[f][1]) / 2) for f in pos1}

def create_plot(df, positions):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(df["x"], df["y"], c="gray", alpha=0.5, label="Batted Balls")
    for name, (x, y) in positions.items():
        ax.scatter(x, y, s=100, label=name)
        ax.text(x+4, y+4, name, color="blue")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 400)
    ax.set_title("Optimized Outfield Positions")
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ------------------------
# FLASK ROUTES
# ------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    positions, plot_data = None, None
    if request.method == "POST":
        player = request.form["player"]
        hand = request.form["pitcher_hand"]

        if hand == "B":
            df_L = load_sample_data(player, "L")
            df_R = load_sample_data(player, "R")
            pos_L = optimize(df_L)
            pos_R = optimize(df_R)
            positions = average_positions(pos_L, pos_R)
            df = pd.concat([df_L, df_R])
        else:
            df = load_sample_data(player, hand)
            positions = optimize(df)

        plot_data = create_plot(df, positions)
        pd.DataFrame.from_dict(positions, orient="index", columns=["X", "Y"]).to_csv("optimized_positions.csv")

    return render_template_string(HTML_TEMPLATE, positions=positions, plot_data=plot_data)

@app.route("/download")
def download():
    return send_file("optimized_positions.csv", as_attachment=True)

# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
