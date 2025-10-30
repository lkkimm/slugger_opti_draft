from flask import Flask, render_template_string, request, send_file
import pandas as pd
import numpy as np
import io
import base64
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# ------------------------------------------------
# 🔵 HTML UI (kept exactly like your previous version)
# ------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Outfield Optimizer</title>
    <style>
        body { font-family: sans-serif; margin: 2rem; background-color: #f4f4f4; }
        h1 { color: #003366; }
        form { margin-bottom: 1rem; }
        button { padding: 0.5rem 1rem; margin-top: 1rem; }
        .result { background: white; padding: 1rem; border-radius: 10px; }
        img { max-width: 600px; margin-top: 1rem; border-radius: 10px; }
    </style>
</head>
<body>
    <h1>⚾ Outfield Optimizer Widget</h1>
    <p>Choose a batter and pitcher handedness to calculate optimized outfield positions (sample data demo).</p>

    <form method="POST" enctype="multipart/form-data">
        <label>Batter:</label>
        <select name="player">
            <option value="Dickerson">Dickerson</option>
            <option value="Turner">Turner</option>
        </select>
        <br><br>
        <label>Pitcher Hand:</label>
        <select name="pitcher_hand">
            <option value="L">Left-handed</option>
            <option value="R">Right-handed</option>
            <option value="B">Both (average)</option>
        </select>
        <br>
        <button type="submit">Compute Optimal Positions</button>
    </form>

    {% if error %}
      <div style="color:red; font-weight:bold;">⚠️ {{ error }}</div>
    {% endif %}

    {% if positions %}
    <div class="result">
        <h3>Optimal Outfield Positions:</h3>
        <ul>
            {% for fielder, coords in positions.items() %}
                <li><b>{{ fielder }}</b>: X={{ coords[0]|round(1) }}, Y={{ coords[1]|round(1) }}</li>
            {% endfor %}
        </ul>
        <a href="/download">📄 Download Printable CSV</a><br>
        <img src="data:image/png;base64,{{ plot_data }}" alt="Spray Chart">
    </div>
    {% endif %}
</body>
</html>
"""

# ------------------------------------------------
# ⚙️ Data loading helpers
# ------------------------------------------------
def safe_load_csv_or_xlsm(player, hand):
    """
    Loads Dickerson/Turner sample data safely from your repo (CSV/XLSM).
    """
    try:
        if player == "Dickerson":
            fname = f"Dickerson_{hand}.csv"
        elif player == "Turner":
            # If you only have Turner_L for demo, fallback to L always
            fname = f"Turner_L.xlsm"
        else:
            raise ValueError("Unknown player.")

        if not os.path.exists(fname):
            raise FileNotFoundError(f"Missing file: {fname}")

        if fname.endswith(".csv"):
            df = pd.read_csv(fname)
        else:
            df = pd.read_excel(fname, engine="openpyxl")

        # Clean up and normalize columns
        df.columns = [c.lower().strip() for c in df.columns]
        if "x" not in df.columns or "y" not in df.columns:
            # Try guessing numeric columns
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if len(numeric_cols) >= 2:
                df = df.rename(columns={numeric_cols[0]: "x", numeric_cols[1]: "y"})
            else:
                raise ValueError(f"{fname} does not have numeric x/y columns")

        return df[["x", "y"]].dropna()

    except Exception as e:
        raise RuntimeError(f"Error loading data for {player} ({hand}): {e}")

# ------------------------------------------------
# 🧮 Optimization logic
# ------------------------------------------------
def reward(ball, fielder_pos):
    x, y = ball["x"], ball["y"]
    fx, fy = fielder_pos
    return -np.sqrt((x - fx)**2 + (y - fy)**2)

def optimize(df):
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
    """Average two position sets (for 'Both' handed)."""
    return {f: ((pos1[f][0] + pos2[f][0]) / 2, (pos1[f][1] + pos2[f][1]) / 2) for f in pos1}

# ------------------------------------------------
# 📈 Visualization
# ------------------------------------------------
def plot_positions(df, positions):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(df["x"], df["y"], c="gray", alpha=0.5, label="Batted Balls")
    for name, (x, y) in positions.items():
        ax.scatter(x, y, s=100, label=name)
        ax.text(x+5, y+5, name, fontsize=9, color='blue')
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Horizontal Distance (ft)")
    ax.set_ylabel("Vertical Distance (ft)")
    ax.set_title("Optimized Outfield Positions")
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ------------------------------------------------
# 🌐 Flask Routes
# ------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    positions, plot_data, error = None, None, None
    try:
        if request.method == "POST":
            player = request.form.get("player")
            hand = request.form.get("pitcher_hand")

            if hand == "B":
                df_L = safe_load_csv_or_xlsm(player, "L")
                df_R = safe_load_csv_or_xlsm(player, "R")
                pos_L = optimize(df_L)
                pos_R = optimize(df_R)
                positions = average_positions(pos_L, pos_R)
                df = pd.concat([df_L, df_R])
            else:
                df = safe_load_csv_or_xlsm(player, hand)
                positions = optimize(df)

            plot_data = plot_positions(df, positions)
            pd.DataFrame.from_dict(positions, orient="index", columns=["X","Y"]).to_csv("optimized_positions.csv")

    except Exception as e:
        error = str(e)

    return render_template_string(HTML_TEMPLATE, positions=positions, plot_data=plot_data, error=error)

@app.route("/download")
def download():
    if not os.path.exists("optimized_positions.csv"):
        return "No results yet. Please run optimization first.", 404
    return send_file("optimized_positions.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
