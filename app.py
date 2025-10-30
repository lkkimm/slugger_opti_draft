from flask import Flask, render_template_string, request, send_file
import pandas as pd
import numpy as np
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# --- Simple HTML UI ---
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
    <p>Choose a batter and pitcher handedness to calculate optimized outfield positions.</p>

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
        </select>
        <br>
        <button type="submit">Compute Optimal Positions</button>
    </form>

    {% if positions %}
    <div class="result">
        <h3>Optimal Outfield Positions:</h3>
        <ul>
            {% for fielder, coords in positions.items() %}
                <li><b>{{ fielder }}</b>: X={{ coords[0] }}, Y={{ coords[1] }}</li>
            {% endfor %}
        </ul>
        <a href="/download">📄 Download Printable CSV</a><br>
        <img src="data:image/png;base64,{{ plot_data }}" alt="Spray Chart">
    </div>
    {% endif %}
</body>
</html>
"""

# --- Utility Functions ---
def load_sample_data(player, hand):
    """Load sample CSVs for players."""
    if player == "Dickerson":
        path = "Dickerson_L.csv" if hand == "L" else "Dickerson_R (1).csv"
    else:
        path = "Turner- L 2021-06-20-2 (2).xlsm"
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path, engine="openpyxl")
    if "x" not in df.columns or "y" not in df.columns:
        df = df.rename(columns={df.columns[0]: "x", df.columns[1]: "y"})
    return df

def reward(ball, fielder_pos):
    x, y = ball["x"], ball["y"]
    fx, fy = fielder_pos
    dist = np.sqrt((x - fx)**2 + (y - fy)**2)
    return -dist

def optimize(df):
    lf_range = [(x, y) for x in range(60, 120, 10) for y in range(250, 350, 10)]
    cf_range = [(x, y) for x in range(120, 180, 10) for y in range(300, 400, 10)]
    rf_range = [(x, y) for x in range(180, 240, 10) for y in range(250, 350, 10)]

    best_score = float('inf')
    best_positions = {}

    for lf in lf_range:
        for cf in cf_range:
            for rf in rf_range:
                total_penalty = 0
                for _, b in df.iterrows():
                    total_penalty += min(
                        reward(b, lf),
                        reward(b, cf),
                        reward(b, rf)
                    )
                if total_penalty < best_score:
                    best_score = total_penalty
                    best_positions = {"LF": lf, "CF": cf, "RF": rf}
    return best_positions

def create_plot(df, positions):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(df["x"], df["y"], c="gray", alpha=0.4, label="Batted Balls")
    for f, (x, y) in positions.items():
        ax.scatter(x, y, s=100, label=f)
        ax.text(x+3, y+3, f, fontsize=9, color="blue")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 400)
    ax.set_title("Optimized Outfield Positions")
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    import base64
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- Flask Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    positions, plot_data = None, None
    if request.method == "POST":
        player = request.form["player"]
        hand = request.form["pitcher_hand"]
        df = load_sample_data(player, hand)
        positions = optimize(df)
        plot_data = create_plot(df, positions)
        # Save printable CSV for download
        pd.DataFrame.from_dict(positions, orient="index", columns=["X", "Y"]).to_csv("optimized_positions.csv")
    return render_template_string(HTML_TEMPLATE, positions=positions, plot_data=plot_data)

@app.route("/download")
def download():
    return send_file("optimized_positions.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
