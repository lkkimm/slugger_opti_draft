# app.py --- demo backend with stadium overlay

import io, base64, random
from flask import Flask, jsonify, request, render_template_string, send_file
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

app = Flask(__name__)

# ---------------------------------------
# Simple inline frontend for demo
# ---------------------------------------
HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Outfield Optimizer (Demo)</title>
  <style>
    body{font-family:sans-serif;margin:2rem;background:#f4f4f4;}
    h1{color:#002b5c;}
    form{background:white;padding:1rem;border-radius:10px;width:300px;float:left;}
    #plot{float:left;margin-left:2rem;}
    img{border-radius:10px;max-width:700px;}
  </style>
</head>
<body>
  <h1>⚾ Outfield Optimizer (Stadium Demo)</h1>
  <form id="form">
    <label>Batter:</label><br>
    <select id="player">
      <option>Dickerson</option>
      <option>Turner</option>
    </select><br><br>
    <label>Pitcher Handedness:</label><br>
    <select id="hand">
      <option value="L">L</option>
      <option value="R">R</option>
      <option value="B">Both</option>
    </select><br><br>
    <button type="button" onclick="run()">Generate Spray Chart</button>
  </form>
  <div id="plot"></div>
<script>
async function run(){
  const player=document.getElementById("player").value;
  const hand=document.getElementById("hand").value;
  const res=await fetch("/api/compute",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({players:[player],handedness:hand})});
  const data=await res.json();
  if(!data.ok){alert("Error: "+data.error);return;}
  document.getElementById("plot").innerHTML=
   `<h3>${data.player} vs ${data.handedness}</h3>
    <ul>${Object.entries(data.positions).map(([f,[x,y]])=>`<li>${f}: X=${x.toFixed(1)}, Y=${y.toFixed(1)}</li>`).join("")}</ul>
    <img src="data:image/png;base64,${data.image_base64}"/>`;
}
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

# ---------------------------------------
# Demo endpoints
# ---------------------------------------
@app.route("/api/players")
def api_players():
    return jsonify(["Dickerson","Turner"])

@app.route("/api/compute", methods=["POST"])
def api_compute():
    """
    Creates a fake spray chart overlayed on a stadium image.
    """
    req = request.get_json(force=True)
    player = (req.get("players") or ["Unknown"])[0]
    hand = str(req.get("handedness","L")).upper()

    # --- generate random spray data ---
    np.random.seed(1)
    n = 80
    x = np.random.uniform(60, 240, n)
    y = np.random.uniform(220, 380, n)
    df = pd.DataFrame({"x":x,"y":y})

    # --- mock optimized positions ---
    positions = {
        "LF": [95, 300],
        "CF": [155, 360],
        "RF": [215, 300],
    }

    # --- make plot ---
    fig, ax = plt.subplots(figsize=(7,6))
    try:
        # if you have a stadium background image (e.g., static/stadium.png)
        bg = mpimg.imread("stadium.png")
        ax.imshow(bg, extent=[0,300,0,420], alpha=0.9, zorder=0)
    except Exception:
        ax.set_facecolor("lightgreen")

    ax.scatter(df.x, df.y, c="orange", s=25, alpha=0.6, label="Batted Balls", zorder=2)

    # draw rectangles for fielders
    for f,(fx,fy) in positions.items():
        rect = Rectangle((fx-5, fy-5), 10, 10, linewidth=2, edgecolor="red", facecolor="none", zorder=3)
        ax.add_patch(rect)
        ax.text(fx+8, fy+8, f, color="red", fontsize=10, weight="bold", zorder=4)

    ax.set_xlim(0,300); ax.set_ylim(200,420)
    ax.set_xlabel("Horizontal (ft)"); ax.set_ylabel("Depth (ft)")
    ax.set_title(f"{player} ({hand}) - Optimized Outfield Positioning")
    ax.legend(loc="upper right")
    buf=io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # CSV for download
    pd.DataFrame.from_dict(positions, orient="index", columns=["X","Y"]).to_csv("optimized_positions.csv")

    return jsonify({
        "ok":True,
        "player":player,
        "handedness":hand,
        "positions":positions,
        "image_base64":img_b64,
        "download_url":"/download"
    })

@app.route("/download")
def download():
    return send_file("optimized_positions.csv", as_attachment=True)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080)
