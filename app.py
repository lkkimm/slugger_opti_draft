# app.py  ----  Guaranteed working demo backend
import io, base64, random
from flask import Flask, jsonify, request, send_file, render_template_string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

# ------------------------------------------
# Minimal front page (keeps your UI layout)
# ------------------------------------------
HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Spray Chart Optimizer (Demo)</title>
  <style>
    body {font-family:sans-serif; margin:2rem; background:#f4f4f4;}
    h1 {color:#002b5c;}
    form {background:white; padding:1rem; border-radius:10px; width:300px; float:left;}
    #plot {float:left; margin-left:2rem;}
    img {border-radius:10px; max-width:600px;}
  </style>
</head>
<body>
  <h1>⚾ Spray Chart Optimizer (Demo)</h1>
  <form id="form">
    <label>Handedness:</label><br>
    <select id="hand">
      <option value="L">Left</option>
      <option value="R">Right</option>
      <option value="B">Both</option>
    </select><br><br>
    <label>Players:</label><br>
    <select id="player">
      <option>Dickerson</option>
      <option>Turner</option>
    </select><br><br>
    <button type="button" onclick="run()">Generate Spray Chart</button>
  </form>
  <div id="plot"></div>

  <script>
    async function run(){
      const player=document.getElementById("player").value;
      const hand=document.getElementById("hand").value;
      const res=await fetch("/api/compute",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({players:[player],handedness:hand})
      });
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

# ------------------------------------------
# Demo compute endpoint (no files required)
# ------------------------------------------
@app.route("/api/players")
def api_players():
    # UI will populate with this list
    return jsonify(["Dickerson","Turner"])

@app.route("/api/compute", methods=["POST"])
def api_compute():
    """
    Returns fake optimized positions + generated spray chart.
    Always works, even with no data files.
    """
    req = request.get_json(force=True)
    player = (req.get("players") or ["Unknown"])[0]
    hand = str(req.get("handedness","L")).upper()

    # --- fake random batted balls ---
    np.random.seed(0)
    n = 60
    x = np.random.uniform(40,250,n)
    y = np.random.uniform(200,400,n)
    df = pd.DataFrame({"x":x,"y":y})

    # --- fake optimized positions (spread realistically) ---
    positions = {
        "LF": [random.uniform(70,110), random.uniform(280,320)],
        "CF": [random.uniform(130,170), random.uniform(340,380)],
        "RF": [random.uniform(200,230), random.uniform(280,320)],
    }

    # --- make plot ---
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(df.x, df.y, c="gray", alpha=0.5, label="Batted Balls")
    for f,(fx,fy) in positions.items():
        ax.scatter(fx,fy,s=100,label=f)
        ax.text(fx+4,fy+4,f,color="blue")
    ax.set_xlim(0,300); ax.set_ylim(0,420)
    ax.set_title(f"Optimized Outfield Positions ({player} vs {hand})")
    ax.set_xlabel("Horizontal (ft)"); ax.set_ylabel("Depth (ft)")
    ax.legend()
    buf=io.BytesIO(); plt.savefig(buf,format="png",bbox_inches="tight"); plt.close(fig)
    img_b64=base64.b64encode(buf.getvalue()).decode("utf-8")

    # --- make printable CSV ---
    pd.DataFrame.from_dict(positions,orient="index",columns=["X","Y"]).to_csv("optimized_positions.csv")

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
    return send_file("optimized_positions.csv",as_attachment=True)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)
