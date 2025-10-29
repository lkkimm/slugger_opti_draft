import os, time, requests
from dotenv import load_dotenv
load_dotenv(override=False)

BASE = os.environ.get("MY_PLATFORM_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("MY_PLATFORM_API_KEY", "")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}" if API_KEY else "",
    "Accept": "application/json",
}

def _get(path, params=None):
    if not BASE:
        raise RuntimeError("Missing MY_PLATFORM_BASE_URL")
    url = f"{BASE.rstrip('/')}/{path.lstrip('/')}"
    r = requests.get(url, headers=HEADERS, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_batted_balls(player_ids=None, handedness=None, start_date=None, end_date=None, limit=5000):
    # Adjust param names/path to match your API
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "hand": handedness,
        "limit": min(limit, 5000)
    }
    if player_ids:
        params["player_ids"] = ",".join(map(str, player_ids))

    rows, cursor = [], None
    while True:
        if cursor: params["cursor"] = cursor
        data = _get("/v1/batted-balls", params)
        items = data.get("items", data.get("results", []))
        rows.extend(items)
        cursor = data.get("next_cursor")
        if not cursor or len(rows) >= limit:
            break
        time.sleep(0.15)
    return rows[:limit]

def fetch_players(start_date=None, end_date=None, handedness=None, limit=5000):
    """
    If your platform has /v1/players, call that instead.
    Fallback dedupes from batted balls.
    """
    # Example if you have a players endpoint:
    # data = _get("/v1/players", {"start_date": start_date, "end_date": end_date, "hand": handedness, "limit": limit})
    # return data.get("items", data.get("results", []))

    rows = fetch_batted_balls(
        player_ids=None,
        handedness=handedness,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    seen, players = set(), []
    for r in rows:
        pid = r.get("batter_id")
        name = r.get("batter_name")
        hand = r.get("bat_side")
        key = (pid, name, hand)
        if key not in seen and pid is not None:
            seen.add(key)
            players.append({"player_id": pid, "player": name, "handedness": hand})
    return players
