# adapter.py (add this)
def fetch_players(start_date=None, end_date=None, handedness=None, limit=5000):
    """
    Returns a list of dicts [{"player_id": ..., "player": ..., "handedness": ...}, ...]
    Implement with your actual endpoint; fallback shown uses batted-balls unique values.
    """
    # Fallback: dedupe from balls (keeps it simple)
    rows = fetch_batted_balls(
        player_ids=None,
        handedness=handedness,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    # Minimal unique set
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
