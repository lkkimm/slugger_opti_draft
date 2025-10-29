import pandas as pd
K_PLAYER      = "batter_name"
K_HAND        = "bat_side"
K_EV          = "exit_velocity_mph"
K_LA          = "launch_angle_deg"
K_SPRAY       = "spray_angle_deg"
K_HT          = "hangtime_s"
K_PLAYER_ID   = "batter_id"
K_AT          = "timestamp"

def to_dataframe(items):
    if not items:
        return pd.DataFrame(columns=[
            "player","player_id","handedness","ev_mph","la_deg","spray_deg","hangtime_s","timestamp"
        ])
    recs = []
    for it in items:
        recs.append({
            "player":      it.get(K_PLAYER),
            "player_id":   it.get(K_PLAYER_ID),
            "handedness":  it.get(K_HAND),
            "ev_mph":      it.get(K_EV),
            "la_deg":      it.get(K_LA),
            "spray_deg":   it.get(K_SPRAY),
            "hangtime_s":  it.get(K_HT),
            "timestamp":   it.get(K_AT),
        })
    return pd.DataFrame.from_records(recs)
