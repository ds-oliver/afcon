from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests



def normalize_player_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested fantasy player data into a simple row."""
    fantasy = entry.get("fantasyPlayer") or {}
    player = fantasy.get("player") or entry.get("player") or {}
    team = fantasy.get("team") or entry.get("team") or {}
    fixtures = entry.get("fixtures") or []

    def _ts(f: Dict[str, Any]) -> Any:
        return f.get("eventStartTimestamp")

    fixtures_sorted = sorted(
        [f for f in fixtures if f.get("eventStartTimestamp") is not None],
        key=_ts,
    )
    next_fixture = fixtures_sorted[0] if fixtures_sorted else (fixtures[0] if fixtures else {})
    next_start_ts = next_fixture.get("eventStartTimestamp")
    next_fixture_team = next_fixture.get("team") or {}

    # print(next_fixture)
    # print(next_fixture.get("substitute", None))
    # print(next_fixture.get("playerFixtureStatus", None))
    # raise Exception("Stop here")

    return {
        "player_id": player.get("id"),
        "name": player.get("name"),
        "slug": player.get("slug"),
        "position": player.get("position"),
        "team": team.get("name"),
        "team_id": team.get("id"),
        "price": entry.get("price") or fantasy.get("price"),
        "expected_points": entry.get("expectedPoints"),
        "average_score": fantasy.get("averageScore", None),
        "average_score_rank": fantasy.get("averageScoreRank"),
        "total_points": entry.get("score"),
        "total_points_rank": fantasy.get("totalScoreRank"),
        "form": fantasy.get("form"),
        "form_rank": fantasy.get("formRank"),
        "owned_percentage": fantasy.get("ownedPercentage"),
        "owned_count": fantasy.get("ownedCount"),
        "owned_rank": fantasy.get("ownedRank"),
        "adds": fantasy.get("adds"),
        "drops": fantasy.get("drops"),
        "total_players_on_position": fantasy.get("totalPlayersOnPosition"),
        "has_left_competition": fantasy.get("hasLeftCompetition"),
        "round_player_id": entry.get("roundPlayerId"),
        "round_starter": next_fixture.get("playerFixtureStatus", True),
        "fantasy_id": fantasy.get("id") or entry.get("id"),
        "status": fantasy.get("status"),
        "fixture_difficulty": next_fixture.get("fixtureDifficulty"),
        "event_id": next_fixture.get("eventId"),
        "event_start_timestamp": next_start_ts,
        "event_start_iso_utc": (
            datetime.utcfromtimestamp(next_start_ts).isoformat() + "Z"
            if next_start_ts
            else None
        ),
        "fixtures_count": len(fixtures),
        "next_opponent": next_fixture_team.get("name"),
        "next_opponent_id": next_fixture_team.get("id"),
    }


def normalize_market(players: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Convert the players list into a DataFrame."""
    rows: List[Dict[str, Any]] = [normalize_player_entry(entry) for entry in players]
    if not rows:
        raise ValueError("No players found in the market payload.")
    return pd.DataFrame(rows)


def main(round_id: int) -> None:
    

    url = "https://www.sofascore.com/api/v1/fantasy/round/803/players?resultsPerPage=800"

    response = requests.get(url)

    players = response.json()["players"]

    # import json
    # with open(f"round{round_id}.json", "r") as f:
    #     players = json.load(f)

    # players = players["players"]

    try:
        df = normalize_market(players)
    except Exception as exc:
        raise SystemExit(f"Error normalizing market payload: {exc}")

    if df.empty:
        raise SystemExit("No players left after applying filters.")

    df = df.sort_values(by=["price", "average_score", "owned_percentage", "form", "total_points", "expected_points"], ascending=False)

    csv_path: Path = "data/afcon_fantasy_market"
    today = datetime.utcnow().strftime("%Y%m%d")
    csv_path_today = f"{csv_path}_{round_id}.csv"

    df.to_csv(csv_path_today, index=False)

    print(df.head(10))


if __name__ == "__main__":
    main(round_id=1)
