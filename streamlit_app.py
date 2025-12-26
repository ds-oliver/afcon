import streamlit as st
import pandas as pd
import seaborn as sns
import requests
from datetime import timedelta

# Page configuration
st.set_page_config(
    page_title="AFCON Fantasy Data",
    page_icon="⚽",
    layout="wide"
)

round_number = 1  # local file suffix
round_id = 803    # sofascore round id used by the API

st.title("⚽ AFCON Fantasy Data")
st.markdown("---")


@st.cache_data
def load_data(rnd: int) -> pd.DataFrame:
    """League ownership enriched dataset (used for overview table)."""
    df = pd.read_csv(f"data/afcon_fantasy_market_{rnd}_with_league_ownership.csv")
    percentage_cols = ['League Own %', 'League Start %', 'League Cpt %']
    for col in percentage_cols:
        df[col] = df[col].fillna(0)
        df[col] = df[col].round(2)
        df[col] = df[col] * 100  # convert to percentage

    if 'Event Start Timestamp' in df.columns:
        df['Event Start Timestamp'] = pd.to_datetime(df['Event Start Timestamp'], utc=True)
    return df


@st.cache_data
def load_market_base(rnd: int) -> pd.DataFrame:
    """Raw market dataset with richer stats, used for transfer suggestions."""
    df = pd.read_csv(f"data/afcon_fantasy_market_{rnd}.csv")
    numeric_cols = ["price", "expected_points", "total_points", "owned_percentage"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def fetch_user_squad(user_id: str, api_round_id: int) -> dict:
    """Fetch a user's squad from SofaScore."""
    url = f"https://www.sofascore.com/api/v1/fantasy/user/{user_id}/round/{api_round_id}/squad"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def fetch_league_participants(league_id: int) -> pd.DataFrame:
    """Fetch league participants to resolve user ids from team names."""
    url = f"https://www.sofascore.com/api/v1/fantasy/league/{league_id}/participants?page=0&q="
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    participants = resp.json().get("participants") or []
    return pd.DataFrame(participants)


def resolve_user_id(team_name: str, league_id: int) -> str:
    """Return the user id for a given team name inside a league (case-insensitive partial match)."""
    participants = fetch_league_participants(league_id)
    if participants.empty:
        raise ValueError(f"No participants returned for league {league_id}")

    matches = participants[
        participants["teamName"].str.contains(team_name, case=False, na=False)
    ]
    if matches.empty:
        raise ValueError(f"Team name '{team_name}' not found in league {league_id}")
    # Prefer exact match if available
    exact = matches[matches["teamName"].str.casefold() == team_name.casefold()]
    row = exact.iloc[0] if not exact.empty else matches.iloc[0]
    return str(row["userId"])


def normalize_squad(payload: dict) -> pd.DataFrame:
    """Flatten squad payload to a friendly dataframe."""
    squad = payload.get("squad", {})
    players = squad.get("players") or []
    rows = []

    for entry in players:
        fantasy = entry.get("fantasyPlayer") or {}
        player = fantasy.get("player") or {}
        team = fantasy.get("team") or entry.get("team") or {}
        fixtures = entry.get("fixtures") or []

        def _ts(f):
            return f.get("eventStartTimestamp") or 0

        fixtures_sorted = sorted(
            [f for f in fixtures if f.get("eventStartTimestamp")], key=_ts
        )
        next_fixture = fixtures_sorted[0] if fixtures_sorted else (fixtures[0] if fixtures else {})
        next_team = next_fixture.get("team") or {}
        kickoff_ts = next_fixture.get("eventStartTimestamp")
        kickoff_dt = pd.to_datetime(kickoff_ts, unit="s", utc=True) if kickoff_ts else None

        rows.append(
            {
                "player_id": player.get("id"),
                "Player": player.get("name"),
                "Team": team.get("name"),
                "Pos": fantasy.get("position") or player.get("position"),
                "Price": entry.get("price") or fantasy.get("price"),
                "Round Points": entry.get("score"),
                "Expected Points": entry.get("expectedPoints"),
                "Is Starter": not entry.get("substitute", False),
                "Captain": entry.get("captain", False),
                "Locked": entry.get("isLocked", False),
                "Next Opponent": next_team.get("name"),
                "Fixture Difficulty": next_fixture.get("fixtureDifficulty"),
                "Kickoff (UTC)": kickoff_dt,
            }
        )

    return pd.DataFrame(rows)


df = load_data(round_number)
market_df = load_market_base(round_number)

# Squad fetcher
st.sidebar.header("Squad viewer")
with st.sidebar.form("squad_form"):
    lookup_mode = st.radio("Lookup by", ["Team name", "User ID"], horizontal=True)
    if lookup_mode == "Team name":
        user_id_input = st.text_input("Team name", value="Draftalchemy")
    else:
        user_id_input = st.text_input("User ID", value="e.g., long alphanumeric string")
    league_id_input = st.number_input("League id (for team lookup)", value=87294, min_value=1, step=1)
    round_id_input = st.number_input("Round id", value=round_id, min_value=1, step=1)
    squad_submit = st.form_submit_button("Load squad")

if "squad_payload" not in st.session_state:
    st.session_state["squad_payload"] = None

if squad_submit:
    try:
        if lookup_mode == "Team name":
            if not team_name_input.strip():
                raise ValueError("Enter a team name.")
            resolved_user_id = resolve_user_id(team_name_input.strip(), int(league_id_input))
        else:
            if not user_id_input.strip():
                raise ValueError("Enter a user id.")
            resolved_user_id = user_id_input.strip()

        with st.spinner("Fetching squad..."):
            payload = fetch_user_squad(resolved_user_id, int(round_id_input))
        st.session_state["squad_payload"] = payload
        st.session_state["squad_meta"] = {
            "user_id": resolved_user_id,
            "round_id": int(round_id_input),
        }
        st.success(f"Squad loaded for {resolved_user_id}.")
    except Exception as exc:
        st.error(f"Unable to fetch squad: {exc}")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Position filter
positions = ['All'] + sorted(df['Pos'].dropna().unique().tolist())
selected_position = st.sidebar.selectbox("Position", positions)

games = ['All', 'Current', 'Remaining']
selected_games = st.sidebar.selectbox("Games", games)

# Apply filters
filtered_df = df.copy()
if selected_position != 'All':
    filtered_df = filtered_df[filtered_df['Pos'] == selected_position]

if selected_games == 'Current':
    now_utc = pd.Timestamp.now(tz='UTC')
    # Filter for events within 2.5 hours before and 1 hour after current time
    start_time = now_utc - timedelta(hours=2.25)
    end_time = now_utc + timedelta(hours=1)
    filtered_df = filtered_df[(filtered_df['Event Start Timestamp'] > start_time) & (filtered_df['Event Start Timestamp'] < end_time)]
elif selected_games == 'Remaining':
    now_utc = pd.Timestamp.now(tz='UTC')
    # Filter for events within 2.5 hours before and 1 hour after current time
    start_time = now_utc - timedelta(hours=2.25)
    filtered_df = filtered_df[(filtered_df['Event Start Timestamp'] > start_time)]


# Team filter
teams = ['All'] + sorted(filtered_df['Team'].dropna().unique().tolist())
selected_team = st.sidebar.selectbox("Team", teams)

if selected_team != 'All':
    filtered_df = filtered_df[filtered_df['Team'] == selected_team]

# Display stats
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Players", len(filtered_df))
col2.metric("Unique Teams", filtered_df['Team'].nunique())
col3.metric("Unique Positions", filtered_df['Pos'].nunique())
col4.metric("Players Owned", filtered_df[filtered_df['League Owners'].notna()]['Player'].nunique())
col5.metric("Total Lge Own %", filtered_df[filtered_df['League Owners'].notna()]['League Own %'].sum().round(2))
col6.metric("Total Global Own %", filtered_df['Global Own %'].sum().round(2))

st.markdown("---")

# 

# Configure column display
column_config = {
    "Player": st.column_config.TextColumn("Player", width="medium"),
    "Team": st.column_config.TextColumn("Team", width="small"),
    "Pos": st.column_config.TextColumn("Position", width="small"),
    "Price": st.column_config.NumberColumn("Price", format="%.1f"),
    "Total Points": st.column_config.NumberColumn("Total Points", format="%.1f"),
    "Round Points": st.column_config.NumberColumn("Round Points", format="%.1f"),
    "Global Own %": st.column_config.NumberColumn("Global Own %", format="%.1f%%"),
    "League Own %": st.column_config.NumberColumn("League Own %", format="%.1f%%"),
    "League Start %": st.column_config.NumberColumn("League Start %", format="%.1f%%"),
    "League Cpt %": st.column_config.NumberColumn("League Cpt %", format="%.1f%%"),
    "League Owners": st.column_config.TextColumn("League Owners", width="medium"),
    "Rnd Strt": st.column_config.NumberColumn("Rnd Strt", format="%.0f"),
}

# Prepare dataframe for display
display_df = filtered_df.copy()

display_df = display_df.drop(columns=['Event Start Timestamp'])

# Create color map for gradient
cm2 = sns.diverging_palette(0, 125, s=60, l=85, as_cmap=True)

# Apply background gradient to percentage columns
styled_df = display_df.style.background_gradient(
    cmap=cm2,
    subset=['Global Own %', 'League Own %', 'League Start %', 'League Cpt %', 'Total Points', 'Round Points', 'Rnd Strt', 'Price']
)

# Display the dataframe
# Try with column_config first (for images), fallback to styled if needed
st.dataframe(
    styled_df,
    column_config=column_config,
    use_container_width=True,
    hide_index=True,
    height=600
)

# Additional info
st.markdown("---")
st.caption(f"Showing {len(filtered_df)} of {len(df)} players")

# Squad + transfer planner
st.markdown("---")
st.header("Squad & transfers")

squad_payload = st.session_state.get("squad_payload")
if not squad_payload:
    st.info("Load your squad from the sidebar to view players and plan transfers.")
else:
    squad_df = normalize_squad(squad_payload)
    squad_meta = squad_payload.get("squad", {})
    user_round = squad_payload.get("userRound", {})

    remaining_budget = float(squad_meta.get("remainingBudget", 0) or 0)
    squad_name = squad_meta.get("name") or "Squad"
    squad_score = user_round.get("score")
    free_transfers = user_round.get("freeTransfers")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Squad", squad_name)
    c2.metric("Score", squad_score if squad_score is not None else "—")
    c3.metric("Free transfers", free_transfers if free_transfers is not None else "—")
    c4.metric("Remaining budget", f"{remaining_budget:.1f}")

    if squad_df.empty:
        st.warning("Squad payload loaded but contains no players.")
    else:
        squad_display = squad_df.copy()
        for bool_col in ["Is Starter", "Captain", "Locked"]:
            if bool_col in squad_display.columns:
                squad_display[bool_col] = squad_display[bool_col].fillna(False)
        squad_display["Kickoff (UTC)"] = pd.to_datetime(
            squad_display["Kickoff (UTC)"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M")

        squad_column_config = {
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "Team": st.column_config.TextColumn("Team", width="small"),
            "Pos": st.column_config.TextColumn("Pos", width="small"),
            "Price": st.column_config.NumberColumn("Price", format="%.1f"),
            "Round Points": st.column_config.NumberColumn("Round Points", format="%.1f"),
            "Expected Points": st.column_config.NumberColumn("Expected Points", format="%.1f"),
            "Is Starter": st.column_config.CheckboxColumn("Starter"),
            "Captain": st.column_config.CheckboxColumn("Captain"),
            "Locked": st.column_config.CheckboxColumn("Locked"),
            "Next Opponent": st.column_config.TextColumn("Next Opponent", width="small"),
            "Fixture Difficulty": st.column_config.TextColumn("Difficulty", width="small"),
            "Kickoff (UTC)": st.column_config.TextColumn("Kickoff (UTC)", width="medium"),
        }

        st.subheader("Current squad")
        st.dataframe(
            squad_display,
            column_config=squad_column_config,
            use_container_width=True,
            hide_index=True,
            height=460,
        )

        # Transfer helper
        st.subheader("Transfer helper")
        selectable = squad_df[["player_id", "Player", "Team", "Pos", "Price"]].copy()
        selectable["label"] = selectable.apply(
            lambda r: f"{r['Player']} ({r['Team']}, {r['Pos']}, {r['Price']:.1f})", axis=1
        )

        if selectable.empty:
            st.info("No players available to suggest transfers for.")
        else:
            selected_label = st.selectbox("Player to replace", selectable["label"])
            selected_row = selectable.loc[selectable["label"] == selected_label].iloc[0]

            base_price = float(selected_row["Price"]) if pd.notna(selected_row["Price"]) else 0.0
            max_price = float(base_price + (remaining_budget or 0))
            excluded_ids = set(squad_df["player_id"].dropna().astype(int).tolist())

            # Build candidate pool
            candidates = market_df.copy()
            candidates = candidates[candidates["position"] == selected_row["Pos"]]
            if excluded_ids:
                candidates = candidates[~candidates["player_id"].isin(excluded_ids)]
            candidates = candidates[pd.to_numeric(candidates["price"], errors="coerce") <= max_price]

            candidates = candidates.sort_values(
                by=["expected_points", "total_points", "owned_percentage"],
                ascending=False,
            )

            top_candidates = candidates.head(15).copy()
            top_candidates = top_candidates.rename(
                columns={
                    "name": "Player",
                    "team": "Team",
                    "position": "Pos",
                    "price": "Price",
                    "expected_points": "Expected Points",
                    "total_points": "Total Points",
                    "owned_percentage": "Global Own %",
                }
            )

            st.caption(
                f"Budget available for replacement: {max_price:.1f} "
                f"(current price {selected_row['Price']:.1f} + remaining {remaining_budget:.1f})."
            )

            transfer_columns = {
                "Player": st.column_config.TextColumn("Player", width="medium"),
                "Team": st.column_config.TextColumn("Team", width="small"),
                "Pos": st.column_config.TextColumn("Pos", width="small"),
                "Price": st.column_config.NumberColumn("Price", format="%.1f"),
                "Expected Points": st.column_config.NumberColumn("Exp Pts", format="%.1f"),
                "Total Points": st.column_config.NumberColumn("Total Pts", format="%.1f"),
                "Global Own %": st.column_config.NumberColumn("Global Own %", format="%.1f"),
                "next_opponent": st.column_config.TextColumn("Next Opponent", width="small"),
                "fixture_difficulty": st.column_config.TextColumn("Diff", width="small"),
            }

            st.dataframe(
                top_candidates[
                    [
                        "Player",
                        "Team",
                        "Pos",
                        "Price",
                        "Expected Points",
                        "Total Points",
                        "Global Own %",
                        "next_opponent",
                        "fixture_difficulty",
                    ]
                ],
                column_config=transfer_columns,
                hide_index=True,
                use_container_width=True,
                height=420,
            )
