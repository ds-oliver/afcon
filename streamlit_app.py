import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="AFCON Fantasy Data",
    page_icon="⚽",
    layout="wide"
)

# Title
st.title("⚽ AFCON Fantasy Data")
st.markdown("---")

round = 1

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(f"data/afcon_fantasy_market_{round}_with_league_ownership.csv")
    # Convert percentage columns to numeric, handling empty strings
    percentage_cols = ['League Own %', 'League Start %', 'League Cpt %']
    for col in percentage_cols:
        df[col] = df[col].fillna(0)
        df[col] = df[col].round(2)
    # Multiply by 100 to convert from decimal to percentage
    for col in percentage_cols:
        df[col] = df[col] * 100
    
    # Parse Event Start Timestamp to datetime (handling UTC format with 'Z')
    if 'Event Start Timestamp' in df.columns:
        df['Event Start Timestamp'] = pd.to_datetime(df['Event Start Timestamp'], utc=True) - timedelta(hours=2)
    
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Position filter
positions = ['All'] + sorted(df['Pos'].dropna().unique().tolist())
selected_position = st.sidebar.selectbox("Position", positions)

only_current_upcoming_game = st.sidebar.checkbox("Only Current Match", value=False)



# Apply filters
filtered_df = df.copy()
if selected_position != 'All':
    filtered_df = filtered_df[filtered_df['Pos'] == selected_position]
if only_current_upcoming_game:
    # Get current UTC time (timestamps are in UTC format)
    now_utc = pd.Timestamp.now(tz='UTC')
    # Filter for events within 2.5 hours before and 1 hour after current time
    start_time = now_utc - timedelta(hours=2.5)
    end_time = now_utc + timedelta(hours=1)
    filtered_df = filtered_df[(filtered_df['Event Start Timestamp'] > start_time) & (filtered_df['Event Start Timestamp'] < end_time)]

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
    "Total Points": st.column_config.NumberColumn("Total Points", format="%.1f"),
    "Round Points": st.column_config.NumberColumn("Round Points", format="%.1f"),
    "Global Own %": st.column_config.NumberColumn("Global Own %", format="%.1f%%"),
    "League Own %": st.column_config.NumberColumn("League Own %", format="%.1f%%"),
    "League Start %": st.column_config.NumberColumn("League Start %", format="%.1f%%"),
    "League Cpt %": st.column_config.NumberColumn("League Cpt %", format="%.1f%%"),
    "League Owners": st.column_config.TextColumn("League Owners", width="medium"),
    "Round Starter": st.column_config.NumberColumn("Round Starter", format="%.1f"),
}

# Prepare dataframe for display
display_df = filtered_df.copy()

display_df = display_df.drop(columns=['Event Start Timestamp'])

# Create color map for gradient
cm2 = sns.diverging_palette(0, 125, s=60, l=85, as_cmap=True)

# Apply background gradient to percentage columns
styled_df = display_df.style.background_gradient(
    cmap=cm2,
    subset=['Global Own %', 'League Own %', 'League Start %', 'League Cpt %', 'Total Points', 'Round Points', 'Round Starter']
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
