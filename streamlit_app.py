import streamlit as st
import pandas as pd
import numpy as np

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
    df = df.fillna(0)
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Position filter
positions = ['All'] + sorted(df['Pos'].dropna().unique().tolist())
selected_position = st.sidebar.selectbox("Position", positions)

# Team filter
teams = ['All'] + sorted(df['Team'].dropna().unique().tolist())
selected_team = st.sidebar.selectbox("Team", teams)

# Apply filters
filtered_df = df.copy()
if selected_position != 'All':
    filtered_df = filtered_df[filtered_df['Pos'] == selected_position]
if selected_team != 'All':
    filtered_df = filtered_df[filtered_df['Team'] == selected_team]

# Display stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Players", len(filtered_df))
col2.metric("Unique Teams", filtered_df['Team'].nunique())
col3.metric("Unique Positions", filtered_df['Pos'].nunique())
col4.metric("Avg League Own %", f"{filtered_df['League Own %'].mean():.1%}")

st.markdown("---")

# Configure column display
column_config = {
    "Player": st.column_config.TextColumn("Player", width="medium"),
    "Player Image": st.column_config.ImageColumn("Player", width="small"),
    "Team": st.column_config.TextColumn("Team", width="small"),
    "Team Image": st.column_config.ImageColumn("Team", width="small"),
    "Pos": st.column_config.TextColumn("Position", width="small"),
    "Total Points": st.column_config.NumberColumn("Total Points", format="%.1f"),
    "Round Points": st.column_config.NumberColumn("Round Points", format="%.1f"),
    "Global Own %": st.column_config.NumberColumn("Global Own %", format="%.1f%%"),
    "League Own %": st.column_config.NumberColumn("League Own %", format="%.1%"),
    "League Start %": st.column_config.NumberColumn("League Start %", format="%.1%"),
    "League Cpt %": st.column_config.NumberColumn("League Cpt %", format="%.1%"),
}

# Create a styled dataframe with color formatting
def color_percentage(val):
    """Color code percentage values"""
    if pd.isna(val):
        return 'background-color: #f0f0f0'
    # Green for high values, yellow for medium, red for low
    if val >= 0.5:
        return 'background-color: #90EE90'  # Light green
    elif val >= 0.25:
        return 'background-color: #FFE4B5'  # Light yellow
    else:
        return 'background-color: #FFB6C1'  # Light pink

# Prepare dataframe for display (exclude image URLs from styling)
display_df = filtered_df.copy()

# Apply color formatting to percentage columns
styled_df = display_df.style.applymap(
    color_percentage,
    subset=['League Own %', 'League Start %', 'League Cpt %']
)

# Display the dataframe
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
