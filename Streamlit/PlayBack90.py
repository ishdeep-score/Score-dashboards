import streamlit as st
from PIL import Image
import os
import sqlite3
import pandas as pd
import time

# --- Config ---
st.set_page_config(page_title="PlayBack90", layout="centered")
st.cache_data(ttl=600) 
LEAGUES = {
    "Premier League": "logos/premier-league.png",
    "La Liga": "logos/laliga.png",
    "Bundesliga": "logos/bundesliga.png",
    "Serie A": "logos/serie-a.png",
    "Ligue 1": "logos/ligue-1.png",
    "Champions League": "logos/champions-league.png",
}

def get_seasons_from_db(league):

    base_path = '/Users/ishdeepchadha/Documents/Score/Football'
    db_path = f'{base_path}/data extraction/score_football.db'
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT season FROM event_data WHERE league = ? ORDER BY season DESC"
    seasons = pd.read_sql_query(query, conn, params=(league,))['season'].tolist()
    conn.close()
    return seasons

def load_data_from_db(league, season):
    base_path = '/Users/ishdeepchadha/Documents/Score/Football'
    db_path = f'{base_path}/data extraction/score_football.db'
    conn = sqlite3.connect(db_path)
    query = """
        SELECT matchId,startDate,h_a,teamName,ftScore FROM event_data
        WHERE league = ? AND season = ?
    """
    df = pd.read_sql_query(query, conn, params=(league, season))
    conn.close()
    return df

def get_fixtures_from_db(league, season, limit=10):
    df = load_data_from_db(league, season)
    if df.empty:
        return []
    match_info = df.groupby('matchId').agg({
        'startDate': 'first',
        'h_a': list,
        'teamName': list,
        'ftScore': 'first'
    }).reset_index()
    match_info = match_info.sort_values('startDate', ascending=False)
    fixtures = []
    for _, row in match_info.head(limit).iterrows():
        home_team = None
        away_team = None
        for ha, team in zip(row['h_a'], row['teamName']):
            if ha == 'h':
                home_team = team
            elif ha == 'a':
                away_team = team
        fixtures.append({
            'matchId': row['matchId'],
            'home_team': home_team if home_team else "Unknown",
            'away_team': away_team if away_team else "Unknown",
            'ft_score': row['ftScore'],
            'startDate': row['startDate']
        })
    return fixtures

# --- App Logo ---
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
if os.path.exists("logos/Logo.png"):
    st.image("logos/Logo.png", width=150)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("##### PlayBack90 is a post-match analytics platform built for football fans, analysts, and coaches. It breaks down each game with rich visuals, performance metrics, and tactical statistics — giving you a deeper understanding of what really happened on the pitch.", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color:gray;'>Select a league to view its latest fixtures</p>", unsafe_allow_html=True)

def chunk_dict(d, n):
    items = list(d.items())
    return [dict(items[i:i+n]) for i in range(0, len(items), n)]

league_chunks = chunk_dict(LEAGUES, 3)

# Ensure session state is initialized
if 'selected_league' not in st.session_state:
    st.session_state['selected_league'] = None

selected_league = st.session_state['selected_league']

# Always show the grid if no league is selected
if not selected_league:
    for chunk in league_chunks:
        cols = st.columns(len(chunk))
        for idx, (league, logo_path) in enumerate(chunk.items()):
            with cols[idx]:
                if os.path.exists(logo_path):
                    img = Image.open(logo_path)
                    img.thumbnail((100, 100))
                    st.image(img, use_container_width=False)
                if st.button(f"{league}", key=f"select_{league}"):
                    with st.spinner(f"Loading {league} data..."):
                        st.session_state['selected_league'] = league
                        time.sleep(0.6)  # Slight delay so spinner is visible
                        st.rerun()

    st.stop()

LEAGUE_NAME_MAP = {
    "Premier League": "premier-league",
    "La Liga": "laliga",
    "Bundesliga": "bundesliga",
    "Serie A": "serie-a",
    "Ligue 1": "ligue-1",
    "Champions League": "champions-league",
}
display_name = st.session_state['selected_league']
db_league_name = LEAGUE_NAME_MAP[display_name]
# Only run this after a league is selected
seasons = get_seasons_from_db(db_league_name)
if not seasons:
    st.warning(f"No seasons found for {selected_league}. Please select another league.")
    st.session_state['selected_league'] = None
    st.stop()

default_season = seasons[0]
selected_season = st.selectbox("Select Season", seasons, index=0)
st.session_state['selected_season'] = selected_season

if selected_season:
    st.markdown(f"### Last 10 Fixtures: {selected_league} ({selected_season})")
    fixtures = get_fixtures_from_db(db_league_name, selected_season, limit=10)

    for i, fixture in enumerate(fixtures):
        btn_label = f"{fixture['home_team']} {fixture['ft_score']} {fixture['away_team']}"
        if st.button(btn_label, key=f"fixture_{fixture['matchId']}"):
            st.session_state['home_team'] = fixture['home_team']
            st.session_state['away_team'] = fixture['away_team']
            st.session_state['league'] = db_league_name
            st.session_state['season'] = selected_season
            st.session_state['matchId'] = fixture['matchId']
            st.switch_page("pages/Post Match Analysis.py")

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Change League"):
            st.session_state['selected_league'] = None
            st.session_state['selected_season'] = None
            st.rerun()  # This forces the script to rerun, reloading the league selection screen

    with colB:
        if st.button("See Previous Fixtures"):
            st.switch_page("pages/Post Match Analysis.py")

# --- Footer ---
st.markdown("<hr style='border: 1px solid #eee;'>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: gray; font-size: 0.85em;'>Built with ❤️ using Streamlit</div>", unsafe_allow_html=True)
