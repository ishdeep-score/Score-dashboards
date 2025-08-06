from utils import *
import random
import streamlit as st
import pandas as pd
import os,glob
#from st_aggrid import AgGrid
#from st_aggrid.grid_options_builder import GridOptionsBuilder
import psycopg2


team_dict = { 
        65: 'Barcelona',
        63: 'Atletico Madrid',
        52: 'Real Madrid',
        53: 'Atletic Club',
        839: 'Villarreal',
        54: 'Real Betis',
        64: 'Rayo Vallecano',
        51: 'Mallorca',
        68: 'Real Sociedad',
        62: 'Celta Vigo',
        131: 'Osasuna',
        67: 'Sevilla',
        2783: 'Girona',
        819: 'Getafe',
        70: 'Espanyol',
        825: 'Leganes',
        838: 'Las Palmas',
        55 : 'Valencia',
        60 : 'Deportivo Alaves',
        58: 'Real Valladolid',
        13: 'Arsenal',
        161: 'Wolves',
        24: 'Aston Villa',
        211: 'Brighton',
        30: 'Tottenham',
        167: 'Man City',
        14: 'Leicester',
        18: 'Southampton',
        183: 'Bournemouth',
        26: 'Liverpool',
        23: 'Newcastle',
        15: 'Chelsea',
        174: 'Nottingham Forest',
        29: 'West Ham',
        32: 'Man Utd',
        170: 'Fulham',
        189: 'Brentford',
        162: 'Crystal Palace',
        31: 'Everton',
        165: 'Ipswich',
        37: 'Bayern Munich',
        36: 'Bayer Leverkusen',
        45: 'Eintracht Frankfurt',
        219: 'Mainz 05',
        50: 'Freiburg',
        7614: 'RB Leipzig',
        33: 'Wolfsburg',
        134: 'Borussia M.Gladbach',
        41: 'VfB Stuttgart',
        44: 'Borussia Dortmund',
        1730: 'Augsburg',
        42: 'Werder Bremen',
        1211: 'Hoffenheim',
        796: 'Union Berlin',
        283: 'St. Pauli',
        1206: 'Holstein Kiel',
        4852: 'FC Heidenheim',
        109: 'Bochum',
        75 : 'Inter',
        276 : 'Napoli',
        300 : 'Atalanta',
        87 : 'Juventus',
        77 : 'Lazio',
        71 : 'Bologna',
        73 : 'Fiorentina',
        84 : 'Roma',
        80 : 'AC Milan',
        86 : 'Udinese',
        72 : 'Torino',
        278 : 'Genoa',
        1290 : 'Como',
        76 : 'Verona',
        78 : 'Cagliari',
        79 : 'Lecce',
        24341 : 'Parma Calcio',
        272 : 'Empoli',
        85 : 'Venezia',
        269 : 'Monza',
        304 : 'PSG',
        249 : 'Marseille',
        613 : 'Nice',
        248 : 'Monaco',
        607 : 'Lille',
        228 : 'Lyon',
        148 : 'Strasbourg',
        246 : 'Toulouse',
        309 : 'Lens',
        2332 : 'Brest',
        313 : 'Rennes',
        308 : 'Auxerre',
        614 : 'Angers',
        302 : 'Nantes',
        950 : 'Reims',
        217 : 'Le Havre',
        145 : 'Saint-Etienne',
        311 : 'Montpellier',
        299 : 'Benfica',
        129 : 'PSV',
        336 : 'Germany',
        340 : 'Portugal',
        338 : 'Spain',
        341 : 'France',
        342 : 'Poland',
        424 : 'Scotland',
        337 : 'Croatia',
        339 : 'Belgium',
        343 : 'Italy',
        325 : 'Israel',
        768 : 'Bosnia',
        335 : 'Netherlands',
        327 : 'Hungary',
        425 : 'Denmark',
        423 : 'Switzerland',
        771 : 'Serbia'
}
    
team_colors = {
    'Barcelona': '#A50044',
    'Atletico Madrid': '#CE3524',
    'Real Madrid': '#FCBF00',
    'Atletic Club': '#E0092C',
    'Villarreal': '#FFE667',
    'Real Betis': '#0BB363',
    'Rayo Vallecano': '#E53027',
    'Mallorca': '#E20613',
    'Real Sociedad': '#0067B1',
    'Celta Vigo': '#8AC3EE',
    'Osasuna': '#E60026',
    'Sevilla': '#D00027',
    'Girona': '#DA291C',
    'Getafe': '#003DA5',
    'Espanyol': '#00529F',
    'Leganes': '#005BAC',
    'Las Palmas': '#FEDD00',
    'Valencia': '#F18E00',
    'Deportivo Alaves': '#005BAC',
    'Real Valladolid': '#7A1E8B',
    'Arsenal': '#EF0107',
    'Wolves': '#FDB913',
    'Aston Villa': '#95BFE5',
    'Brighton': '#0057B8',
    'Tottenham': '#132257',
    'Man City': '#6CABDD',
    'Leicester': '#003090',
    'Southampton': '#D71920',
    'Bournemouth': '#DA291C',
    'Liverpool': '#C8102E',
    'Newcastle': "#F8F5F6",
    'Chelsea': '#034694',
    'Nottingham Forest': '#E53233',
    'West Ham': '#7A263A',
    'Man Utd': '#DA291C',
    'Fulham': "#FFFFFF",
    'Brentford': '#E30613',
    'Crystal Palace': '#1B458F',
    'Everton': '#003399',
    'Ipswich': '#005BAC',
    'Bayern Munich': '#DC052D',
    'Bayer Leverkusen': '#E30613',
    'Eintracht Frankfurt': "#FFFFFF",
    'Mainz 05': '#C8102E',
    'Freiburg': "#F9F4F4",
    'RB Leipzig': '#E4002B',
    'Wolfsburg': '#65B32E',
    'Borussia M.Gladbach': "#EEF3ED",
    'VfB Stuttgart': '#E30613',
    'Borussia Dortmund': '#FDE100',
    'Augsburg': '#C8102E',
    'Werder Bremen': '#1A9F3D',
    'Hoffenheim': '#005BAC',
    'Union Berlin': '#E30613',
    'St. Pauli': '#A45A2A',
    'Holstein Kiel': '#005BAC',
    'FC Heidenheim': '#E30613',
    'Bochum': '#005BAC',
    'Inter': '#1E2943',
    'Napoli': '#0082CA',
    'Atalanta': '#1C1C1C',
    'Juventus': "#F8F4F4",
    'Lazio': '#A8C6E5',
    'Bologna': '#D4001F',
    'Fiorentina': '#592C82',
    'Roma': '#8E1111',
    'AC Milan': '#FB090B',
    'Udinese': "#F2EBEB",
    'Torino': '#8B1B3A',
    'Genoa': '#C8102E',
    'Como': '#005BAC',
    'Verona': '#FCE500',
    'Cagliari': '#C8102E',
    'Lecce': '#FCE500',
    'Parma Calcio': '#FCE500',
    'Empoli': '#005BAC',
    'Venezia': "#F2ECEC",
    'Monza': '#E30613',
    'PSG': '#004170',
    'Marseille': '#009DDC',
    'Nice': '#E30613',
    'Monaco': '#ED1C24',
    'Lille': '#E30613',
    'Lyon': '#E30613',
    'Strasbourg': '#005BAC',
    'Toulouse': '#5F259F',
    'Lens': '#E30613',
    'Brest': '#E30613',
    'Rennes': '#E30613',
    'Auxerre': '#005BAC',
    'Angers': "#EEECEC",
    'Nantes': '#FCE500',
    'Reims': '#E30613',
    'Le Havre': '#005BAC',
    'Saint-Etienne': '#009639',
    'Montpellier': '#005BAC',
    'Benfica': '#E30613',
    'PSV': '#E30613',
    'Ajax': '#E30613',
    'Feyenoord': '#E30613',
    'Utrecht': '#E30613',
    'AZ-Alkmaar': '#E30613',
    'Twente': '#E30613',
    'Go Ahead Eagles': '#E30613',
    'FC Groningen': '#007A33',
    'Fortuna Sittard': '#FCE500',
    'Heracles': "#FAF7F7",
    'SC Heerenveen': '#005BAC',
    'NEC Nijmegen': '#E30613',
    'NAC Breda': '#FCE500',
    'PEC Zwolle': '#005BAC',
    'Sparta Rotterdam': '#E30613',
    'Willem II': '#E30613',
    'RKC Waalwijk': '#FCE500',
    'Almere City': '#E30613',
    'Palmeiras' : "#216348",
    'Inter Miami' : "#E067C2",
    'Porto' : '#005BAC',
    'Al Ahly' : '#E30613',
    'Botafogo' : "#E3DADA",
    'Seattle Sounders' : "#238486",
    'Boca Juniors' : "#E0E723",
    'Auckland City' : '#005BAC',
    'Flamengo' : '#E30613',
    'Esperance' : "#7B79D9",
    'Los Angeles FC' : "#949430",
    'River Plate' : '#E30613',
    'Monterrey' : '#005BAC',
    'Urawa Red Diamonds' : '#E30613',
    'Mamelodi Sundowns' : "#AEE54F",
    'Fluminense' : "#68202B",
    'Ulsan HD FC' : '#005BAC',
    'Wydad' : '#E30613',
    'Al Ain' : "#721A70",
    'Salzburg' : '#E30613',
    'Al Hilal' : '#005BAC',
    'Pachuca' : "#E39606"
    }

base_path = '/Users/ishdeepchadha/Documents/Score/Football'
logo_path = f'logos/Logo.png'
# Set the path to the locally downloaded font file
font_path = f'{base_path}/Sora_Font/Sora-Regular.ttf'

# Add the font to matplotlib
font_prop = fm.FontProperties(fname=font_path)
st.set_page_config(layout="centered")

league = st.session_state.get('league')
season = st.session_state.get('season')
home_team = st.session_state.get('home_team')
away_team = st.session_state.get('away_team')
matchId = st.session_state.get('matchId')

st.title(f"{home_team} vs {away_team}")

theme = st.radio(
    '',
    options=['🌙 Dark', '☀️ Light'],
    index=0,
    horizontal=True
)

#st.dataframe(df[df['subOn'] != 0][['index','matchId','startDate','teamName','playerName','type','situation']], width=1000)


# Now use these variables as needed

match_df = load_and_process_match_data(base_path, league, season, matchId, team_colors)
#st.write("Columns in match_df:", match_df.columns.tolist())

#match_df = match_df.sort_values(by='id').reset_index(drop=True)
home_team_col = match_df[match_df['teamName'] == home_team]['teamColor'].unique()[0]
away_team_col = match_df[match_df['teamName'] == away_team]['teamColor'].unique()[0]

## Shots - ShotMap , xG Flow , OnGoal Shots
## Passing - Passing Network , Average On Ball Positions - Passes Played , Passes Received


viz = st.selectbox(
    'Select Action',
    ['Match Dynamics','Shots', 'In Possession', 'Duels', 'Out of Possession', 'Set Pieces', 'Transitions'],
    index=0
)

if theme == '🌙 Dark':
    background = "#010b14"
    line_color = 'white'
    text_color = 'white'
    logo = mpimg.imread(logo_path)

else:
    background = "#FFFFFF"
    line_color = 'black'
    text_color = 'black'
    logo = mpimg.imread(logo_path)

if viz == 'Match Dynamics':
    st.markdown("## Match Dynamics")
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20,12))
    fig.set_facecolor(background)
    axs.set_facecolor(background)

    st.markdown("### xT Momentum Flow")
    xT_momemtum(axs,match_df,home_team,away_team,home_team_col,away_team_col,background,text_color)    
    st.pyplot(fig)

    st.markdown("### Ball Possession % and Pass Accuracy %")
    fig2, axs2 = plt.subplots(nrows=1, ncols=2, figsize=(20,9))
    fig2.set_facecolor(background)
    axs2[0].set_facecolor(background)
    axs2[1].set_facecolor(background)

    poss_df = tag_sequences_and_possessions_all_matches(match_df)
    plot_possession_windows_time_weighted(
        ax=axs2[0],
        df=poss_df,
        home_team=home_team,
        away_team= away_team,
        team_colors={home_team: home_team_col, away_team: away_team_col},
        background=background,
        text_color=text_color,
        font_prop=font_prop
    )
    st.pyplot(fig2)

if viz == 'Shots':
    #fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20,22))
    st.markdown("## Shot Map")
    pitch = Pitch(pitch_type='uefa',half=False, corner_arcs=True, pitch_color=background, line_color=line_color, linewidth=1.5)
    fig, axs = pitch.jointgrid(figheight=20,grid_width =1, left=None, bottom=None, grid_height=0.9,
                           axis=False,title_space=0,endnote_height=0,title_height=0,ax_top=False)
    fig.set_facecolor(background)

    situations = match_df['situation'].dropna().unique().tolist()
    #print(situations)
    situations = ['All'] + situations
    situation = st.radio('',options=situations,index=0,horizontal=True,label_visibility='collapsed')
    
    ax_image = add_image(
        logo, fig, left=0.77, bottom=0.85, width=0.08, height=0.08,aspect='equal'
    )
    summary_df,player_df,home_shots_df,away_shots_df = shotMap_ws(match_df,axs,fig,pitch,home_team,away_team,home_team_col,away_team_col,text_color,background,situation)
    #shotMap(match_df,axs[1],away_team,'red')
    axs['pitch'].set_xlim(-10, 115)  # example: pitch length from 0 to 120
    axs['pitch'].set_ylim(-10, 80)   # example: pitch width from 0 to 80
    st.pyplot(fig)
    st.dataframe(summary_df, width=1000)
    st.dataframe(player_df, width=1000)

    st.markdown("## xG Flow")
    fig2, axs2 = plt.subplots(nrows=1, ncols=1, figsize=(20,12))
    fig2.set_facecolor(background)

    ax_image = add_image(
        logo, fig2, left=0.83, bottom=0.88, width=0.08, height=0.08,aspect='equal'
    )

    xgFlow(axs2,home_shots_df,away_shots_df,home_team,away_team,home_team_col,away_team_col,text_color,background)
    st.pyplot(fig2)

if viz == 'In Possession':
    st.markdown("## Passing Network and Pass Combination Matrix")
    #st.dataframe(match_df[match_df['subOff'] != 0][['index','matchId','minute','second','event_time','playerName','isFirstEleven']], width=1000)
    passes_df = get_passes_df(match_df)
    team = st.radio('',options=[home_team, away_team],index=0,horizontal=True, label_visibility='collapsed')
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20,15))
    fig.set_facecolor(background)
    
    ax_image = add_image(
        logo, fig, left=0.82, bottom=0.83, width=0.07, height=0.07,aspect='equal'
    )


    if team == home_team:
        
        # Get and sort substitution minutes
        minute_of_subs = match_df[
            (match_df['teamName'] == home_team) & (match_df['type'] == 'SubstitutionOn')
        ]['minute'].unique()

        minute_of_subs = np.sort(minute_of_subs[minute_of_subs < 90])

        # Only keep substitutions that result in a 5+ minute window
        filtered_minutes = []
        prev = 0
        for m in minute_of_subs:
            if m - prev >= 5:
                filtered_minutes.append(m)
                prev = m

        # If the last window (from last sub to 90) is too short, drop the last one
        if len(filtered_minutes) > 0 and 90 - filtered_minutes[-1] < 5:
            filtered_minutes.pop()

        # Generate the options
        def ordinal(n):
            return ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth'][n - 1]

        options = ['Starting 11']
        for i in range(len(filtered_minutes)):
            options.append(f'{ordinal(i+1)} Substitution')

        # Radio selector
        substitutions = st.radio('', options=options, index=0, horizontal=True, label_visibility='collapsed')

        # Determine minute_start and minute_end
        if substitutions == 'Starting 11':
            minute_start = 0
            minute_end = filtered_minutes[0] if filtered_minutes else 90
        else:
            index = options.index(substitutions) - 1
            minute_start = filtered_minutes[index]
            minute_end = filtered_minutes[index + 1] if index + 1 < len(filtered_minutes) else 90


        fig.text(
        0.16, 0.86, f"Minute {minute_start}-{minute_end}",fontproperties=font_prop,
        ha='left', va='center', fontsize=45, color=text_color
        )
        filtered_passes_df = filter_passes_for_subwindow(match_df, passes_df, home_team, minute_start, minute_end)

        home_passes_between_df, home_average_locs_and_count_df = get_passes_between_df(home_team, filtered_passes_df)
        ci = calculate_centralization_index(home_team,filtered_passes_df)
        pass_network_visualization(axs, match_df, home_passes_between_df, home_average_locs_and_count_df, text_color, background, home_team_col, home_team, 20, False,ci)
        st.pyplot(fig)

        st.markdown("##### The centralization index signifies how much a team's passing network is focused around a few players — a higher value indicates greater reliance on central figures, while a lower value reflects a more balanced, distributed passing structure.")

        fig2,ax2 = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        fig2.set_facecolor(background)
        pass_matrix = get_pass_matrix(passes_df, home_team)
        custom_cmap = LinearSegmentedColormap.from_list("custom_green", [background, home_team_col])
        annotations = pass_matrix.map(lambda x: f"{x}" if x >= 5 else "")
        sns.heatmap(pass_matrix, annot=annotations, cmap=custom_cmap, fmt='', linewidths=1, square=True,cbar=False,annot_kws={"size": 14},ax=ax2)

        #ax.set_title('Pass Combination Matrix')
        ax2.set_xlabel('Receiver', fontproperties=font_prop,fontsize=18, color=text_color)
        ax2.set_ylabel('Passer', fontproperties=font_prop,fontsize=18, color=text_color)
        plt.xticks(rotation=90, fontproperties=font_prop,fontsize=12, color=text_color)
        plt.yticks(rotation=0, fontproperties=font_prop,fontsize=12, color=text_color)
        st.pyplot(fig2)
    else:
        # Get and sort substitution minutes
        minute_of_subs = match_df[
            (match_df['teamName'] == away_team) & (match_df['type'] == 'SubstitutionOn')
        ]['minute'].unique()

        minute_of_subs = np.sort(minute_of_subs[minute_of_subs < 90])

        # Only keep substitutions that result in a 5+ minute window
        filtered_minutes = []
        prev = 0
        for m in minute_of_subs:
            if m - prev >= 5:
                filtered_minutes.append(m)
                prev = m

        # If the last window (from last sub to 90) is too short, drop the last one
        if len(filtered_minutes) > 0 and 90 - filtered_minutes[-1] < 5:
            filtered_minutes.pop()

        # Generate the options
        def ordinal(n):
            return ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth'][n - 1]

        options = ['Starting 11']
        for i in range(len(filtered_minutes)):
            options.append(f'{ordinal(i+1)} Substitution')

        # Radio selector
        substitutions = st.radio('', options=options, index=0, horizontal=True, label_visibility='collapsed')

        # Determine minute_start and minute_end
        if substitutions == 'Starting 11':
            minute_start = 0
            minute_end = filtered_minutes[0] if filtered_minutes else 90
        else:
            index = options.index(substitutions) - 1
            minute_start = filtered_minutes[index]
            minute_end = filtered_minutes[index + 1] if index + 1 < len(filtered_minutes) else 90

        
        fig.text(
        0.16, 0.86, f"Minute {minute_start}-{minute_end}",fontproperties=font_prop,
        ha='left', va='center', fontsize=45, color=text_color
        )
        filtered_passes_df = filter_passes_for_subwindow(match_df, passes_df, away_team, minute_start, minute_end)

        away_passes_between_df, away_average_locs_and_count_df = get_passes_between_df(away_team, filtered_passes_df)
        ci = calculate_centralization_index(away_team,filtered_passes_df)
        pass_network_visualization(axs, match_df, away_passes_between_df, away_average_locs_and_count_df, text_color, background, away_team_col, away_team, 20, False,ci)
        st.pyplot(fig)

        st.markdown("##### The centralization index signifies how much a team's passing network is focused around a few players — a higher value indicates greater reliance on central figures, while a lower value reflects a more balanced, distributed passing structure.")


        fig2,ax2 = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        fig2.set_facecolor(background)
        pass_matrix = get_pass_matrix(passes_df, away_team)
        custom_cmap = LinearSegmentedColormap.from_list("custom_green", [background, away_team_col])
        annotations = pass_matrix.map(lambda x: f"{x}" if x >= 5 else "")
        sns.heatmap(pass_matrix, annot=annotations, cmap=custom_cmap, fmt='', linewidths=1, square=True,cbar=False,annot_kws={"size": 14},ax=ax2)

        #ax.set_title('Pass Combination Matrix')
        ax2.set_xlabel('Receiver', fontproperties=font_prop,fontsize=18, color=text_color)
        ax2.set_ylabel('Passer', fontproperties=font_prop,fontsize=18, color=text_color)
        plt.xticks(rotation=90, fontproperties=font_prop,fontsize=12, color=text_color)
        plt.yticks(rotation=0, fontproperties=font_prop,fontsize=12, color=text_color)
        st.pyplot(fig2)


    st.markdown("## Passing Stats Comparison")
    # Define desired order for metrics and teams
    metric_order = [
        'Total Passes', 'Passing Accuracy (%)', 'Final Third Entries',
        'Key Passes', 'Crosses', 'Long Balls',
        'Through Balls', 'Progressive Passes', 'Pen Box Passes','Expected Threat By Pass'
    ]

    team_order = [home_team, away_team]  # You can reverse this if needed

    # Get passing stats
    passes_stats_hteam = get_passing_stats(match_df, home_team)
    passes_stats_ateam = get_passing_stats(match_df, away_team)

    # Combine and reshape
    combined_stats = pd.concat([passes_stats_hteam, passes_stats_ateam], axis=0)
    combined_stats = combined_stats.melt(id_vars='Team', var_name='Metric', value_name='Value')
    combined_stats = combined_stats.pivot(index='Metric', columns='Team', values='Value')

    # Reorder rows and columns
    combined_stats = combined_stats.loc[metric_order, team_order]
    styled_stats = combined_stats.style \
    .apply(lambda row: [highlight_higher(v, row.values) for v in row], axis=1) \
    .format("{:.2f}")

    st.dataframe(styled_stats, width=1000)


    # --- Passmaps Section ---
    # Team filter above pitch and player stats
    team_filter = st.radio("Select Team", [home_team, away_team], horizontal=True, index=0)

    # Add half selection radio
    half_option = st.radio(
        "Select Half",
        options=["Full 90", "First Half", "Second Half"],
        index=0,
        horizontal=True
    )

    # Pass type selection
    passtypes = ['All', 'Crosses', 'Long Balls', 'Through Balls', 'Final Third Entries']
    passtype = st.radio('', options=passtypes, index=0, horizontal=True, label_visibility='collapsed')

    # Filter match_df for the selected half
    if half_option == "First Half":
        match_df_half = match_df[match_df['period'] == 'FirstHalf']
    elif half_option == "Second Half":
        match_df_half = match_df[match_df['period'] == 'SecondHalf']
    else:
        match_df_half = match_df

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 12))
    fig.set_facecolor(background)
    plt.subplots_adjust(wspace=0.02)

    if "selected_pass_player" not in st.session_state:
        st.session_state.selected_pass_player = None
    selected_player = st.session_state.selected_pass_player

    # --- Passmaps plotting ---
    top_passers_h, top_passers_a = passmaps(
        ax, match_df_half, home_team, home_team_col, away_team, away_team_col,
        background, text_color, passtype, selected_player, team_filter
    )
    top_passers_h = top_passers_h.reset_index(drop=True)
    top_passers_a = top_passers_a.reset_index(drop=True)

    # --- Build pass type counts, assists, xA for each player ---
    def get_pass_type_counts(df, team):
        passes = df[(df['type'] == 'Pass') & (df['teamName'] == team)].copy()
        passes['Final Third Entries'] = ((passes['x'] < 75) & (passes['endX'] >= 75) & (passes['outcomeType'] == 'Successful')).astype(int)
        passes['Crosses'] = (passes['qualifiers'].str.contains('Cross', na=False) & (passes['outcomeType'] == 'Successful')).astype(int)
        passes['Long Balls'] = (passes['qualifiers'].str.contains('Longball', na=False) & (passes['outcomeType'] == 'Successful')).astype(int)
        passes['Through Balls'] = (passes['qualifiers'].str.contains('Throughball', na=False) & (passes['outcomeType'] == 'Successful')).astype(int)
        passes['Total Passes'] = 1
        #passes['Assist'] = passes['assist'].astype(int) if 'assist' in passes.columns else 0
        passes['xA'] = passes['xA'] if 'xA' in passes.columns else 0.0

        summary = passes.groupby('playerName').agg({
            'Final Third Entries': 'sum',
            'Crosses': 'sum',
            'Long Balls': 'sum',
            'Through Balls': 'sum',
            'Total Passes': 'sum',
            'assist': 'sum',
            'xA': 'sum'
        }).reset_index()
        summary = summary.rename(columns={'assist': 'Assist'})
        summary = summary.sort_values('Total Passes', ascending=False)
        return summary

    home_pass_summary = get_pass_type_counts(match_df_half, home_team)
    away_pass_summary = get_pass_type_counts(match_df_half, away_team)

    # --- Display pitch ---
    st.pyplot(fig)
    st.markdown("Star marker indicates key pass / chance created and green line indicates an assist.")

    # --- Interactive player pass stats DataFrame ---
    if team_filter == home_team:
        pass_summary = home_pass_summary
    else:
        pass_summary = away_pass_summary

    # Add playerName as buttons in the DataFrame
    def player_button(label, key):
        return st.button(label, key=key)

    # Build a DataFrame with playerName as buttons
    button_cols = st.columns([2, 1, 1, 1, 1, 1, 1, 1])
    header = ["Player", "Final Third", "Crosses", "Long Balls", "Through Balls", "Total Passes", "Assists", "xA"]
    for i, h in enumerate(header):
        button_cols[i].markdown(f"**{h}**")

    for i, row in pass_summary.iterrows():
        cols = st.columns([2, 1, 1, 1, 1, 1, 1, 1])
        with cols[0]:
            if player_button(row['playerName'], key=f"{team_filter}_pass_{row['playerName']}"):
                st.session_state.selected_pass_player = row['playerName']
                st.rerun()
        with cols[1]:
            st.write(int(row['Final Third Entries']))
        with cols[2]:
            st.write(int(row['Crosses']))
        with cols[3]:
            st.write(int(row['Long Balls']))
        with cols[4]:
            st.write(int(row['Through Balls']))
        with cols[5]:
            st.write(int(row['Total Passes']))
        with cols[6]:
            st.write(int(row['Assist']))
        with cols[7]:
            st.write(round(row['xA'], 2))

    # --- Clear selection button ---
    if st.button("Clear Selection"):
        st.session_state.selected_pass_player = None
        st.rerun()

if viz == 'Duels':
    st.markdown("## Duels")
    duel_type = st.radio(
        '',
        options = ['Total', 'Offensive', 'Defensive', 'Aerial'],
        index=0, horizontal=True, label_visibility='collapsed'
    )
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20,15))
    fig.set_facecolor(background)
    axs.set_facecolor(background)
    h_players , a_players = plot_duels_by_type(axs,match_df,home_team,away_team,duel_type,home_team_col,away_team_col,background,text_color)
    # Rename columns to indicate teams
    h_players.columns = [f'{home_team} Player', f'{home_team} Duels Won']
    a_players.columns = [f'{away_team} Player', f'{away_team} Duels Won']

    # Concatenate side by side
    result = pd.concat([h_players, a_players], axis=1).reset_index(drop=True)
    st.pyplot(fig)
    st.dataframe(result, width=1000)

if viz == 'Out of Possession':
    st.markdown("## Defensive Actions")
    team = st.radio('',options=[home_team, away_team],index=0,horizontal=True, label_visibility='collapsed')
    halves = st.radio('',options=['Full 90','First Half', 'Second Half'],index=0,horizontal=True, label_visibility='collapsed')
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20,15))
    fig.set_facecolor(background)
    axs.set_facecolor(background)
    if team == home_team:
        defensive_actions_df = match_df[match_df['teamName'] == home_team]
        team_color = home_team_col
        if halves == 'First Half':
            defensive_actions_df = defensive_actions_df[defensive_actions_df['period'] == 'FirstHalf']
        elif halves == 'Second Half':
            defensive_actions_df = defensive_actions_df[defensive_actions_df['period'] == 'SecondHalf']
        
        if "selected_player" not in st.session_state:
            st.session_state.selected_player = None
        #defensive_block(axs,defensive_actions_df,away_team,away_team_col,background,text_color,True)
        defensive_block_with_player_actions(axs, defensive_actions_df, home_team, home_team_col, background, text_color,
                                        flipped=False, selected_player_name=st.session_state.selected_player)
        st.pyplot(fig)

        zone_filter = st.selectbox("Select Defensive Zone", ["All", "Attacking Third", "Middle Third", "Defensive Third"])
        team_df = get_defensive_action_distribution_by_type(defensive_actions_df, zone=zone_filter, halves=halves)
        # Show table row by row with a button for each player
        if st.button("Clear Selected Player"):
            st.session_state.selected_player = None
            st.rerun()

        for i, row in team_df.reset_index().iterrows():
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button(row['playerName'], key=f"player_{row['playerName']}"):
                    st.session_state.selected_player = row['playerName']  # ✅ FIXED HERE
                    st.rerun()  # Immediately rerun to update pitch
            with col2:
                st.dataframe(pd.DataFrame([row.drop(['index', 'playerName'])]), use_container_width=True)


    else:
        defensive_actions_df = match_df[match_df['teamName'] == away_team]
        team_color = away_team_col
        if halves == 'First Half':
            defensive_actions_df = defensive_actions_df[defensive_actions_df['period'] == 'FirstHalf']
        elif halves == 'Second Half':
            defensive_actions_df = defensive_actions_df[defensive_actions_df['period'] == 'SecondHalf']
        
        if "selected_player" not in st.session_state:
            st.session_state.selected_player = None
        #defensive_block(axs,defensive_actions_df,away_team,away_team_col,background,text_color,True)
        defensive_block_with_player_actions(axs, defensive_actions_df, away_team, away_team_col, background, text_color,
                                        flipped=True, selected_player_name=st.session_state.selected_player)
        st.pyplot(fig)

        zone_filter = st.selectbox("Select Defensive Zone", ["All", "Attacking Third", "Middle Third", "Defensive Third"])
        team_df = get_defensive_action_distribution_by_type(defensive_actions_df, zone=zone_filter, halves=halves)
        # Show table row by row with a button for each player
        if st.button("Clear Selected Player"):
            st.session_state.selected_player = None
            st.rerun()

        for i, row in team_df.reset_index().iterrows():
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button(row['playerName'], key=f"player_{row['playerName']}"):
                    st.session_state.selected_player = row['playerName']  # ✅ FIXED HERE
                    st.rerun()  # Immediately rerun to update pitch
            with col2:
                st.dataframe(pd.DataFrame([row.drop(['index', 'playerName'])]), use_container_width=True)

    
    
