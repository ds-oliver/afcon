import requests
import pandas as pd
import json

round = 1

url = "https://www.sofascore.com/api/v1/fantasy/league/87294/participants?page=0&q="

response = requests.get(url)

participants = pd.DataFrame(response.json()["participants"])

print(len(participants))

team_info = {}

for index, row in participants.iterrows():
    user_id = row["userId"]
    team_name = row["teamName"]

    url2 = f"https://www.sofascore.com/api/v1/fantasy/user/{user_id}/round/803/squad"

    response2 = requests.get(url2)

    score = response2.json()['userRound']["score"]
    name = response2.json()["squad"]["name"]
    players = response2.json()["squad"]["players"]
    starters = []
    substitutes = []
    captain = None
    for player_dict in players:
        player_id = player_dict['fantasyPlayer']['player']["id"]
        is_starter = not player_dict['substitute']
        is_captain = player_dict['captain']

        if is_starter:
            starters.append(player_id)
        else:
            substitutes.append(player_id)

        if is_captain:
            captain = player_id


    team_info[user_id] = {
        "team_name": team_name,
        "score": score,
        "starters": starters,
        "substitutes": substitutes,
        "captain": captain,
    }


player_team_counts = {}
player_starters_counts = {}
player_captains_counts = {}
player_owners = {}

for team_id, team_data in team_info.items():
    print(team_data["starters"])
    for starter in team_data["starters"]:
        if starter in player_team_counts:
            player_team_counts[starter] += 1
            player_owners[starter].append(team_data["team_name"])
        else:
            player_team_counts[starter] = 1
            player_owners[starter] = [team_data["team_name"]]
            

    for substitute in team_data["substitutes"]:
        if substitute in player_team_counts:
            player_team_counts[substitute] += 1
            player_owners[substitute].append(team_data["team_name"])
        else:
            player_team_counts[substitute] = 1
            player_owners[substitute] = [team_data["team_name"]]

    for starter in team_data["starters"]:
        if starter in player_starters_counts:
            player_starters_counts[starter] += 1
        else:
            player_starters_counts[starter] = 1

    if team_data["captain"] in player_captains_counts:
        player_captains_counts[team_data["captain"]] += 1
    else:
        player_captains_counts[team_data["captain"]] = 1

# print(player_team_counts['982615'])

# print(player_owners[1156353])

sorted_player_team_counts_list = sorted(player_team_counts.items(), key=lambda x: x[1], reverse=True)
sorted_player_team_counts_df = pd.DataFrame(sorted_player_team_counts_list, columns=["player_id", "team_count"])
sorted_player_team_counts_dict = dict(zip(sorted_player_team_counts_df["player_id"], sorted_player_team_counts_df["team_count"]))
# print(sorted_player_team_counts_dict)

round_players = pd.read_csv(f'data/afcon_fantasy_market_{round}.csv')
round_players['league_own_pct'] = round_players['player_id'].map(sorted_player_team_counts_dict) / len(participants)
round_players['league_start_pct'] = round_players['player_id'].map(player_starters_counts) / len(participants)
round_players['league_cpt_pct'] = round_players['player_id'].map(player_captains_counts) / len(participants)

round_players['league_owners'] = round_players['player_id'].map(player_owners)

edited_df = round_players[['name','team','position', 'total_points', 'total_points', 'owned_percentage', 'league_own_pct', 'league_start_pct', 'league_cpt_pct', 'league_owners', 'event_start_iso_utc']]
edited_df.columns = ['Player', 'Team', 'Pos', 'Total Points', 'Round Points', 'Global Own %', 'League Own %', 'League Start %', 'League Cpt %', 'League Owners', 'Event Start Timestamp']



edited_df = edited_df.sort_values(by=['League Own %', 'League Start %', 'League Cpt %', 'Global Own %'], ascending=[False, False, False, False])

edited_df.to_csv(f'data/afcon_fantasy_market_{round}_with_league_ownership.csv', index=False)