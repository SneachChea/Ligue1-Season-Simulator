import pandas as pd

import finisher_l1.leagues as leagues


def init_teams(rank_home: pd.DataFrame, rank_away: pd.DataFrame) -> dict[str, leagues.Team]:
    """
    function that uses dataframes with the same column name as dataframes in the
    folder data_df/df_rank_* (from fbref).
    Args:
        rank_home (pd.DataFrame): DataFrame that contains information of home ranks
        rank_away (pd.DataFrame): DataFrame that contains information of away ranks
    Returns:
        dict[str, leagues.Team]: Dictionary of Team objects. Keys are team's
        name
    """
    list_name_team = rank_home.Équipe.unique()
    dico_team = {}
    for name in list_name_team:
        home_rank_team = rank_home[rank_home.Équipe == name]
        num_match_home = home_rank_team.MJ.item()
        goals_scored_home = home_rank_team.BM.item()
        goals_conceded_home = home_rank_team.BE.item()
        away_rank_team = rank_away[rank_away.Équipe == name]
        num_match_away = away_rank_team.MJ.item()
        goals_scored_away = away_rank_team.BM.item()
        goals_conceded_away = away_rank_team.BE.item()
        dico_team[name] = leagues.Team(
            num_match_home=num_match_home,
            goals_scored_home=goals_scored_home,
            goals_conceded_home=goals_conceded_home,
            name=name,
            goals_conceded_away=goals_conceded_away,
            goals_scored_away=goals_scored_away,
            num_match_away=num_match_away,
        )
    return dico_team


def init_rank(rank_home: pd.DataFrame, rank_away: pd.DataFrame) -> leagues.Rank:
    """
    Function that initialize the leagues Rank.

    Args:
        rank_home (pd.DataFrame): DataFrame that contains information of home ranks
        rank_away (pd.DataFrame): DataFrame that contains information of away ranks

    Returns:
        leagues.Rank: Rank object initialized with inputs dataFrames.
    """
    list_name_team = rank_home.Équipe.unique()
    dico_team = {}
    for name in list_name_team:
        home_points = rank_home[rank_home.Équipe == name].Pts.item()
        away_points = rank_away[rank_away.Équipe == name].Pts.item()
        dico_team[name] = home_points + away_points
    return leagues.Rank(dico_team)
