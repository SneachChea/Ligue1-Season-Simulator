import collections
import dataclasses

import numpy as np
import pandas as pd
import scipy.stats as stats

from finisher_l1.utils import get_probability


@dataclasses.dataclass
class Rank:
    fixed_points: dict[str, int]
    added_points: dict[str, list] = dataclasses.field(
        default_factory=dict, init=False
    )
    num_iter: int = dataclasses.field(default=0, init=False)

    def __post_init__(self):
        self.added_points = collections.defaultdict(list)

    def update_rank(self, new_dico_points: dict[str, int]):
        for k in self.fixed_points.keys():
            self.added_points[k].append(new_dico_points[k])
        self.num_iter += 1

    def final_rank(self):
        final_points = collections.defaultdict(float)
        for k in self.fixed_points.keys():
            final_points[k] = (
                np.mean(self.added_points[k]) + self.fixed_points[k]
            )
        sorted_dict = {
            k: v
            for k, v in sorted(
                final_points.items(), key=lambda item: item[1], reverse=True
            )
        }
        return sorted_dict

    def std_final_rank(self):
        final_points = collections.defaultdict(float)
        for k in self.fixed_points.keys():
            final_points[k] = np.std(self.added_points[k])
        return final_points


@dataclasses.dataclass
class Team:
    num_match_home: int
    goals_scored_home: int
    goals_conceded_home: int
    num_match_away: int
    goals_scored_away: int
    goals_conceded_away: int
    name: str

    def update_home(self, goals_scored, goals_conceded):
        self.goals_scored_home += goals_scored
        self.goals_conceded_home += goals_conceded
        self.num_match_home += 1

    def update_away(self, goals_scored, goals_conceded):
        self.goals_scored_away += goals_scored
        self.goals_conceded_away += goals_conceded
        self.num_match_away += 1

    @property
    def mean_goals_scored_home(self):
        return self.goals_scored_home / self.num_match_home

    @property
    def mean_goals_scored_away(self):
        return self.goals_scored_away / self.num_match_away


@dataclasses.dataclass
class League:
    dico_team: dict[str, Team]
    dico_points: dict[str, int] = dataclasses.field(
        default_factory=dict, init=False
    )
    per_team_goals_conceded_home: float
    per_team_goals_conceded_away: float

    def __post_init__(self):
        self.dico_points = collections.defaultdict(int)

    def get_schedule(
        self, schedule: pd.DataFrame, week: int, unplayed_games: bool = True
    ):
        if unplayed_games:
            condition = (schedule["Sem."] == week) & (
                schedule["Notes"] == "Match annulé"
            )
        else:
            condition = schedule["Sem."] == week
        home_teams = schedule[condition].Domicile.to_list()
        away_teams = schedule[condition].Extérieur.to_list()
        return home_teams, away_teams

    def results_simulated_match(self, home_team_name: str, away_team_name: str):
        lambda_home = (
            self.dico_team[home_team_name].mean_goals_scored_home
            * self.dico_team[away_team_name].goals_conceded_away
            / self.per_team_goals_conceded_away
        )
        lambda_away = (
            self.dico_team[away_team_name].mean_goals_scored_away
            * self.dico_team[home_team_name].goals_conceded_home
            / self.per_team_goals_conceded_home
        )
        goals_scored_home_team = stats.poisson.rvs(lambda_home)
        goals_scored_away_team = stats.poisson.rvs(lambda_away)
        return goals_scored_home_team, goals_scored_away_team

    def probabilities_match(
        self, home_team_name: str, away_team_name: str, num_max_goals: int
    ):
        lambda_home = (
            self.dico_team[home_team_name].mean_goals_scored_home
            * self.dico_team[away_team_name].goals_conceded_away
            / self.per_team_goals_conceded_away
        )
        lambda_away = (
            self.dico_team[away_team_name].mean_goals_scored_away
            * self.dico_team[home_team_name].goals_conceded_home
            / self.per_team_goals_conceded_home
        )
        return stats.poisson.pmf(
            range(num_max_goals), mu=lambda_home
        ), stats.poisson.pmf(range(num_max_goals), mu=lambda_away)

    def attribute_points(
        self, goals_home: int, goals_away: int, home_team: str, away_team: str
    ):
        if goals_home > goals_away:
            self.dico_points[home_team] += 3
        elif goals_home == goals_away:
            self.dico_points[home_team] += 1
            self.dico_points[away_team] += 1
        else:
            self.dico_points[away_team] += 3

    def update_team_ranks(
        self, goals_home: int, goals_away: int, home_team: str, away_team: str
    ):
        self.dico_team[home_team].update_home(goals_home, goals_away)
        self.dico_team[away_team].update_away(goals_away, goals_home)

    def simulate_week(self, schedule: pd.DataFrame, week: int) -> None:
        home_teams, away_teams = self.get_schedule(schedule=schedule, week=week)
        tmp_goals_conceded_home = 0.0
        tmp_goals_conceded_away = 0.0
        for idx, home_team in enumerate(home_teams):
            away_team = away_teams[idx]
            (
                goals_scored_home_team,
                goals_scored_away_team,
            ) = self.results_simulated_match(
                home_team_name=home_team, away_team_name=away_team
            )
            self.attribute_points(
                goals_home=goals_scored_home_team,
                goals_away=goals_scored_away_team,
                home_team=home_team,
                away_team=away_team,
            )
            self.update_team_ranks(
                goals_home=goals_scored_home_team,
                goals_away=goals_scored_home_team,
                home_team=home_team,
                away_team=away_team,
            )
            tmp_goals_conceded_home += goals_scored_away_team
            tmp_goals_conceded_away += goals_scored_home_team
        self.per_team_goals_conceded_away += tmp_goals_conceded_away / 20
        self.per_team_goals_conceded_home += tmp_goals_conceded_home / 20

    def MPMA_week(self, schedule: pd.DataFrame, week: int) -> dict[str, float]:
        home_teams, away_teams = self.get_schedule(schedule=schedule, week=week)
        mpma_dict = {}
        for idx, home_team in enumerate(home_teams):
            away_team = away_teams[idx]
            goal_home, goal_away = self.probabilities_match(
                home_team_name=home_team,
                away_team_name=away_team,
                num_max_goals=12,
            )
            matrix = np.outer(goal_home, goal_away)
            prob_home, _, prob_away = get_probability(matrix)
            mpma_dict[home_team] = prob_away
            mpma_dict[away_team] = prob_home
        return mpma_dict

    def output_league(self) -> dict[str, int]:
        return self.dico_points
