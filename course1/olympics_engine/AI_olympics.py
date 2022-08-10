from scenario import Running_competition, table_hockey, football, wrestling, curling_competition, billiard_joint
import sys
from pathlib import Path
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)
from olympics_engine.generator import create_scenario

import random


class AI_Olympics:
    def __init__(self, random_selection, minimap):

        self.random_selection = True
        self.minimap_mode = minimap

        self.max_step = 400
        self.vis = 200
        self.vis_clear = 5

        running_Gamemap = create_scenario("running-competition")
        self.running_game = Running_competition(running_Gamemap, vis = 200, vis_clear=5, agent1_color = 'light red', agent2_color='blue')

        self.tablehockey_game = table_hockey(create_scenario("table-hockey"))
        self.football_game = football(create_scenario('football'))
        self.wrestling_game = wrestling(create_scenario('wrestling'))
        self.curling_game = curling_competition(create_scenario('curling-IJACA-competition'))
        self.billiard_game = billiard_joint(create_scenario("billiard-joint"))

        self.running_game.max_step = self.max_step
        self.tablehockey_game.max_step = self.max_step
        self.football_game.max_step = self.max_step
        self.wrestling_game.max_step = self.max_step
        # self.curling_game.max_step =

        self.game_pool = [{"name": 'running-competition', 'game': self.running_game},
                          {"name": 'table-hockey', "game": self.tablehockey_game},
                          {"name": 'football', "game": self.football_game},
                          {"name": 'wrestling', "game": self.wrestling_game},
                          {"name": "curling", "game": self.curling_game},
                          {"name": "billiard", "game": self.billiard_game}]
        self.view_setting = self.running_game.view_setting

    def reset(self):

        self.done = False
        selected_game_idx_pool = list(range(len(self.game_pool)))
        if self.random_selection:
            random.shuffle(selected_game_idx_pool)            #random game playing sequence

        self.selected_game_idx_pool = selected_game_idx_pool                           #fix game playing sequence
        self.current_game_count = 0
        selected_game_idx = self.selected_game_idx_pool[self.current_game_count]


        print(f'Playing {self.game_pool[selected_game_idx]["name"]}')
        # if self.game_pool[selected_game_idx]['name'] == 'running-competition':
        #     self.game_pool[selected_game_idx]['game'] = \
        #         Running_competition.reset_map(meta_map= self.running_game.meta_map,map_id=None, vis=200, vis_clear=5,
        #                                       agent1_color = 'light red', agent2_color = 'blue')     #random sample a map
        #     self.game_pool[selected_game_idx]['game'].max_step = self.max_step

        self.current_game = self.game_pool[selected_game_idx]['game']
        self.game_score = [0,0]

        init_obs = self.current_game.reset()
        if self.current_game.game_name == 'running-competition':
            init_obs = [{'agent_obs': init_obs[i], 'id': f'team_{i}'} for i in [0,1]]
        for i in init_obs:
            i['game_mode'] = 'NEW GAME'

        for i,j in enumerate(init_obs):
            if 'curling' in self.current_game.game_name:
                j['energy'] = 1000
            else:
                j['energy'] = self.current_game.agent_list[i].energy

        return init_obs

    def step(self, action_list):

        obs, reward, done, _ = self.current_game.step(action_list)

        if self.current_game.game_name == 'running-competition':
            obs = [{'agent_obs': obs[i], 'id': f'team_{i}'} for i in [0,1]]
        for i in obs:
            i['game_mode'] = ''

        for i,j in enumerate(obs):
            if 'curling' in self.current_game.game_name:
                j['energy'] = 1000
            elif 'billiard' in self.current_game.game_name:
                j['energy'] = self.current_game.agent_energy[i]
            else:
                j['energy'] = self.current_game.agent_list[i].energy

        if done:
            winner = self.current_game.check_win()
            if winner != '-1':
                self.game_score[int(winner)] += 1

            if self.current_game_count == len(self.game_pool)-1:
                self.done = True
            else:
                # self.current_game_idx += 1
                self.current_game_count += 1
                self.current_game_idx = self.selected_game_idx_pool[self.current_game_count]

                self.current_game = self.game_pool[self.current_game_idx]['game']
                print(f'Playing {self.game_pool[self.current_game_idx]["name"]}')
                obs = self.current_game.reset()
                if self.current_game.game_name == 'running-competition':
                    obs = [{'agent_obs': obs[i], 'id': f'team_{i}'} for i in [0,1]]
                for i in obs:
                    i['game_mode'] = 'NEW GAME'
                for i,j in enumerate(obs):
                    if 'curling' in self.current_game.game_name:
                        j['energy'] = 1000
                    else:
                        j['energy'] = self.current_game.agent_list[i].energy

        if self.done:
            print('game score = ', self.game_score)
            if self.game_score[0] > self.game_score[1]:
                self.final_reward = [100, 0]
                print('Results: team 0 win!')
            elif self.game_score[1] > self.game_score[0]:
                self.final_reward = [0, 100]
                print('Results: team 1 win!')
            else:
                self.final_reward = [0,0]
                print('Results: Draw!')

            return obs, self.final_reward, self.done, ''
        else:
            return obs, reward, self.done, ''

    def is_terminal(self):
        return self.done

    def __getattr__(self, item):
        return getattr(self.current_game, item)


    def render(self):
        self.current_game.render()



