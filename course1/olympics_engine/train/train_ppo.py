# here lies an example of how to train an RL agent on individual subgame


import argparse
import datetime

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import os
from pathlib import Path
import sys
base_dir = str(Path(__file__).resolve().parent.parent.parent)
# print(base_dir)
sys.path.append(base_dir)
engine_path = os.path.join(base_dir, "olympics_engine")
sys.path.append(engine_path)


from collections import deque, namedtuple
import random

from olympics_engine.generator import create_scenario
from env_wrapper.chooseenv import make
from train.log_path import *
from train.algo.ppo import PPO
from train.algo.random import random_agent

from olympics_engine.scenario import table_hockey, football, wrestling, Running_competition
from olympics_engine.agent import *


parser = argparse.ArgumentParser()
parser.add_argument('--game_name', default="running-competition", type=str, help='running-competition/table-hockey/football/wrestling')
parser.add_argument('--algo', default="ppo", type=str, help="ppo/sac")
parser.add_argument('--max_episodes', default=100000, type=int)
parser.add_argument('--episode_length', default=500, type=int)

parser.add_argument('--seed', default=1, type=int)

parser.add_argument("--save_interval", default=1000, type=int)
parser.add_argument("--model_episode", default=0, type=int)

parser.add_argument("--load_model", action='store_true')
parser.add_argument("--load_run", default=2, type=int)
parser.add_argument("--load_episode", default=900, type=int)


device = 'cuda'
RENDER = True
actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}           #dicretise action space


def main(args):
    num_agents = 2
    ctrl_agent_index = 0        #controlled agent index


    if args.game_name == 'running-competition':
        map_id = random.randint(1,4)
        map_id = 1
        Gamemap = create_scenario(args.game_name)
        env = Running_competition(meta_map=Gamemap,map_id=map_id, vis = 200, vis_clear=5, agent1_color = 'light red',
                                   agent2_color = 'blue')
        env.max_step = 400
    elif args.game_name == 'table-hockey':
        Gamemap = create_scenario(args.game_name)
        env = table_hockey(Gamemap)
        env.max_step = 400
    elif args.game_name == 'football':
        Gamemap = create_scenario(args.game_name)
        env = football(Gamemap)
        env.max_step = 400
    elif args.game_name == 'wrestling':
        Gamemap = create_scenario(args.game_name)
        env = wrestling(Gamemap)
        env.max_step = 400
    else:
        raise NotImplementedError


    print(f'Playing game {args.game_name}')
    print("==algo: ", args.algo)
    print(f'device: {device}')
    print(f'model episode: {args.model_episode}')
    print(f'save interval: {args.save_interval}')

    print(f'Total agent number: {num_agents}')
    print(f'Agent control by the actor: {ctrl_agent_index}')


    width = env.view_setting['width']+2*env.view_setting['edge']
    height = env.view_setting['height']+2*env.view_setting['edge']
    print(f'Game board width: {width}')
    print(f'Game board height: {height}')

    obs_dim = 40*40
    print(f'observation dimension: {obs_dim}')

    torch.manual_seed(args.seed)
    # 定义保存路径
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    if not args.load_model:
        writer = SummaryWriter(os.path.join(str(log_dir), "{}_{} on subgames {}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),args.algo, args.game_name)))
        save_config(args, log_dir)

    record_win = deque(maxlen=100)
    record_win_op = deque(maxlen=100)

    if args.load_model:
        model = PPO()
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_run))
        model.load(load_dir,episode=args.load_episode)
    else:
        model = PPO(run_dir)
        Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'done'])

    opponent_agent = random_agent()     #we use random opponent agent here

    episode = 0
    train_count = 0

    while episode < args.max_episodes:
        state = env.reset()
        if RENDER:
            env.render()
        if isinstance(state[ctrl_agent_index], type({})):
            obs_ctrl_agent, energy_ctrl_agent = state[ctrl_agent_index]['agent_obs'].flatten(), env.agent_list[ctrl_agent_index].energy
            obs_oppo_agent, energy_oppo_agent = state[1-ctrl_agent_index]['agent_obs'], env.agent_list[1-ctrl_agent_index].energy
        else:
            obs_ctrl_agent, energy_ctrl_agent = state[ctrl_agent_index].flatten(), env.agent_list[ctrl_agent_index].energy
            obs_oppo_agent, energy_oppo_agent = state[1-ctrl_agent_index], env.agent_list[1-ctrl_agent_index].energy

        episode += 1
        step = 0
        Gt = 0

        while True:
            action_opponent = opponent_agent.act(obs_oppo_agent)        #opponent action
            action_opponent = [0,0]  #here we assume the opponent is not moving in the demo

            action_ctrl_raw, action_prob= model.select_action(obs_ctrl_agent, False if args.load_model else True)
            #inference
            action_ctrl = actions_map[action_ctrl_raw]

            action = [action_opponent, action_ctrl] if ctrl_agent_index == 1 else [action_ctrl, action_opponent]

            next_state, reward, done, _ = env.step(action)

            if isinstance(next_state[ctrl_agent_index], type({})):
                next_obs_ctrl_agent, next_energy_ctrl_agent = next_state[ctrl_agent_index]['agent_obs'].flatten(), env.agent_list[ctrl_agent_index].energy
                next_obs_oppo_agent, next_energy_oppo_agent = next_state[1-ctrl_agent_index]['agent_obs'], env.agent_list[1-ctrl_agent_index].energy
            else:
                next_obs_ctrl_agent, next_energy_ctrl_agent = next_state[ctrl_agent_index], env.agent_list[ctrl_agent_index].energy
                next_obs_oppo_agent, next_energy_oppo_agent = next_state[1-ctrl_agent_index], env.agent_list[1-ctrl_agent_index].energy

            step += 1

            if not done:
                post_reward = [0, 0]
            else:
                if reward[0] != reward[1]:
                    post_reward = [reward[0]-100, reward[1]] if reward[0]<reward[1] else [reward[0], reward[1]-100]
                else:
                    post_reward=[0, 0]

            if not args.load_model:
                trans = Transition(obs_ctrl_agent, action_ctrl_raw, action_prob, post_reward[ctrl_agent_index],
                                   next_obs_ctrl_agent, done)
                model.store_transition(trans)

            obs_oppo_agent, energy_oppo_agent = next_obs_oppo_agent, next_energy_oppo_agent
            obs_ctrl_agent, energy_ctrl_agent = np.array(next_obs_ctrl_agent).flatten(), next_energy_ctrl_agent

            if RENDER:
                env.render()
            Gt += reward[ctrl_agent_index] if done else -1

            if done:
                win_is = 1 if reward[ctrl_agent_index]>reward[1-ctrl_agent_index] else 0
                win_is_op = 1 if reward[ctrl_agent_index]<reward[1-ctrl_agent_index] else 0
                record_win.append(win_is)
                record_win_op.append(win_is_op)
                print("Episode: ", episode, "controlled agent: ", ctrl_agent_index, "; Episode Return: ", Gt,
                      "; win rate(controlled & opponent): ", '%.2f' % (sum(record_win)/len(record_win)),
                      '%.2f' % (sum(record_win_op)/len(record_win_op)), '; Trained episode:', train_count)

                if not args.load_model:
                    if args.algo == 'ppo' and len(model.buffer) >= model.batch_size:
                        if win_is == 1:
                            model.update(episode)
                            train_count += 1
                        else:
                            model.clear_buffer()

                    writer.add_scalar('training Gt', Gt, episode)

                break
        if episode % args.save_interval == 0 and not args.load_model:
            model.save(run_dir, episode)





if __name__ == '__main__':
    args = parser.parse_args()
    main(args)