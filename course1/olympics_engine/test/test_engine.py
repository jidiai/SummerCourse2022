import sys
from pathlib import Path
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)
print(base_path)


from core import OlympicsBase
from object import *

import time

gamemap = {'objects':[], 'agents':[]}

#rectangular
#gamemap['objects'].append(Wall(init_pos=[[50, 200], [90, 600]], length = None, color = 'black'))
#gamemap['objects'].append(Wall(init_pos=[[50, 200], [700, 150]], length = None, color = 'black'))
#gamemap['objects'].append(Wall(init_pos = [[700, 150], [715,650]], length = None, color = 'black'))
#gamemap['objects'].append(Wall(init_pos=[[90, 600], [715, 650]], length = None, color = 'black'))

#triangle
#gamemap['objects'].append(Wall(init_pos=[[50, 200], [90, 600]], length = None, color = 'black'))
#gamemap['objects'].append(Wall(init_pos=[[50, 200], [700, 400]], length = None, color = 'black'))
#gamemap['objects'].append(Wall(init_pos=[[700, 400], [90, 600]], length = None, color = 'black'))

#triangle obstacle
#gamemap['objects'].append(Wall(init_pos=[[200, 350], [200, 400]], length = None, color = 'black'))
#gamemap['objects'].append(Wall(init_pos=[[200, 350], [250, 400]], length = None, color = 'black'))
#gamemap['objects'].append(Wall(init_pos=[[200, 400], [250, 400]], length = None, color = 'black'))


#arc
#gamemap['objects'].append(Arc(init_pos = [50, 50, 900, 900], start_radian = 0, end_radian = -90, passable = False, color = 'black'))
#gamemap['objects'].append(Arc(init_pos = [250, 250, 500, 500], start_radian = 0, end_radian = -90, passable = False, color = 'black'))

#gamemap['objects'].append(Wall(init_pos = [[950, 500], [750, 500]]))
#gamemap['objects'].append(Wall(init_pos = [[500, 750], [950, 750]]))
#gamemap['objects'].append(Wall(init_pos = [[950, 950],[500, 950]]))
#gamemap['objects'].append(Wall(init_pos = [[950, 750],[950,950]]))

#gamemap['agents'].append(Agent(position=[500, 800], r = 30, mass = 1))


#gamemap['objects'].append(Arc(init_pos=[200, 200, 450,450], start_radian= 0, end_radian=-0.00000001, passable = False, color = 'black'))

#gamemap['objects'].append(Wall(init_pos=[[700, 100], [700,700]], color = 'black'))
#gamemap['objects'].append(Wall(init_pos=[[700, 100], [100,100]], color = 'black'))
#gamemap['objects'].append(Wall(init_pos=[[100, 100], [100,700]], color = 'black'))
#gamemap['objects'].append(Wall(init_pos=[[100, 700], [700,700]], color = 'black'))

#arc-running
#gamemap['objects'].append(Arc(init_pos = [100, 100, 700, 700], start_radian = -90, end_radian = 90, passable=False, color = 'black'))
#gamemap['objects'].append(Arc(init_pos = [300,300, 300, 300], start_radian = -90, end_radian=90, passable=False, color = 'black'))
#gamemap['objects'].append(Arc(init_pos = [200, 200, 500, 500], start_radian= -90, end_radian = 90, passable=True, color = 'grey'))

#gamemap['objects'].append(Wall(init_pos=[[20, 100], [470, 100]], color = 'black'))
#gamemap['objects'].append(Wall(init_pos=[[20, 300], [470, 300]]))
#gamemap['objects'].append(Wall(init_pos=[[20,600],[460, 600]]))
#gamemap['objects'].append(Wall(init_pos=[[20, 800], [480, 800]]))
#gamemap['objects'].append(Cross(init_pos=[[20,700], [460, 700]], color = 'grey'))
#gamemap['objects'].append(Cross(init_pos=[[20,200],[460, 200]], color = 'grey'))

#gamemap['objects'].append(Wall(init_pos=[[20,100],[20,300]]))
#gamemap['objects'].append(Wall(init_pos=[[20,600],[20,800]]))
#gamemap['objects'].append(Cross(init_pos = [[50,100], [50,300]], color = 'red'))


#table hockey
#gamemap['objects'].append(Wall(init_pos=[[20,200], [20,600]], color = 'black'))
#gamemap['objects'].append(Wall(init_pos=[[20,200], [780,200]], color = 'black'))

#gamemap['objects'].append(Wall(init_pos=[[780,200], [780, 600]], color = 'black'))
#gamemap['objects'].append(Wall(init_pos=[[20, 600], [780,600]], color = 'black'))
#gamemap['objects'].append(Wall(init_pos= [[400, 200], [400, 600]], color = 'grey', ball_can_pass=True))

#gamemap['objects'].append(Cross(init_pos = [[780, 300],[780, 500]], width = 4, color = 'red'))
#gamemap['objects'].append(Cross(init_pos = [[20, 300],[20, 500]], width = 4, color = 'red'))



#gamemap['agents'].append(Agent(position=[80, 401], r = 30, mass = 1))
#gamemap['agents'].append(Agent(position = [500, 400], r= 30, mass= 1))
#gamemap['agents'].append(Ball(position=[80, 500], r = 20, mass = 1))
#gamemap['agents'].append(Agent(position=[80, 750], r = 30, mass = 1))

#print(gamemap)
#gamemap['objects'].append(Wall(init_pos=[[20,20],[120,20]]))
#gamemap['objects'].append(Wall(init_pos=[[20,20],[20,220]]))
#gamemap['objects'].append(Wall(init_pos=[[120,20],[120,120]]))
#gamemap['objects'].append(Wall(init_pos=[[120,120],[620,120]]))
#gamemap['objects'].append(Wall(init_pos=[[20,220],[520,220]]))
#gamemap['objects'].append(Wall(init_pos=[[520, 220],[520, 620]]))
#gamemap['objects'].append(Wall(init_pos=[[620, 120],[620, 720]]))
#gamemap['objects'].append(Wall(init_pos=[[520,620],[20,620]]))
#gamemap['objects'].append(Wall(init_pos=[[620, 720],[20,720]]))
#gamemap['objects'].append(Wall(init_pos=[[20, 720], [20,620]]))


#gamemap['agents'].append(Agent(position=[60, 60], r = 30, mass = 1))



#test
#gamemap['objects'].append(Wall(init_pos = [[100,100], [300,100]]))
#gamemap['objects'].append(Wall(init_pos = [[100,100], [100,200]]))
#gamemap['objects'].append(Wall(init_pos = [[100,200], [200,200]]))
#gamemap['objects'].append(Wall(init_pos = [[200,200], [200,300]]))
#gamemap['objects'].append(Wall(init_pos = [[200,300], [300,300]]))
#gamemap['objects'].append(Wall(init_pos = [[300,300], [300,100]]))

#gamemap['objects'].append(Wall(init_pos = [[200,100], [350,100]]))
#gamemap['objects'].append(Arc(init_pos = [300,100,100,100], start_radian = 0, end_radian = 90, color = 'black', passable = False))
#gamemap['objects'].append(Wall(init_pos = [[400,150], [400,300]]))#\

#gamemap['objects'].append(Wall(init_pos = [[200, 200], [250, 200]]))
#gamemap['objects'].append(Arc(init_pos = [200, 200, 100, 100],start_radian = 0, end_radian = 90, color = 'black', passable = False))
#gamemap['objects'].append(Wall(init_pos = [[300, 250], [300, 300]]))

#gamemap['objects'].append(Wall(init_pos = [[200,100], [200,200]]))
#gamemap['objects'].append(Wall(init_pos = [[300,300], [400,300]]))

#gamemap['objects'].append(Wall(init_pos = [[200,100], [200,300]]))
#gamemap['objects'].append(Wall(init_pos = [[100,300], [400,300]]))
#gamemap['objects'].append(Arc(init_pos = [200, 200, 100, 100], start_radian = 0, end_radian = 90, color = 'black', passable = False))
#gamemap['objects'].append(Wall(init_pos = [[200,200],[250,200]]))
#gamemap['objects'].append(Wall(init_pos = [[300, 250],[300,400]]))
#gamemap['objects'].append(Wall(init_pos = [[400, 300], [300,300]]))

#gamemap['objects'].append(Wall(init_pos= [[200,500], [400,500]]))
#gamemap['objects'].append(Wall(init_pos= [[400,500], [800,500]]))



       # map5
# gamemap['objects'].append(Arc(init_pos=[87.5,237.5,225,225], start_radian=90, end_radian=180, passable=False, color='black', collision_mode = 0))
# gamemap['objects'].append(Arc(init_pos=[87.5,237.5,225,225], start_radian=0, end_radian=90, passable=False, color='black', collision_mode=0))
#
# gamemap['objects'].append(Arc(init_pos=[162.5,312.5,75,75], start_radian=90, end_radian=180, passable=False, color='black', collision_mode=0))
# gamemap['objects'].append(Arc(init_pos=[162.5,312.5,75,75], start_radian=0, end_radian=90, passable=False, color='black', collision_mode = 0))
# #
# gamemap['objects'].append(Arc(init_pos=[387.5,237.5,225,225], start_radian=90, end_radian=180, passable=False, color='black', collision_mode=0))
# gamemap['objects'].append(Arc(init_pos=[387.5,237.5,225,225], start_radian=0, end_radian=90, passable=False, color='black', collision_mode=0))
# #
# gamemap['objects'].append(Arc(init_pos=[462.5,312.5,75,75], start_radian=90, end_radian=180, passable=False, color='black', collision_mode=0))
# gamemap['objects'].append(Arc(init_pos=[462.5,312.5,75,75], start_radian=0, end_radian=90, passable=False, color='black', collision_mode=0))
# #
# gamemap['objects'].append(Arc(init_pos=[-62.5,237.5,225,225], start_radian=-90, end_radian=0, passable=False, color='black', collision_mode=1))
#
# gamemap['objects'].append(Arc(init_pos=[12.5,312.5,75,75], start_radian=-90, end_radian=0, passable=False, color='black', collision_mode=1))
#
# gamemap['objects'].append(Arc(init_pos=[237.5,237.5,225,225], start_radian=-180, end_radian=-90, passable=False, color='black', collision_mode=1))
# gamemap['objects'].append(Arc(init_pos=[237.5,237.5,225,225], start_radian=-90, end_radian=0, passable=False, color='black', collision_mode=1))
# #
# #
# gamemap['objects'].append(Arc(init_pos=[312.5,312.5,75,75], start_radian=-180, end_radian=-90, passable=False, color='black', collision_mode=1))
# gamemap['objects'].append(Arc(init_pos=[312.5,312.5,75,75], start_radian=-90, end_radian=0, passable=False, color='black', collision_mode = 1))
# #
# gamemap['objects'].append(Arc(init_pos=[537.5,237.5,225,225], start_radian=-180, end_radian=-90, passable=False, color='black', collision_mode=1))
# gamemap['objects'].append(Arc(init_pos=[612.5,312.5,75,75], start_radian=-180, end_radian=-90, passable=False, color='red', collision_mode = 1))
#
# gamemap['objects'].append(Wall(init_pos=[[650,387.5],[650,462.5]], color='red'))
# gamemap['objects'].append(Wall(init_pos=[[50,387.5],[50,462.5]], color='black'))
#
# gamemap['objects'].append(Arc(init_pos=[100,100,100,100], passable=False, start_radian=0, end_radian=90, color='black', collision_mode=1))

# gamemap['objects'].append(Wall(init_pos=[[200, 150],[200,300]]))
# gamemap['objects'].append(Wall(init_pos=[[200, 150],[300,150]]))
#
# gamemap['agents'].append(Agent(position = [100, 150], mass = 1, r = 10, color = 'green'))

# gamemap['objects'].append(Arc(init_pos = [275, 275, 150,150], start_radian=0, end_radian=-90, passable=False, color='black', collision_mode=3))
# gamemap['objects'].append(Wall(init_pos = [[350,425],[650,425]]))
#
# gamemap['agents'].append(Agent(position = [612.5,462.5], mass = 1, r = 18.75))
#
# boxing
# gamemap['objects'].append(Arc(init_pos = [100,150, 400, 400], start_radian = 90, end_radian = -90,
#                               passable=True, color = 'red', collision_mode = 3))
# gamemap['objects'].append(Arc(init_pos = [100,150, 400, 400], start_radian = -90, end_radian = 90,
#                               passable=True, color = 'red', collision_mode = 3))
#
#
# gamemap['agents'].append(Agent(position = [300, 200], mass = 1, r = 20, color = 'purple'))
# gamemap['agents'].append(Agent(position = [300, 500], mass = 1, r = 20, color='green'))

gamemap['objects'].append(Wall(init_pos = [[200, 10], [400, 10]], color = 'black'))
gamemap['objects'].append(Wall(init_pos = [[200, 10], [100, 200]], color = 'black'))
gamemap['objects'].append(Wall(init_pos = [[400, 10], [500, 200]], color = 'black'))
gamemap['objects'].append(Wall(init_pos = [[500, 200], [100, 200]], color = 'black'))








gamemap['agents'].append(Agent(position = [300,26], mass=1, r=15, color='purple'))
gamemap['agents'].append(Agent(position = [300,60], mass=1, r=15, color='purple'))
gamemap['agents'].append(Agent(position = [300,100], mass=1, r=15, color='purple'))
gamemap['agents'].append(Agent(position = [300,140], mass=1, r=15, color='purple'))
gamemap['agents'].append(Agent(position = [220,26], mass=1, r=15, color='purple'))
gamemap['agents'].append(Agent(position = [200,60], mass=1, r=15, color='purple'))
gamemap['agents'].append(Agent(position = [200,100], mass=1, r=15, color='purple'))
gamemap['agents'].append(Agent(position = [200,140], mass=1, r=15, color='purple'))
gamemap['agents'].append(Agent(position = [200,180], mass=1, r=15, color='purple'))
gamemap['agents'].append(Agent(position = [300,140], mass=1, r=15, color='purple'))
gamemap['agents'].append(Agent(position = [300,180], mass=1, r=15, color='purple'))





# gamemap['objects'].append(Wall(init_pos = [[100,100], [100, 200]]))
# gamemap['objects'].append(Wall(init_pos = [[100,200], [100, 300]], color = 'red'))
#
# gamemap['agents'].append(Agent(position = [200,200], r = 15))



#gamemap['agents'].append(Agent(position = [200, 289], mass = 1, r = 10))
#gamemap['agents'].append(Agent(position = [280, 280], mass = 1, r = 10))


#gamemap['agents'].append(Agent(position = [389.9, 200], mass = 1, r = 10))

#gamemap['agents'].append(Agent(position = [200, 469.9], mass = 1, r = 30))


#gamemap['agents'].append(Agent(position = [300, 150], mass = 1, r = 20))

gamemap['view'] = {'width': 600, 'height':600, 'edge': 50}



class env_test(OlympicsBase):
    def __init__(self, map=gamemap):
        super(env_test, self).__init__(map)

        self.gamma = 1  # v衰减系数
        self.restitution = 1
        self.print_log = True
        self.tau = 0.1

        self.draw_obs = False
        self.show_traj = False


    def check_overlap(self):
        pass

    def check_action(self, action_list):
        action = []
        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].type == 'agent':
                action.append(action_list[0])
                _ = action_list.pop(0)
            else:
                action.append(None)

        return action

    def step(self, actions_list):

        actions_list = self.check_action(actions_list)

        self.stepPhysics(actions_list, self.step_cnt)


        self.step_cnt += 1
        step_reward = 1 #self.get_reward()
        obs_next = 1#self.get_obs()
        # obs_next = 1
        done = False #self.is_terminal()

        #check overlapping
        #self.check_overlap()

        #return self.agent_pos, self.agent_v, self.agent_accel, self.agent_theta, obs_next, step_reward, done
        return obs_next, step_reward, done, ''

import random

env = env_test()


for _ in range(100):

    env.reset()
    env.render()
    # env.agent_theta[0][0] = 180
    done = False
    step = 0
    while not done:
        print('\n step = ', step)
        #if step < 10:
        #    action = [[random.randint(-100,200),random.randint(-30, 30)]]#, [2,1]]#, [2,2]]#, [2,1]]#[[2,1], [2,1]] + [ None for _ in range(4)]
        #else:
        #    action = [[random.randint(-100,200),random.randint(-30, 30)]]#, [2,1]]#, [2,1]]#, [2,random.randint(0,2)]] #[[2,1], [2,1]] + [None for _ in range(4)]
        #action1 = [random.randint(-100, 200), random.randint(-30, 30)]
        #action2 = [random.randint(-100, 200), random.randint(-30, 30)]
        action1 = [100+random.uniform(0,1)*10, 0]
        action2 = [100+random.uniform(0,1)*15, 0]
        action = [[200, 0] for _ in range(20)]
        _,_,done, _ = env.step(action)

        print('agent v = ', env.agent_v)
        env.render()
        step += 1

        if step < 60:
            time.sleep(0.05)
        else:
            time.sleep(0.05)






