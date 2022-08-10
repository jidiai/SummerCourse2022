import pygame


from olympics_engine.tools.settings import *

grid_node_width = 2     #for view drawing
grid_node_height = 2


class Viewer():
    def __init__(self, setting):
        pygame.init()
        width = setting["width"]
        height = setting["height"]
        edge = setting["edge"]
        self.WIN_SIZE = width+2*edge, height+2*edge
        # self.background = pygame.display.set_mode(self.WIN_SIZE)
        #
        # self.draw_background()
        self.color_list = [ [255, 0, 0], [0, 255, 0], [0,0,255]  , [0,0,0], [160, 32, 240]]

        self.screen_list = []

        # WIN_SIZE = 1000, 1000
    def set_mode(self):
        self.background = pygame.display.set_mode(self.WIN_SIZE)

    def set_screen(self, size, color, pos):
        tmp_screen = pygame.Surface(size)
        tmp_screen.fill(color)
        self.screen_list.append({'screen':tmp_screen, "pos": pos})

    def draw_background(self, color_code=(255,255,255)):
        self.background.fill(color_code)

    def draw_ball(self, pos_list, agent_list):
        # self.background.fill((255, 255, 255))
        assert len(pos_list) == len(agent_list)
        for i in range(len(pos_list)):
            t = pos_list[i]
            r = agent_list[i].r
            color = agent_list[i].color
            #print('color in viewer', color)
            pygame.draw.circle(self.background, COLORS[color], t, r, 0)
            pygame.draw.circle(self.background, COLORS['black'], t, 2, 2)


    def draw_direction(self, pos_list, a_list):
        """
        :param pos_list: position of circle center
        :param a_list: acceleration of circle
        :return:
        """
        assert len(pos_list) == len(a_list)
        for i in range(len(pos_list)):
            a_x, a_y = a_list[i]
            if a_x != 0 or a_y != 0:
                t = pos_list[i]
                start_x, start_y = t
                end_x = start_x + a_x/5
                end_y = start_y + a_y/5

                pygame.draw.line(self.background, color = [0,0,0], start_pos=[start_x, start_y], end_pos = [end_x, end_y], width = 2)


    def draw_map(self, object):
        # (left, top), width, height
        #pygame.draw.rect(self.background, [0, 0, 0], [0, 200, 800, 400], 2) # black
        #pygame.draw.rect(self.background, [255, 0, 0], [700, 200, 2, 400], 1) # red
        #print("check color: ", object.color)
        if object.type == 'arc':
            pygame.draw.arc(self.background, COLORS[object.color], object.init_pos, object.start_radian, object.end_radian, object.width)
            # pygame.draw.arc(self.background, COLORS[object.color], object.init_pos, object.start_radian-0.02, object.end_radian+0.02, object.width)

        else:
            s, e = object.init_pos
            pygame.draw.line(surface = self.background, color = COLORS[object.color], start_pos = s, end_pos = e, width = object.width)


    def draw_trajectory(self, trajectory_list, agent_list):
        for i in range(len(trajectory_list)):
            for t in trajectory_list[i]:
                pygame.draw.circle(self.background, COLORS[agent_list[i].color], t, 2, 1)

    def draw_obs(self, points, agent_list):
        if len(points) >= len(agent_list):
            for b in range(len(agent_list)):
                if points[b] is not None:
                    pygame.draw.lines(self.background, COLORS[agent_list[b].color], 1, points[b], 2)
        else:
            for b in range(len(points)):
                if points[b] is not None:
                    pygame.draw.lines(self.background, COLORS[agent_list[b].color], 1, points[b], 2)

    # def draw_energy_bar(self, agent_list, height = 100):
    #     #coord = [570 + 70 * i for i in range(len(agent_list))]
    #     coord = [570 + 70 * i for i in range(len(agent_list))]
    #
    #     for agent_idx in range(len(agent_list)):
    #         if agent_list[agent_idx].type == 'ball':
    #             continue
    #         remaining_energy = agent_list[agent_idx].energy/agent_list[agent_idx].energy_cap
    #         start_pos = [coord[agent_idx], height]
    #         end_pos=  [coord[agent_idx] + 50*remaining_energy, height]
    #         pygame.draw.line(self.background, color=COLORS[agent_list[agent_idx].color], start_pos=start_pos,
    #                          end_pos=end_pos, width = 5)
    #



    # def draw_view(self, obs, agent_list, view_y=30):       #obs: [2, 100, 100] list
    #
    #     #draw agent 1, [50, 50], [50+width, 50], [50, 50+height], [50+width, 50+height]
    #     count = 0
    #     coord = 580
    #
    #     # coord = [580 + 70 * i for i in range(len(obs))]
    #     for agent_idx in range(len(obs)):
    #         matrix = obs[agent_idx]
    #         if matrix is None:
    #             continue
    #
    #         obs_weight, obs_height = matrix.shape[0], matrix.shape[1]
    #         y = view_y - obs_height
    #         for row in matrix:
    #             x = coord- obs_height/2
    #             for item in row:
    #                 pygame.draw.rect(self.background, COLORS[IDX_TO_COLOR[int(item)]], [x,y,grid_node_width, grid_node_height])
    #                 x+= grid_node_width
    #             y += grid_node_height
    #
    #         pygame.draw.circle(self.background, COLORS[agent_list[agent_idx].color], [coord+20, 55 + agent_list[agent_idx].r],
    #                            agent_list[agent_idx].r, width=0)
    #         pygame.draw.circle(self.background, COLORS["black"], [coord+10, 55 + agent_list[agent_idx].r], 2,
    #                            width=0)
    #
    #         pygame.draw.lines(self.background, points =[[566+70*count,5],[566+70*count, 55], [566+50+70*count, 55], [566+50+70*count, 5]], closed=True,
    #                           color = COLORS[agent_list[agent_idx].color], width=2)
    #         count += 1
    #         coord += 70

    def draw_view(self, obs, agent_list, leftmost_x, upmost_y, gap = 70, view_ifself=True, energy_width = 5):

        count = 0
        x_start = leftmost_x
        y_start = upmost_y
        obs_height = None

        for agent_idx in range(len(obs)):
            matrix = obs[agent_idx]
            if matrix is None:
                continue

            obs_width, obs_height = matrix.shape[0], matrix.shape[1]
            y = y_start
            for row in matrix:
                x = x_start
                for item in row:
                    pygame.draw.rect(self.background, COLORS[IDX_TO_COLOR[int(item)]], [x,y,grid_node_width, grid_node_height])
                    x += grid_node_width
                y += grid_node_height

            center_x = x_start + ((obs_width)*grid_node_width)/2 #- agent_list[agent_idx].r
            center_y = y_start + (obs_height)*grid_node_height + agent_list[agent_idx].r

            if not view_ifself:
                pygame.draw.circle(self.background, COLORS[agent_list[agent_idx].color], [center_x, center_y],
                                   agent_list[agent_idx].r, width=0)
                pygame.draw.circle(self.background, COLORS['black'], [center_x, center_y],
                                   2, width=0)

            pygame.draw.lines(self.background, points =[[x_start,y_start],
                                                        [x_start, y_start+obs_height*grid_node_height],
                                                        [x_start+obs_width*grid_node_width, y_start+obs_height*grid_node_height],
                                                        [x_start+obs_width*grid_node_width, y_start]], closed=True,
                              color = COLORS[agent_list[agent_idx].color], width=1)

            count += 1
            x_start += gap

        if obs_height is not None:
            count2 = 0
            x_start2 = leftmost_x
            y_start2 = upmost_y + obs_height * grid_node_height + agent_list[agent_idx].r + 25

            #draw energy bar
            for agent_idx in range(len(agent_list)):
                if agent_list[agent_idx].type == 'ball':
                    continue


                remaining_energy = agent_list[agent_idx].energy/agent_list[agent_idx].energy_cap
                start_pos = [x_start2 , y_start2]
                end_pos=  [x_start2 + obs_width*2*remaining_energy, y_start2]
                pygame.draw.line(self.background, color=COLORS[agent_list[agent_idx].color], start_pos=start_pos,
                                 end_pos=end_pos, width = energy_width)

                debug(f"team {count2}", x = x_start2+obs_width*0.5,y = y_start2 + 15, c='black')

                count2 += 1
                x_start2 += gap






pygame.init()
font = pygame.font.Font(None, 18)
def debug(info, y = 10, x=10, c='black'):
    display_surf = pygame.display.get_surface()
    debug_surf = font.render(str(info), True, COLORS[c])
    debug_rect = debug_surf.get_rect(topleft = (x,y))
    display_surf.blit(debug_surf, debug_rect)


