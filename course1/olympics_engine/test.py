import pygame
import math
import sys

pygame.init()
size = width, height = 800, 900
screen = pygame.display.set_mode(size)
screen.fill((255, 255, 255))


while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()


    #pygame.draw.ellipse(screen, [0,0,0], rect = [100,100, 500,800], width = 2)

    pygame.draw.line(screen, [0,0,0], start_pos = [20,600], end_pos = [400, 600], width = 2)
    pygame.draw.line(screen, [0,0,0], start_pos = [20,800], end_pos = [400, 800], width = 2)
    pygame.draw.line(screen, [0,0,0], start_pos = [20,600], end_pos = [20,800], width = 2)
    pygame.draw.line(screen, [176,196,222], start_pos = [20,700], end_pos = [425,700], width = 2)

    pygame.draw.rect(screen, [0,0,0], [0,0,100,200], width = 2)
    pygame.draw.arc(screen, [0,0,0], [0,0,100,200], -1/2*math.pi, 1/2*math.pi, 2)
    pygame.draw.arc(screen, [0,0,0], [200,100,400,700], -1/2*math.pi, 1/2*math.pi, 2)
    pygame.draw.arc(screen, [0,0,0], [350,300,100,300], -1/2*math.pi, 1/2*math.pi, 2)

    pygame.draw.arc(screen, [176,196,222], [325, 200, 200, 500], -1/2*math.pi, 1/2*math.pi, 2)

    #pygame.draw.arc(screen, [0,0,0], [(250,300),(300, 300)], -1/2*math.pi, 1/2*math.pi, 2)
    #pygame.draw.arc(screen, [0,0,0], [50,100,700, 700], -1/2*math.pi, 1/2*math.pi, 2)
    #pygame.draw.arc(screen, [176,196,222], [150,200,500, 500], -1/2*math.pi, 1/2*math.pi, 2)


    pygame.draw.line(screen, [0,0,0], start_pos = [20,100], end_pos = [415, 100], width = 2)
    pygame.draw.line(screen, [0,0,0], start_pos = [20,300], end_pos = [415, 300], width = 2)
    pygame.draw.line(screen, [176,196,222], start_pos = [20,200], end_pos = [430,200], width = 2)


    pygame.draw.line(screen, [255, 0, 0], start_pos = [20,100], end_pos = [20,300], width = 2)

    pygame.display.flip()