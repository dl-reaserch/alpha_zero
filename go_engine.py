#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import pygame
import os
import time

# 设置我们的屏幕大小和标题
# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

FPS = 30


class GoBoard(object):
    def __init__(self, width, height, lines, fps):
        self.width = width
        self.height = height
        self.lines = lines
        self.grid_width = width // self.lines
        self.fps = fps
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("SmyGo")
        self.clock = pygame.time.Clock()
        self.all_sprites = pygame.sprite.Group()
        base_folder = os.path.dirname(__file__)
        # 加载资源

        img_folder = os.path.join(base_folder, 'images')
        # <class 'pygame.Surface'>
        background_img = pygame.image.load(os.path.join(img_folder, 'back.png'))
        snd_folder = os.path.join(base_folder, 'music')
        hit_sound = pygame.mixer.Sound(os.path.join(snd_folder, 'buw.wav'))
        back_music = pygame.mixer.music.load(os.path.join(snd_folder, 'background.mp3'))
        pygame.mixer.music.set_volume(0.4)
        self.background = pygame.transform.scale(background_img, (width, self.height))
        # 字体
        self.font_name = pygame.font.get_default_font()
        self.back_rect = self.background.get_rect()

    # draw background lines
    def draw_background(self, surf):
        # 加载背景图片

        self.screen.blit(self.background, self.back_rect)
        # 画网格线，棋盘为 19行 19列的
        # 1. 画出边框
        rect_lines = [
            ((self.grid_width, self.grid_width), (self.grid_width, self.height - self.grid_width)),
            ((self.grid_width, self.grid_width), (self.width - self.grid_width, self.grid_width)),
            ((self.grid_width, self.height - self.grid_width),
             (self.width - self.grid_width, self.height - self.grid_width)),
            ((self.width - self.grid_width, self.grid_width),
             (self.width - self.grid_width, self.height - self.grid_width)),
        ]

        for line in rect_lines:
            pygame.draw.line(surf, BLACK, line[0], line[1], 2)

        for i in range(7):
            pygame.draw.line(surf, BLACK,
                             (self.grid_width * (2 + i), self.grid_width),
                             (self.grid_width * (2 + i), self.height - self.grid_width))
            pygame.draw.line(surf, BLACK,
                             (self.grid_width, self.grid_width * (2 + i)),
                             (self.height - self.grid_width, self.grid_width * (2 + i)))
            # circle_center = [
            #     (self.grid_width * 4, self.grid_width * 4),
            #     (WIDTH - self.grid_width * 4, self.grid_width * 4),
            #     (WIDTH - self.grid_width * 4, self.height - self.grid_width * 4),
            #     (self.grid_width * 4, self.height - self.grid_width * 4),
            #     (self.grid_width * 10, self.grid_width * 10)
            # ]

    def draw_text(self, surf, text, size, x, y, color=WHITE):
        font = pygame.font.Font(self.font_name, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)
        surf.blit(text_surface, text_rect)

    def show_go_screen(self, surf, winner=None):

        if winner is not None:
            self.draw_text(surf, ' {0} Win !'.format(winner),
                           64, self.width // 2, self.lines, RED)
        else:
            self.screen.blit(self.background, self.back_rect)
        self.draw_text(surf, 'Five in row', 64, self.width // 2, self.lines + self.height // 4, BLACK)
        self.draw_text(surf, 'Press any key to start', 22, self.width // 2, self.lines + self.height // 2,
                       BLUE)

        pygame.display.flip()
        # waiting = True
        # while waiting:
        #     self.clock.tick(FPS)
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #     elif event.type == pygame.KEYUP:
        #         waiting = False

    def init_engine(self, screen):
        self.all_sprites.draw(screen)
        self.draw_background(screen)


    def show_stone(self, screen, color, point, radius):

        pygame.draw.circle(screen, color, point, radius)

    def show_stones(self, screen, points, radius):
        screen.fill(WHITE)
        self.init_engine(screen)
        for point, color in points.items():
            pygame.draw.circle(screen, color, point, radius)
        pygame.display.flip()

    def reset_board(self, screen):
        self.all_sprites.draw(screen)
        self.draw_background(screen)
        pygame.display.flip()

        #
        # board = GoBoard(500, 500, 10, 30)
        # for j in range(10):
        #     moves =[(2,7),(4,6),(7,3),(2,1),(8,6)]
        #
        #     game_over = False
        #     running = True
        #     winner = None
        #     board.init_engine(board.screen)
        #
        #     for i,move in enumerate(moves):
        #         point = ((move[0]+1)*board.grid_width,(move[1]+1)*board.grid_width)
        #         color = BLACK if j % 2 ==0 else WHITE
        #         board.show_stone(board.screen, color, point, 10)
        #     pygame.display.flip()
        # board.screen.fill(BLACK)
        # for i,move in enumerate(moves):
        #     point = ((move[0]+1)*board.grid_width,(move[1]+1)*board.grid_width)
        #     color = BLACK if i % 2 ==0 else BLACK
        #     board.show_stone(board.screen, color, point, 10)

        # time.sleep(3)




        # clone_screen = board.screen
        # game_over = False
        # running = True
        # winner = None
        # board.all_sprites.draw(clone_screen)
        # board.draw_background(clone_screen)
        # pygame.draw.circle(clone_screen, BLACK, (50, 50), 10)
        # pygame.display.flip()
        # time.sleep(3)
        # pygame.draw.circle(clone_screen, WHITE, (100, 50), 10)
        # pygame.display.flip()
        # time.sleep(3)
        # board.all_sprites.draw(board.screen)
        # board.draw_background(board.screen)
        # pygame.display.flip()
        # time.sleep(3)


        # Update

        # if game_over:
        #     show_go_screen(screen, winner)
        #     game_over = False
        #     movements = []
        #     remain = set(range(1, 19 ** 2 + 1))
        #
        #     player_score_metrix = [[0] * 20 for i in range(20)]
        #     ai_score_metrix = [[0] * 20 for i in range(20)]
        #     color_metrix = [[None] * 20 for i in range(20)]
        #
        #     ai_possible_list = []
        #     ai_optimal_list = []
        #     ai_tabu_list = []
        #     player_optimal_set = set()
        #     player_tabu_list = []
        #
        # clock.tick(FPS)
        # # 处理不同事件
        # for event in pygame.event.get():
        #
        #     # 检查是否关闭窗口
        #     if event.type == pygame.QUIT:
        #         running = False
        #     if event.type == pygame.MOUSEBUTTONDOWN:
        #        # response = move(screen, event.pos)
        #        #  if response is not None and response[0] is False:
        #        #      game_over = True
        #        #      winner = response[1]
        #             continue
        # Update
        # all_sprites.update()

# all_sprites.draw(screen)
# pygame.init()
