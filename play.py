from push2d_simulator import Env
from omegaconf import OmegaConf
import pygame

pygame.display.init()
a = OmegaConf.load("config_example.yaml")
env = Env(a)
env.follow()
