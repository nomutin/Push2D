import hydra
import pygame
from omegaconf import DictConfig


@hydra.main(config_path=".", config_name="config_example", version_base=None)
def main(config: DictConfig) -> None:
    pygame.display.init()

    from env import Env  # nopep8

    env = Env(config)
    env.save()


if __name__ == "__main__":
    main()
