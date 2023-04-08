from random import randint

import hydra
import pygame
from omegaconf import DictConfig


from components import Circle, Space, mouse_track


class Keys:
    @classmethod
    def is_quit(cls, event) -> bool:
        return event.type == pygame.QUIT

    @staticmethod
    def is_key(event) -> bool:
        return event.type == pygame.KEYDOWN

    @classmethod
    def is_q(cls, event) -> bool:
        return cls.is_key(event) and event.key == pygame.K_q

    @classmethod
    def is_r(cls, event) -> bool:
        return cls.is_key(event) and event.key == pygame.K_r


class Env:
    def __init__(self, config: DictConfig) -> None:
        pygame.init()
        self.config = config
        self.space = Space(
            width=config.screen.width,
            height=config.screen.height,
            fps=config.screen.fps,
            color=config.screen.color,
        )
        self.reset()

    def reset(self) -> None:

        self.space.space.remove(*self.space.space.bodies)
        self.space.space.remove(*self.space.space.shapes)

        self.tracker = Circle(
            radius=self.config.tracker.radius,
            color=self.config.tracker.color,
            position=(1, 1),
        )
        self.space.add(self.tracker)
        for obj in self.config.objects:
            x = randint(obj.radius, self.config.screen.width - obj.radius)
            y = randint(obj.radius, self.config.screen.height - obj.radius)
            circle = Circle(
                radius=obj.radius, color=obj.color, position=(x, y)
            )
            self.space.add(circle)

        self.space.add_segments()

        print(self.space.space.bodies)
        print(self.space.space.shapes)

    def save(self) -> None:
        while True:
            for event in pygame.event.get():
                if Keys.is_quit(event) or Keys.is_q(event):
                    pygame.quit()
                    exit()
                if Keys.is_r(event):
                    self.reset()

            mouse_track(
                tracker=self.tracker,
                velocity=self.config.tracker.velocity,
                x_limit=self.config.screen.width,
                y_limit=self.config.screen.height,
            )
            self.space.step()

    def rollout(self) -> None:
        pass


@hydra.main(config_path=".", config_name="config_example", version_base=None)
def main(config: DictConfig) -> None:
    env = Env(config)
    env.save()


if __name__ == "__main__":
    main()
