import hydra
import matplotlib.animation as anime
import matplotlib.pyplot as plt
import numpy as np
import pygame
from omegaconf import DictConfig


def array2gif(save_path: str, array) -> None:
    """Convert numpy array to gif"""

    fig = plt.figure(tight_layout=True)
    # array = np.clip(array, 0, 1)
    imgs = [[plt.imshow(i)] for i in array]
    ani = anime.ArtistAnimation(fig, imgs, interval=100)
    ani.save(save_path, writer="pillow")
    plt.cla()
    print(f"saved movie on {save_path}")


@hydra.main(config_path=".", config_name="config_example", version_base=None)
def main(config: DictConfig) -> None:
    pygame.display.init()

    from env import Env  # nopep8

    env = Env(config)
    # env.follow()
    a = np.load("data/4_12_3/raw_action_0.npy")
    obs = []
    for i in a:
        env.replay(i)
        obs.append(env.get_observation())
    array2gif("replay.gif", np.stack(obs))

    a = np.load("data/4_12_3/raw_observation_0.npy")
    array2gif("raw.gif", a)


if __name__ == "__main__":
    main()
