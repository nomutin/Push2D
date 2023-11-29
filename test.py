from push_2d.environment import Push2D
from push_2d.wrapper import Saver

env = Push2D.from_yaml("button")
saver = Saver(env=env, seq_len=200)
saver.reset(seed=42)

while True:
    saver.listen()
