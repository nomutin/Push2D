from push_2d.core import Push2D
from push_2d.wrapper import Saver

env = Push2D.from_yaml("push_2d_env")
saver = Saver(env=env, seq_len=100)
saver.reset(seed=42)

while True:
    saver.listen()
