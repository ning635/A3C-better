import gymnasium as gym
from gymnasium.wrappers import (
    RecordEpisodeStatistics,
    AtariPreprocessing,
    FrameStack,
)

def make_env(env_id: str, seed: int = 1, frame_stack: int = 4):
    """Create a Gymnasium Atari env with canonical A3C preprocessing.
    - NoFrameskip rom
    - Grayscale, downscale to 84x84, frame skip=4, terminal_on_life_loss
    - FrameStack K=4
    Returns obs in shape (84, 84, K), dtype=uint8
    """
    env = gym.make(env_id)
    env.reset(seed=seed)
    # AtariPreprocessing: scale to 84x84, grayscale, frame_skip=4, clip_reward=False, terminal_on_life_loss=True
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False, screen_size=84, terminal_on_life_loss=True)
    env = FrameStack(env, frame_stack)
    env = RecordEpisodeStatistics(env)
    return env
