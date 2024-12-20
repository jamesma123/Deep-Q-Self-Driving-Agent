import torch
from DQN_Control.replay_buffer import ReplayBuffer
from DQN_Control.model import DQN

from config import action_map, env_params
from utils import *
from environment import SimEnv
import os

def run():
    try:
        buffer_size = 1e4
        batch_size = 32
        state_dim = (128, 128)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_actions = len(action_map)
        in_channels = 1
        episodes = 10000  # Total number of episodes

        # Initialize replay buffer and DQN model
        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        model = DQN(num_actions, state_dim, in_channels, device)

        # Initialize the environment
        env = SimEnv(visuals=True, **env_params)

        # Path to saved model weights
        saved_model_path = "weights/dqnX3200_model_ep_1000"  # Adjust the path to where your model is stored
        model.load(saved_model_path)

        # Continue training from the loaded model
        for ep in range(episodes):
            env.create_actors()
            env.generate_episode(model, replay_buffer, ep, action_map, eval=False)
            env.reset()


    finally:
        env.quit()

if __name__ == "__main__":
    run()
