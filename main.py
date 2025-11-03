from typing import Union, Dict, Any
import torch
from collections import deque
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

from boolrl import config
from boolrl.environments.boolean_env_mlp import BooleanSimplificationEnv as MlpEnv
from boolrl.agents.agent_mlp import DQNAgent as MlpAgent
from boolrl.environments.boolean_env_gnn import BooleanSimplificationEnvGNN as GnnEnv
from boolrl.agents.agent_gnn import DQNAgentGNN as GnnAgent
from boolrl.environments.boolean_env_seq import BooleanSimplificationEnvSeq as SeqEnv
from boolrl.agents.agent_seq import DQNAgentSeq as SeqAgent
from boolrl.environments.base_env import BaseBooleanEnv
from boolrl.agents import agent_mlp, agent_gnn, agent_seq

def train(agent: Union[MlpAgent, GnnAgent, SeqAgent], 
          env: BaseBooleanEnv, 
          model_type: str, 
          save_every: int = 100) -> None:
    scores = deque(maxlen=100)
    all_scores = []
    start_time = time.time()

    for episode in range(1, config.EPISODES + 1):
        state = env.reset()
        initial_expr = env.current_expression
        episode_reward = 0.0
        for step in tqdm(range(config.MAX_STEPS_PER_EPISODE), desc=f"Episode {episode}/{config.EPISODES}", leave=False):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                break

        scores.append(episode_reward)
        all_scores.append(episode_reward)
        avg_score = np.mean(scores)

        COLOR_LAVENDER = "\u001b[95m"
        COLOR_BLUE = "\u001b[94m"
        COLOR_GREEN = "\u001b[92m"
        COLOR_RESET = "\u001b[0m"

        print(f"{COLOR_LAVENDER}--- Episode {episode}/{config.EPISODES} ---{COLOR_RESET}")
        print(f"Initial Expression: {COLOR_BLUE}{initial_expr}{COLOR_RESET}")
        print(f"Final Expression:   {COLOR_GREEN}{env.current_expression}{COLOR_RESET}")
        print(f"Episode Reward:     {episode_reward:.2f}")
        print(f"Average Score:      {avg_score:.2f}")
        print(f"Epsilon:            {agent.epsilon:.2f}")
        print(f"Initial Complexity: {env.initial_complexity}")
        print(f"Final Complexity:   {env._get_complexity(env.current_expression)}")
        print("-" * (23 + len(str(episode)) + len(str(config.EPISODES))))

        if episode % save_every == 0:
            torch.save(agent.qnet_policy.state_dict(), f'checkpoint_{model_type}_e{episode}.pth')
            print(f"Saved checkpoint to checkpoint_{model_type}_e{episode}.pth")

        if episode >= 100 and avg_score >= config.SOLVE_SCORE:
            print(f"Environment solved in {episode} episodes.\tAverage Score: {avg_score:.2f}")
            torch.save(agent.qnet_policy.state_dict(), f'checkpoint_{model_type}_final.pth')
            break

    torch.save(agent.qnet_policy.state_dict(), f'checkpoint_{model_type}.pth')
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    plot_scores(all_scores, model_type)

def plot_scores(scores: list, model_type: str) -> None:
    avg_scores = [np.mean(scores[max(0, i - 100):i + 1]) for i in range(len(scores))]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(scores, label='Episode Score', color='#3498db', alpha=0.6)
    ax.plot(avg_scores, label='100-Episode Average', color='#e74c3c', linewidth=2)

    ax.set_title(f'Training Progress ({model_type.upper()})', fontsize=18, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)

    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(fontsize=12)

    plt.savefig(f'training_plot_{model_type}.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'gnn', 'seq'])
    args = parser.parse_args()

    if args.model_type == 'mlp':
        env = MlpEnv(max_expression_depth=config.MAX_EXPRESSION_DEPTH,
                     max_literals=config.MAX_LITERALS,
                     max_steps=config.MAX_STEPS_PER_EPISODE)
        agent = MlpAgent(
            state_size=env.get_state_size(),
            action_size=env.get_action_size(),
            seed=config.AGENT_SEED,
            hidden_size=config.HIDDEN_SIZE,
            learning_rate=config.LEARNING_RATE,
            gamma=config.GAMMA,
            tau=config.TAU,
            buffer_size=config.BUFFER_SIZE,
            batch_size=config.BATCH_SIZE,
            update_every=config.UPDATE_EVERY
        )
    elif args.model_type == 'gnn':
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)

        env = GnnEnv(max_expression_depth=config.MAX_EXPRESSION_DEPTH,
                     max_literals=config.MAX_LITERALS,
                     max_steps=config.MAX_STEPS_PER_EPISODE)
        agent = GnnAgent(
            gnn_input_size=env.get_gnn_input_size(),
            global_feature_size=env.get_global_feature_size(),
            action_size=env.get_action_size(),
            seed=config.AGENT_SEED,
            hidden_size=config.HIDDEN_SIZE,
            learning_rate=config.LEARNING_RATE,
            gamma=config.GAMMA,
            tau=config.TAU,
            buffer_size=config.BUFFER_SIZE,
            batch_size=config.BATCH_SIZE,
            update_every=config.UPDATE_EVERY
        )
    elif args.model_type == 'seq':
        env = SeqEnv(max_expression_depth=config.MAX_EXPRESSION_DEPTH,
                     max_literals=config.MAX_LITERALS,
                     max_steps=config.MAX_STEPS_PER_EPISODE)
        agent = SeqAgent(
            vocab_size=env.vocab_size,
            action_size=env.get_action_size(),
            seed=config.AGENT_SEED,
            hidden_size=config.HIDDEN_SIZE,
            learning_rate=config.LEARNING_RATE,
            gamma=config.GAMMA,
            tau=config.TAU,
            buffer_size=config.BUFFER_SIZE,
            batch_size=config.BATCH_SIZE,
            update_every=config.UPDATE_EVERY
        )
    
    print(f"Starting training for {args.model_type.upper()} model...")
    train(agent, env, args.model_type)
