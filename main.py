import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from environment.market_env import MarketEnv
from models.A2C import Actor, Critic
from models.benchmark_agent import BenchmarkAgent
from utils.visualize import visualize_episode_data
from utils.train import train
from utils.evaluate import evaluate
from config import (INIT_PRICE, ACTION_RANGE, QUANTITY_EACH_TRADE,
                    MAX_STEPS, N_ITERS, EVAL_INTERVAL, device)

def setup_environment():
    env = MarketEnv(init_price=INIT_PRICE, action_range=ACTION_RANGE, 
                    quantity_each_trade=QUANTITY_EACH_TRADE, max_steps=MAX_STEPS)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.nvec[0]
    print(f"Dimension of state: {state_size}")
    print(f"Dimension of each action: {action_size}")
    return env, state_size, action_size

def main():
    env, state_size, action_size = setup_environment()
    
    # Initialize and train RL agent
    actor = Actor(state_size, ACTION_RANGE).to(device)
    critic = Critic(state_size).to(device)
    actor, critic = train(env, actor, critic, n_iters=N_ITERS, eval_interval=EVAL_INTERVAL)

    # Load the best model for evaluation
    checkpoint = torch.load('best_model.pth')
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])

    # Evaluate RL agent
    rl_results = evaluate(env, actor, num_episodes=100, is_rl_agent=True)
    visualize_episode_data(rl_results['episode_data'], 'RL')
    
    # Initialize and evaluate benchmark agent
    benchmark_agent = BenchmarkAgent(ACTION_RANGE)
    benchmark_results = evaluate(env, benchmark_agent, num_episodes=100, is_rl_agent=False)
    visualize_episode_data(benchmark_results['episode_data'], 'Benchmark')
    
    # Compare results
    print("\nRL Agent vs Benchmark Agent:")
    for key in rl_results.keys():
        if key != 'episode_data':
            rl_value = rl_results[key]
            benchmark_value = benchmark_results[key]
            print(f"{key}:")
            print(f"  RL Agent: {rl_value:.4f}")
            print(f"  Benchmark: {benchmark_value:.4f}")
            print(f"  Difference: {rl_value - benchmark_value:.4f}")
            print()

if __name__ == "__main__":
    main()
