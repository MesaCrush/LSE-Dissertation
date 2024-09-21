import torch
import argparse
import os
from models.A2C import Actor, Critic
from models.DQN import DQN
from models.DQN_BC import DQN_BC
from models.DQN_BC_DW import DQN_BC_DW
from environment.market_env_simulated import SimulatedMarketEnv
from environment.market_env_historical import HistoricalMarketEnv
from environment.market_env_toy import SimplifiedMarketEnv
from utils.train_A2C import ActorCriticTrainer
from utils.train_A2C_BC import A2C_BC_Trainer  # Import the new A2C_BC_Trainer
from utils.train_DQN import DQNTrainer
from utils.train_DQN_BC import DQN_BC_Trainer 
from utils.train_DQN_BC_DW import DQN_BC_DW_Trainer
from utils.evaluate import evaluate
from utils.visualize import visualize_episode_data
from models.benchmark_agent import BenchmarkAgent
from config import *

def create_env(env_type):
    if env_type == 'simulated':
        return SimulatedMarketEnv(
            init_price=INIT_PRICE,
            action_range=ACTION_RANGE,
            quantity_each_trade=QUANTITY_EACH_TRADE,
            max_steps=MAX_STEPS
        )
    elif env_type == 'historical':
        return HistoricalMarketEnv(
            data_path=HISTORICAL_DATA_PATH,
            action_range=ACTION_RANGE,
            quantity_each_trade=QUANTITY_EACH_TRADE,
            max_steps=MAX_STEPS
        )
    elif env_type == 'toy':
        return SimplifiedMarketEnv(
            init_price=INIT_PRICE,
            action_range=ACTION_RANGE,
            quantity_each_trade=QUANTITY_EACH_TRADE,
            max_steps=MAX_STEPS
        )
    else:
        raise ValueError("Invalid environment type. Choose 'simulated', 'historical', or 'toy'.")

import torch
import argparse
import os
from models.A2C import Actor, Critic
from models.DQN import DQN
from models.DQN_BC import DQN_BC
from models.DQN_BC_DW import DQN_BC_DW
from environment.market_env_simulated import SimulatedMarketEnv
from environment.market_env_historical import HistoricalMarketEnv
from environment.market_env_toy import SimplifiedMarketEnv
from utils.train_A2C import ActorCriticTrainer
from utils.train_DQN import DQNTrainer
from utils.train_DQN_BC import DQN_BC_Trainer 
from utils.train_DQN_BC_DW import DQN_BC_DW_Trainer
from utils.train_A2C_BC import A2C_BC_Trainer  # Import the new A2C_BC_Trainer
from utils.evaluate import evaluate
from utils.visualize import visualize_episode_data
from models.benchmark_agent import BenchmarkAgent
from config import *

# ... (keep the create_env function as is)

def main(env_type, model_type):
    # Environment selection
    env = create_env(env_type)
    state_dim = env.observation_space.shape[0]
    action_dim = ACTION_RANGE

    # Model selection and training
    if model_type == 'a2c':
        actor = Actor(state_dim, action_dim).to(device)
        critic = Critic(state_dim).to(device)
        trainer = ActorCriticTrainer(
            env=env,
            actor=actor,
            critic=critic,
            n_iters=N_ITERS,
            eval_interval=EVAL_INTERVAL,
            save_path=f'best_{model_type}_{env_type}_model.pth'
        )
        trained_actor, trained_critic = trainer.train()
        trained_model = (trained_actor, trained_critic)
    elif model_type == 'a2c_bc':
        actor = Actor(state_dim, action_dim).to(device)
        critic = Critic(state_dim).to(device)
        trainer = A2C_BC_Trainer(
            env=env,
            actor=actor,
            critic=critic,
            n_iters=N_ITERS,
            eval_interval=EVAL_INTERVAL,
            save_path=f'best_{model_type}_{env_type}_model.pth',
            bc_weight=0.5  # Adjust this value as needed
        )
        trained_actor, trained_critic = trainer.train()
        trained_model = (trained_actor, trained_critic)
    elif model_type == 'dqn':
        model = DQN(state_dim, action_dim).to(device)
        trainer = DQNTrainer(
            env=env,
            model=model,
            n_iters=N_ITERS,
            eval_interval=EVAL_INTERVAL,
            save_path=f'best_{model_type}_{env_type}_model.pth'
        )
        trained_model = trainer.train()
    elif model_type == 'dqn_bc':
        model = DQN_BC(state_dim, action_dim).to(device)
        trainer = DQN_BC_Trainer(
            env=env,
            model=model,
            n_iters=N_ITERS,
            eval_interval=EVAL_INTERVAL,
            save_path=f'best_{model_type}_{env_type}_model.pth'
        )
        trained_model = trainer.train()
    elif model_type == 'dqn_bc_dw':
        model = DQN_BC_DW(state_dim, action_dim).to(device)
        trainer = DQN_BC_DW_Trainer(
            env=env,
            model=model,
            n_iters=N_ITERS,
            eval_interval=EVAL_INTERVAL,
            save_path=f'best_{model_type}_{env_type}_model.pth',
            lambda1_start=0.2,
            lambda2_start=0.8
        )
        trained_model = trainer.train()
    else:
        raise ValueError("Invalid model type. Choose 'a2c', 'a2c_bc', 'dqn', 'dqn_bc', or 'dqn_bc_dw'.")

    print("Training completed.")

    # Load the best model
    try:
        if model_type in ['a2c', 'a2c_bc']:
            best_actor = Actor(state_dim, action_dim).to(device)
            best_critic = Critic(state_dim).to(device)
            checkpoint = torch.load(f'best_{model_type}_{env_type}_model.pth')
            best_actor.load_state_dict(checkpoint['actor_state_dict'])
            best_critic.load_state_dict(checkpoint['critic_state_dict'])
            best_model = (best_actor, best_critic)
        elif model_type in ['dqn', 'dqn_bc', 'dqn_bc_dw']:
            model_class = {'dqn': DQN, 'dqn_bc': DQN_BC, 'dqn_bc_dw': DQN_BC_DW}[model_type]
            best_model = model_class(state_dim, action_dim).to(device)
            best_model.load_state_dict(torch.load(f'best_{model_type}_{env_type}_model.pth')['model_state_dict'])
        else:
            raise ValueError(f"Invalid model type: {model_type}")
    except FileNotFoundError:
        print(f"Best model file not found. Using the last trained model.")
        best_model = trained_model
    except Exception as e:
        print(f"Error loading the best model: {e}")
        print("Using the last trained model.")
        best_model = trained_model

    # Create a benchmark agent
    benchmark_agent = BenchmarkAgent(ACTION_RANGE)

    # Evaluate the best RL model
    print(f"Evaluating the best {model_type.upper()} model...")
    rl_results = evaluate(env, best_model, num_episodes=100, is_rl_agent=True)
    
    # Evaluate the benchmark agent
    print("Evaluating the benchmark agent...")
    benchmark_results = evaluate(env, benchmark_agent, num_episodes=100, is_rl_agent=False)

    # Print evaluation results
    print(f"\n{model_type.upper()} Agent Results:")
    for key, value in rl_results.items():
        if key != 'episode_data':
            print(f"{key}: {value}")

    print("\nBenchmark Agent Results:")
    for key, value in benchmark_results.items():
        if key != 'episode_data':
            print(f"{key}: {value}")

    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Visualize episode data for both agents
    visualize_episode_data(rl_results['episode_data'], f"{model_type.upper()}_{env_type}")
    visualize_episode_data(benchmark_results['episode_data'], f"Benchmark_{env_type}")

    print("Visualization completed. Check the 'plots' directory for the generated images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate RL agents in different market environments.")
    parser.add_argument('--env', type=str, choices=['simulated', 'historical', 'toy'], default='simulated',
                        help="Type of environment to use")
    parser.add_argument('--model', type=str, choices=['a2c', 'a2c_bc', 'dqn', 'dqn_bc', 'dqn_bc_dw'], default='a2c',
                        help="Type of model to train")
    args = parser.parse_args()

    main(args.env, args.model)