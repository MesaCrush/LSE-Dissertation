import ray
from ray import tune
from gymnasium.envs.registration import register
import gymnasium as gym
from market_env import MarketEnv  # Make sure this is the correct import for your MarketEnv
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQN 
from ray.rllib.algorithms.ppo import PPO
import matplotlib.pyplot as plt
import numpy as np





""""
Gerneral steps for Training and Evaluation the model:

1. regist env in gym
2. regist env in ray using with a function 
3. train a model and save the checkpoint
4. restore the model from the checkpoint and evaluate it
"""


def register_env(env_id, entry_point, max_episode_steps, env_kwargs):
    register(
        id=env_id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        kwargs=env_kwargs
    )
    gym.make(env_id)

def create_env(env_config):
    return MarketEnv(**env_config)

def train_dqn(env_id, config, stop_criteria):
     # Registers the custom environment with Ray
    tune.register_env(env_id, create_env)

    # Initialize Ray
    ray.init(ignore_reinit_error=True,num_cpus=8)
    
    # Configure and instantiate the DQN algorithm
    algo_config = DQNConfig().environment(
        env=env_id, 
        env_config=config["env_config"]  # Pass env_config here
    ).framework(config["framework"])
    
    # Train the agent
    result = tune.run(
        "DQN",
        config=algo_config.to_dict(),
        stop=stop_criteria,
        checkpoint_at_end=True,
    )
    
    # Get the best checkpoint after training is done
    best_trial = result.get_best_trial("episode_reward_mean", mode="max", scope="all")
    best_checkpoint = result.get_best_checkpoint(best_trial, "episode_reward_mean", mode="max")
    
    # Shutdown Ray
    ray.shutdown()
    return best_checkpoint



def train_PPO(env_id, config, stop_criteria):
    # Registers the custom environment with Ray
    tune.register_env(env_id, lambda config: create_env(config))

    # Initialize Ray
    ray.init(ignore_reinit_error=True, num_cpus=8)
    
    # Configure and instantiate the Actor-Critic algorithm (using A3C in this example)
    algo_config = PPOConfig().environment(
        env=env_id,
        env_config=config["env_config"]  # Pass env_config here
    ).framework(config["framework"])
    
    # Train the agent
    result = tune.run(
        "PPO",
        config=algo_config.to_dict(),
        stop=stop_criteria,
        checkpoint_at_end=True,
    )
    
    # Get the best checkpoint after training is done
    best_trial = result.get_best_trial("episode_reward_mean", mode="max", scope="all")
    best_checkpoint = result.get_best_checkpoint(best_trial, "episode_reward_mean", mode="max")
    
    # Shutdown Ray
    ray.shutdown()
    return best_checkpoint




def evaluation(model_type, config, checkpoint_path, env_kwargs):
    

    # Setup the Trainer based on the model type
    if model_type == 'DQN':
        trainer = DQN(config=config)
    elif model_type == 'PPO':
        trainer = PPO(config=config)
    else:
        raise ValueError("Unsupported model type. Please choose 'DQN' or 'A3C'.")

    # Restore from the checkpoint
    trainer.restore(checkpoint_path)

    # Now you can use the trainer to compute actions and interact with the environment
    register_env( 'MarketEnv-v0', 'market_env:MarketEnv', 2000, env_kwargs)
    env = create_env(config["env_config"])
    num_episodes = 10

    all_actions = []
    all_mid_prices = []
    all_rewards = []
    all_diff = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_actions = []
        episode_mid_prices = []
        episode_reward = []
        twap = 0
        traded_qty = 0

        while not done:
            action = trainer.compute_single_action(obs, explore=False)
            obs, reward, truncated, done, info = env.step(action)
            mid_price = info['best_ask'] + 0.07 # Assuming obs[1] is the mid-price

            # Collect data for plotting
            episode_actions.append(action)
            episode_mid_prices.append(mid_price)
            episode_reward.append(reward)
            twap += mid_price * (env_kwargs['target_quantity']/env_kwargs['target_time'])
            traded_qty += (env_kwargs['target_quantity']/env_kwargs['target_time'])
            

        twap = twap / traded_qty
        overall_trade_price = env.overall_trade_price/env_kwargs['target_quantity']
        all_actions.append(episode_actions)
        all_mid_prices.append(episode_mid_prices)
        all_rewards.append(sum(episode_reward))
        all_diff.append(twap-overall_trade_price)
        print(f"Episode {episode + 1} reward: {sum(episode_reward)}, compare with twap:{twap},{overall_trade_price},{twap-overall_trade_price}")

    # Plotting outside the loop after collecting all data
    plot_actions_vs_mid_prices(all_actions, all_mid_prices)

    # Shutdown Ray
    ray.shutdown()
    return np.mean(all_rewards), np.mean(all_diff)

def plot_actions_vs_mid_prices(actions, mid_prices):
    for episode_index, (episode_actions, episode_mid_prices) in enumerate(zip(actions, mid_prices), start=1):
        plt.figure(figsize=(10, 5))
        
        # Create the first axis for the actions
        ax1 = plt.gca()  # Gets the current axis (or 'ax1')
        ax1.plot(episode_actions, label='Actions', color='b', marker='o', linestyle='--')
        ax1.set_ylabel('Action Value', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xlabel('Step')
        ax1.set_title(f'Actions vs Mid Price for Episode {episode_index}')
        
        # Create a second y-axis for the mid prices
        ax2 = ax1.twinx()  # Creates a twin of the original axis that shares the x-axis
        ax2.plot(episode_mid_prices, label='Mid Price', color='r', marker='x', linestyle='-')
        ax2.set_ylabel('Mid Price', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Adding legends which consider both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best')
        
        plt.show()


def main():
    max_episode_steps = 200
    env_id = 'MarketEnv-v0'
    env_kwargs = {
        'init_price': 100.0,
        'action_range': 50,
        'price_range': 200.0,
        'target_quantity': 3000,
        'target_time': 100,
        'penalty_rate': 0.01
    }
    register_env(env_id, 'market_env:MarketEnv', max_episode_steps, env_kwargs=env_kwargs)

    config = {
        "env": env_id,
        "env_config": env_kwargs,
        "framework": "torch",
        "resources_per_trial": {
            "cpu": 4,  # Allocating CPUs for parallelism
        },
    }


    stop_criteria = {
        "training_iteration": 100,
        "timesteps_total": 100*max_episode_steps
}

    checkpoint_path = train_dqn(env_id, config, stop_criteria)
    print(f"Checkpoint saved at: {checkpoint_path}")
    res = evaluation(model_type ='DQN', config=config, checkpoint_path=checkpoint_path,env_kwargs=env_kwargs)
    print(res)

if __name__ == "__main__":
    main()
