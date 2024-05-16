class Rewards:

    def reward_for_market_impact(self, agent_avg_price, initial_market_price, zero_impact_reward=50, tolerance=0.1):
        """
        Calculate the reward for market impact based on the price difference.

        agent_avg_price: The average price at which the agent's orders were executed.
        initial_market_price: The market price at the beginning of the time step before any market orders
        were executed by any party.
        zero_impact_reward: High fixed reward for minimal or no impact.
        tolerance: Price difference tolerance within which minimal impact is considered.
        """
        price_impact = agent_avg_price - initial_market_price

        if agent_avg_price == 0:
            return 0  # No orders executed, no reward.
        elif abs(price_impact) <= tolerance:
            return zero_impact_reward * 0.01 # Provide a high fixed reward for low or no impact.
        elif price_impact < 0:
            # If agent's average price is better than the initial price, penalise.
            return abs(price_impact) 
        else:
            # Reward based on the magnitude of negative impact when the average price is better.
            return - abs(price_impact)
    
    def reward_based_obs(self, imbalance_best, action):
        if imbalance_best > 0.5:
            return 0.01 * 50 * 1 if action-25>=0 else -1
        else:
             return 0.01 * 50 * -1 if action-25>=0 else 1


    def calculate_dynamic_penalty(self, current_step, total_steps, remaining_quantity, penalty_rate, base_penalty, scale_type):
        """
        Calculate a time-scaled penalty for holding inventory.

        Parameters:
        - current_step (int): The current step in the simulation.
        - total_steps (int): The total number of steps in the simulation.
        - remaining_quantity (float): The quantity of inventory held by the agent.
        - penalty_rate (float): The base penalty rate for holding inventory.
        - base_penalty (float, optional): The base penalty factor. Default is 0.1.
        - scale_type (str, optional): The type of scaling to use ('linear' or 'exponential'). Default is 'linear'.

        Returns:
        - float: The calculated penalty.
        """
        if scale_type == 'linear':
            scaling_factor = current_step / total_steps
        elif scale_type == 'exponential':
            scaling_factor = (current_step / total_steps) ** 2
        else:
            raise ValueError("Invalid scale_type. Use 'linear' or 'exponential'.")

        return penalty_rate * remaining_quantity * base_penalty * scaling_factor


