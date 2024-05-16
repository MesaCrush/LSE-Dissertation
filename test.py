import unittest
from market_env import MarketEnv

class TestMarketEnv(unittest.TestCase):
    def setUp(self):
        # Initialize MarketEnv with some default parameters
        self.env = MarketEnv(
            init_price=100.0, 
            action_range=10, 
            price_range=200.0, 
            target_quantity=1000, 
            target_time=60, 
            penalty_rate=0.01
        )

    def test_initial_state(self):
        # Test the initial state of the environment
        initial_state = self.env.reset()
        self.assertIn('remaining_quantity', initial_state)
        self.assertIn('mid_price', initial_state)
        # Add assertions for other state components here...
        self.assertEqual(initial_state['remaining_quantity'], 1000)
        self.assertEqual(initial_state['mid_price'], 100.0)
        self.assertEqual(initial_state['remaining_time'], 60)
       
    
    def test_step_function(self):
        # Test the step function with a sample action
        last_state = self.env.reset()
        action = self.env.action_space.sample()  # Choose a random action
        state, reward, done, info = self.env.step(action, last_state=last_state)

        # Assert the structure of the output
        self.assertIsInstance(state, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

        # Check if the remaining quantity is updated correctly
        self.assertLessEqual(state['remaining_quantity'], self.env.remaining_quantity)
        # Check if remaining time is decremented
        self.assertEqual(state['remaining_time'], self.env.remaining_time - 1)

    def test_action_space(self):
        # Test if the action space contains values within the expected range
        action = self.env.action_space.sample()
        self.assertGreaterEqual(action, 0)
        self.assertLessEqual(action, 10)

    def tearDown(self):
        # Clean up and close the environment
        self.env.close()

# This allows the test to be executed when the script is run directly
# if __name__ == '__main__':
#     unittest.main()