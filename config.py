import torch

# Environment parameters
INIT_PRICE = 100.0
ACTION_RANGE = 10
QUANTITY_EACH_TRADE = 10
MAX_STEPS = 100
N_ITERS = 100
EVAL_INTERVAL = 20

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")