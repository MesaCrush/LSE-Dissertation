import torch

# Common parameters
ACTION_RANGE = 5
QUANTITY_EACH_TRADE = 50
MAX_STEPS = 500

N_ITERS = 300
EVAL_INTERVAL = 10

# MarketEnv parameters
INIT_PRICE = 100.0

# HistoricalMarketEnv parameters
HISTORICAL_DATA_PATH = '~/Documents/GitHub/LSE-Dissertation/data/data_filled.csv'

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")