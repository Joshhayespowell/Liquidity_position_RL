
import gym
from gym import spaces
import numpy as np
from keras import backend as K
import numpy as np
import random
import gym
from collections import deque
from sklearn.preprocessing import StandardScaler
import pandas as pd
import offline_fee_revenue_calculator_diff_activeLic as offline_fee_revenue_calculator
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import ta
import numpy as np
from collections import deque
import numpy as np
import random
import torch
from torch import nn, optim
import torch.nn.functional as F
from collections import deque
import tensorflow as tf
import os
import os
import tensorflow as tf
import sys
import pandas as pd
import sklearn as sk
import tensorflow as tf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'




swaps_data = pd.read_csv("CSV_Data/0.30_swaps.csv")
swaps_data.drop_duplicates(inplace=True)
swaps_data['timestamp'] = pd.to_datetime(swaps_data['transaction.timestamp'], unit='s')
swaps_data.set_index('timestamp', inplace=True)
swaps_data = swaps_data.sort_index() 



# Filter the dataset
granular_price_data = swaps_data.copy()
granular_price_data = granular_price_data[granular_price_data['amount0'] >= 0.001]
granular_price_data['price'] = abs(granular_price_data['amount0'] / granular_price_data['amount1'])

# differencing and normalising the uniswap_price_data

granular_price_data['price_diff'] = granular_price_data['price'].diff()

granular_price_data.dropna(inplace=True)

# standardise the differenced prices
scalar_price = StandardScaler()
granular_price_data['price_diff_norm'] = scalar_price.fit_transform(granular_price_data[['price_diff']])



# load data
uniswap_price_data = pd.read_csv('CSV_Data/0.30_close_data.csv')
uniswap_price_data = uniswap_price_data.drop([0])
uniswap_hourly_volume_data = pd.read_csv('CSV_Data/0.30_hourly_volume_data.csv')
gas_cost_to_burn_then_mint = pd.read_csv('CSV_Data/0.30_dollar_burn_then_mint_cost.csv')
CSV_Data_fee_revenue_data = pd.read_csv('CSV_Data/0.30_fee_revenue_data.csv')

uniswap_price_data.rename(columns={'close': 'price'}, inplace=True)

uniswap_hourly_volume_data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)

# Convert 'date' column to datetime format for each DataFrame
uniswap_price_data['date'] = pd.to_datetime(uniswap_price_data['date'])
uniswap_hourly_volume_data['date'] = pd.to_datetime(uniswap_hourly_volume_data['date'])
gas_cost_to_burn_then_mint['date'] = pd.to_datetime(gas_cost_to_burn_then_mint['date'])
CSV_Data_fee_revenue_data['date'] = pd.to_datetime(CSV_Data_fee_revenue_data['date'])

# Define the start date
start_date = pd.Timestamp("2022-07-01 21:00:00")

# Filter each DataFrame based on the start date
uniswap_price_data = uniswap_price_data[uniswap_price_data['date'] >= start_date].copy()
uniswap_hourly_volume_data = uniswap_hourly_volume_data[uniswap_hourly_volume_data['date'] >= start_date].copy()
gas_cost_to_burn_then_mint = gas_cost_to_burn_then_mint[gas_cost_to_burn_then_mint['date'] >= start_date].copy()
CSV_Data_fee_revenue_data = CSV_Data_fee_revenue_data[CSV_Data_fee_revenue_data['date'] >= start_date].copy()


uniswap_price_data.reset_index(drop=True, inplace=True)
uniswap_hourly_volume_data.reset_index(drop=True, inplace=True)
gas_cost_to_burn_then_mint.reset_index(drop=True, inplace=True)
CSV_Data_fee_revenue_data.reset_index(drop=True, inplace=True)


# calculate indices for 60% and 80% of data
sixty_percent_index_price = int(len(uniswap_price_data) * 0.6)
eighty_percent_index_price = int(len(uniswap_price_data) * 0.8)

sixty_percent_index_volume = int(len(uniswap_hourly_volume_data) * 0.6)
eighty_percent_index_volume = int(len(uniswap_hourly_volume_data) * 0.8)

sixty_percent_index_gas = int(len(gas_cost_to_burn_then_mint) * 0.6)
eighty_percent_index_gas = int(len(gas_cost_to_burn_then_mint) * 0.8)

# slice dataframe into train, validate, and test
uniswap_price_data_train = uniswap_price_data[:sixty_percent_index_price].copy()
uniswap_hourly_volume_data_train = uniswap_hourly_volume_data[:sixty_percent_index_volume].copy()
gas_cost_to_burn_then_mint_train = gas_cost_to_burn_then_mint[:sixty_percent_index_gas].copy()

uniswap_price_data_validate = uniswap_price_data[sixty_percent_index_price:eighty_percent_index_price].copy()
uniswap_hourly_volume_data_validate = uniswap_hourly_volume_data[sixty_percent_index_volume:eighty_percent_index_volume].copy()
gas_cost_to_burn_then_mint_validate = gas_cost_to_burn_then_mint[sixty_percent_index_gas:eighty_percent_index_gas].copy()

uniswap_price_data_test = uniswap_price_data[eighty_percent_index_price:].copy()
uniswap_hourly_volume_data_test = uniswap_hourly_volume_data[eighty_percent_index_volume:].copy()
gas_cost_to_burn_then_mint_test = gas_cost_to_burn_then_mint[eighty_percent_index_gas:].copy()


uniswap_price_data['date'] = pd.to_datetime(uniswap_price_data['date'])



# Get the first date from the DataFrame
first_date = uniswap_price_data['date'].iloc[0]

# Convert to Unix time
unix_time = int(first_date.timestamp())
unix_time



uniswap_price_data = uniswap_price_data_train
uniswap_hourly_volume_data = uniswap_hourly_volume_data_train
gas_cost_to_burn_then_mint = gas_cost_to_burn_then_mint_train



def preprocess_data(uniswap_hourly_volume_data, uniswap_price_data, gas_cost_to_burn_then_mint):
    # Normalizing hourly volume data
    volume_scaler = MinMaxScaler()
    normalised_volume = volume_scaler.fit_transform(uniswap_hourly_volume_data[['hourly_volume']])
    normalised_volume = pd.DataFrame(normalised_volume, columns=['normalised_hourly_volume'])
    uniswap_hourly_volume_data = pd.concat([uniswap_hourly_volume_data, normalised_volume], axis=1)

    # Adding MACD and RSI indicators to price data
    uniswap_price_data['MACD'] = ta.trend.MACD(uniswap_price_data['price'], window_slow=240, window_fast=120, window_sign=216).macd()
    uniswap_price_data['RSI'] = ta.momentum.RSIIndicator(uniswap_price_data['price'], window=168).rsi()

    # Differencing and normalizing price data
    uniswap_price_data['price_diff'] = uniswap_price_data['price'].diff()
    uniswap_price_data.dropna(inplace=True)

    scalar_price = StandardScaler()
    uniswap_price_data['price_diff_norm'] = scalar_price.fit_transform(uniswap_price_data[['price_diff']])

    # Applying log transformation to gas cost data
    gas_cost_to_burn_then_mint['log_dollar_gas_cost'] = np.log1p(gas_cost_to_burn_then_mint['dollar_gas_cost'])

    scalar_gas = StandardScaler()
    gas_cost_to_burn_then_mint['log_dollar_gas_cost_norm'] = scalar_gas.fit_transform(gas_cost_to_burn_then_mint[['log_dollar_gas_cost']])

    # Standardizing MACD and RSI indicators
    scalar_ta = StandardScaler()
    uniswap_price_data['MACD_norm'] = scalar_ta.fit_transform(uniswap_price_data[['MACD']])
    uniswap_price_data['RSI_norm'] = scalar_ta.fit_transform(uniswap_price_data[['RSI']])

    return uniswap_hourly_volume_data, uniswap_price_data, gas_cost_to_burn_then_mint


uniswap_hourly_volume_data, uniswap_price_data, gas_cost_to_burn_then_mint = preprocess_data(uniswap_hourly_volume_data, uniswap_price_data, gas_cost_to_burn_then_mint)



# Define the start date
start_date = uniswap_price_data['date'].iloc[0]

# Filter each DataFrame based on the start date
uniswap_price_data = uniswap_price_data[uniswap_price_data['date'] >= start_date].copy()
uniswap_hourly_volume_data = uniswap_hourly_volume_data[uniswap_hourly_volume_data['date'] >= start_date].copy()
gas_cost_to_burn_then_mint = gas_cost_to_burn_then_mint[gas_cost_to_burn_then_mint['date'] >= start_date].copy()


uniswap_price_data.reset_index(inplace=True)
uniswap_hourly_volume_data.reset_index(inplace=True)
gas_cost_to_burn_then_mint.reset_index(inplace=True)


class Normaliser:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def normalise(self, x):
        if self.n < 2:
            return x
        else:
            variance_n = self.M2 / self.n
            variance = variance_n * self.n / (self.n - 1)
            std_dev = math.sqrt(variance) if variance > 0 else 1
            if std_dev == 0:
                return x
            else:
                return (x - self.mean) / std_dev


def standard_deviation(current_time, granular_price_data, steps):





    # Convert current_time to datetime object
    current_time = pd.to_datetime(current_time, unit='s')

    # Calculate the standard deviation
    start_time = current_time - pd.Timedelta(hours=1) - pd.Timedelta(hours=steps)
    end_time = current_time
    subset = granular_price_data.loc[start_time:end_time]
    std_dev = subset.std()

    return std_dev





class UniswapV3Env(gym.Env):
    
    def __init__(self, uniswap_price_data, gas_cost_to_burn_then_mint, uniswap_hourly_volume_data,outside_penalty, held_position_bonus_amount, too_wide_penalty):
        super(UniswapV3Env, self).__init__()
        

        
        self.too_wide_penalty = too_wide_penalty
        self.held_position_bonus_amount = held_position_bonus_amount
        self.outside_penalty = outside_penalty
        
        

        self.amount_invested = 100000
#         self.change_in_initial_investment = 0

        
        self.granular_price_data = granular_price_data

        
        self.previously_outside_range = False
        
        
        self.granular_price_data = granular_price_data
    
    
        self.consecutive_action_zero = 0
    
        self.cumulative_fees = 0
        self.overall_positions_fees = 0
        self.new_position = True
    
    
        # initialise current price, upper and lower bounds so they are accessible outside the step class
        self.current_price = None
        self.lower_bound = None
        self.upper_bound = None
        
        
        
        # variable to track gas prices for calculating the dynamic penalty
        self.gas_prices = deque(maxlen=10)

        
        # a multiplier for doing consecutive 0's to try and make it develop the correct strategy
        self.consecutive_outside_zero = 0
    
        self.step_count = 0
    
        self.impermanent_loss_normaliser = Normaliser()
        self.fees_earned_normaliser = Normaliser()
        self.penalty_normaliser = Normaliser()
        self.reward_normaliser = Normaliser()


    
        
        self.start = True
        self.previous_action = None # update previous action
        self.previous_price = None # update previous price
        self.previous_std_dev = None # update previous std dev



        self.last_bounds = None
        self.initial_amounts = None
        self.initial_amounts_value = None # this is put in so can print SHOULD BE REMOVED LATER AND SO NEED TO MODIFY CODE THAT USES SELF.INITIAL_AMOUNTS_VALUE TO JUST INITIAL_AMOUNTS_VALUE
        self.previous_position_current_amounts_value = None  # SAME WITH THIS NEEDS TO BE REMOVED AND REST OF CODE FIXED
        
        self.uniswap_price_data = uniswap_price_data
        self.uniswap_hourly_volume_data = uniswap_hourly_volume_data
        self.gas_cost_to_burn_then_mint = gas_cost_to_burn_then_mint
        
        
        self.num_bins = 100  # number of bins used for the histogram
        self.std_dev_window = 168  # number of prices for the price window
        self.price_window = 168 # IF END UP REMOVING PRICE WINDOW THEN REMOVE THIS AND ADJUST CURRENT PRICE TO STD_DEV_WINDOW AND ADJUST CODE ACCORDINGLY
        self.largest_ta_window = 240
        
        
#         self.start_time_unix = 1640995200+3600*self.std_dev_window  # initialize start time
#         self.end_time_unix = 1641002400+3600*self.std_dev_window  # initialize end time

        
        
        # Action space and state space
        self.action_space = spaces.Box(low = np.array([0,0]), high = np.array([200,1]), dtype=np.float32)
        self.observation_space = spaces.Dict({ #SHOULD PROBABLY MAKE THESE MORE ACCURATE ALSO NOW THAT HAVE NORMALISED THE DATA THESE NEED TO BE CHANGED
            "current_price_diff_norm": spaces.Box(low=-3, high=3, shape=(1,), dtype=np.float32),
            "current_std_dev_diff_norm": spaces.Box(low=-3, high=3, shape=(1,), dtype=np.float32),
            "liquidity_position_norm": spaces.Box(low=-3, high=3, shape=(2,), dtype=np.float32),
#             "price_window_diff_norm": spaces.Box(low=-8, high=8, shape=(self.price_window,), dtype=np.float32),
            "MACD_norm": spaces.Box(low=-3, high=3, shape=(1,), dtype=np.float32),
            "RSI_norm": spaces.Box(low=-3, high=3, shape=(1,), dtype=np.float32),
            "hourly_volume_norm": spaces.Box(low =-3, high=3, shape=(1,), dtype = np.float32),
            "gas_price_log_norm": spaces.Box(low=-3, high=3, shape=(1,), dtype=np.float32), # FOR NOW THIS IS HIGH THIS SHOULD BE REDUCED WHEN THE GAS PRICE DATA PREPROCESSING IS REFINED
            "unrealised_IL_norm": spaces.Box(low = -5, high=5, shape=(1,), dtype=np.float32), 
            
            "positions_accumulated_fees_norm": spaces.Box(low=-4, high=4, shape=(1,), dtype=np.float32)
            
            #             "liquidity_distribution": spaces.Box(low=0, high=np.inf, shape=(num_bins,), dtype=np.float32)
        })
        
        # Initialise state
        self.state = None
        
        self.current_step = 0
        self.current_reward = 0.0
        self.done = False
        self.info = {}
        
        
    def clip_state_to_space(self, state):
        '''ensure state values are within the bounds stated by the observation_space'''
        for key in state:
            low = self.observation_space[key].low
            high = self.observation_space[key].high
            state[key] = np.clip(np.array(state[key]), low, high)
            
        return state
    
    
        
    def reset(self):
        
        print('Starting reset')
        
        self.amount_invested = 100000
        self.change_in_initial_investment = 0
        
        self.done = False
        self.step_count = 0
        
        self.start = True
        

        self.current_step = self.largest_ta_window
        print('real_current_step', self.current_step)
        fast_forward_time = self.current_step - 1 # we subtract two as the end price is the end of the previous period and so is the start of the current state
        
        # Get the first date from the DataFrame
        first_date = self.uniswap_price_data['date'].iloc[0]
        

        # Convert to Unix time
        unix_time = int(first_date.timestamp())
        print('first_date', unix_time)

        
        self.start_time_unix = unix_time + 3600 * fast_forward_time
        self.end_time_unix = (unix_time +7200) + 3600 * fast_forward_time
    
        # Reset accumulated reward
        self.current_episode_reward = 0.0
        
        initial_action = self.action_space.sample()
        initial_action[1] = round(initial_action[1])

        while initial_action[1] < 0.5:
            initial_action = self.action_space.sample()
            initial_action[1] = round(initial_action[1])


        # Reset state variables
        self.current_price = self.uniswap_price_data['price'][self.current_step]
        
        current_std_dev = standard_deviation(self.end_time_unix, self.granular_price_data['price'], self.std_dev_window)
        
        print("THIS IS THE CURRENT STD:",current_std_dev)
        diff_norm_current_price = self.uniswap_price_data['price_diff_norm'][self.current_step]
        
        diff_norm_current_std_dev = standard_deviation(self.end_time_unix, self.granular_price_data['price_diff_norm'], self.std_dev_window)
        
        MACD_norm = self.uniswap_price_data['MACD_norm'][self.current_step]
        RSI_norm = self.uniswap_price_data['RSI_norm'][self.current_step]
        norm_hourly_volume = self.uniswap_hourly_volume_data['normalised_hourly_volume'][self.current_step]
        norm_log_gas_cost = self.gas_cost_to_burn_then_mint['log_dollar_gas_cost_norm'][self.current_step]

        
        
        
        print('Reset Liquidity Position: Action', initial_action)
        
        # Reset state variables
        self.state = {
            "current_price_diff_norm": diff_norm_current_price,
            "current_std_dev_diff_norm": diff_norm_current_std_dev,
            "liquidity_position_norm": self.calculate_liquidity_position(initial_action, diff_norm_current_price, diff_norm_current_std_dev),
#             "price_window_diff_norm": self.uniswap_price_data['price_diff_norm'][self.current_step-self.price_window:self.current_step],
            "MACD_norm": MACD_norm,
            "RSI_norm": RSI_norm,
            "hourly_volume_norm": norm_hourly_volume,
            "gas_price_log_norm": norm_log_gas_cost,
            "unrealised_IL_norm": 0,
            "positions_accumulated_fees_norm": 0
#             "liquidity_distribution": self.liquidity_distribution_data.iloc[self.current_step]
        }
        

        
        self.previous_action = initial_action
        
        self.previous_std_dev = current_std_dev
        
        self.last_bounds = self.calculate_liquidity_position(initial_action, self.current_price, current_std_dev)
        
        self.state = self.clip_state_to_space(self.state)
        
        self.previous_price = self.current_price
        
        _, amounts = offline_fee_revenue_calculator.calculate_fee_revenue(
                                                                        self.last_bounds[0],  # Lower bound
                                                                        self.last_bounds[1],  # Upper bound
                                                                        self.amount_invested,
                                                                        self.start_time_unix,  # Start time (in UNIX format)
                                                                        self.end_time_unix,
                                                                        swaps_data,
                                                                        CSV_Data_fee_revenue_data)  # End time (in UNIX format)
            
        self.previous_amounts = amounts

        
        print('Ending reset')
        
        # Return the initial state of the environment
        return self.state
        
        
        
    
    def step(self, action):
        
        self.step_count += 1
        print(f"Starting step number: {self.step_count}")
        
        
        
        
        if self.done:
            raise RuntimeError("Episode is done")

        # Update step
        self.current_step += 1
        
        # Update the current price, standard deviation 
        self.current_price = self.uniswap_price_data['price'][self.current_step]
        current_std_dev = standard_deviation(self.end_time_unix, self.granular_price_data['price'], self.std_dev_window)
        diff_norm_current_price = self.uniswap_price_data['price_diff_norm'][self.current_step]
        diff_norm_current_std_dev = standard_deviation(self.end_time_unix, self.granular_price_data['price_diff_norm'], self.std_dev_window)
        
#         price_window_diff_norm = self.uniswap_price_data['price_diff_norm'][self.current_step-self.price_window:self.current_step]
        
        MACD_norm = self.uniswap_price_data['MACD_norm'][self.current_step]
        RSI_norm = self.uniswap_price_data['RSI_norm'][self.current_step]
        norm_hourly_volume = self.uniswap_hourly_volume_data['normalised_hourly_volume'][self.current_step]
        norm_log_gas_cost = self.gas_cost_to_burn_then_mint['log_dollar_gas_cost_norm'][self.current_step]
        
        # Update the current price, standard deviation, and liquidity position
        self.state['current_price_diff_norm'] = diff_norm_current_price
        self.state['current_std_dev_diff_norm'] = diff_norm_current_std_dev
        print('Step Liquidity Position: Action', action)
        self.state['liquidity_position_norm'] = self.calculate_liquidity_position(action, diff_norm_current_price, diff_norm_current_std_dev)
        

        
        self.state['MACD_norm'] = MACD_norm,
        self.state['RSI_norm'] = RSI_norm,
        self.state['hourly_volume_norm'] = norm_hourly_volume,
        
        print('Liquidity Position:', self.calculate_liquidity_position(action, self.current_price, current_std_dev))
        print('Normalised Liquidity Position:', self.state['liquidity_position_norm'])
        print('diff_Normalised current price:', self.state['current_price_diff_norm'])
        
        # Update other state variables
        self.state['gas_price_log_norm'] = norm_log_gas_cost
        # self.state['liquidity_distribution'] = self.liquidity_distribution_data.iloc[self.current_step]
        
        
        
        '''Impermanent Loss calculation'''
        def calculate_impermanent_loss(df, previous_position_start_time, current_time, initial_price, current_price, previous_position_lower_bound,  previous_position_upper_bound, amounts):




            def weighted_average_price(df, start_time, end_time, initial_price, lower_bound, upper_bound):

                # Convert Unix timestamps to datetime objects
                start_time = pd.to_datetime(start_time, unit='s')
                end_time = pd.to_datetime(end_time, unit='s')


                # Select the relevant time period
                df_subset = df.loc[start_time:end_time]

                # Check if the price is within the specified boundaries
                if current_price <= lower_bound:

                    df_subset = df_subset[(df_subset['price'] >= lower_bound) & (df_subset['price'] <= initial_price)]
                    # Calculate the weights
                    weights = df_subset['amount0'] / df_subset['amount0'].sum()
                    weighted_average = (df_subset['price'] * weights).sum()
                    if weighted_average == 0:
                        weighted_average = (initial_price+lower_bound)/2

                elif upper_bound <= current_price:
                    df_subset = df_subset[(df_subset['price'] >= initial_price) & (df_subset['price'] <= upper_bound)]
                    # Calculate the weights
                    weights = df_subset['amount0'] / df_subset['amount0'].sum()
                    weighted_average = (df_subset['price'] * weights).sum()
                    if weighted_average == 0:
                        weighted_average = (initial_price+upper_bound)/2


                return weighted_average
            HODL = (self.previous_amounts[0] + self.previous_amounts[1] * current_price)

            if previous_position_lower_bound < current_price < previous_position_upper_bound:
                k = current_price / initial_price
                impermanent_loss_percent = (((2 * math.sqrt(k)) / (1 + k)) - 1) * (1 / (1 - (((math.sqrt(previous_position_lower_bound / initial_price) + k) * math.sqrt(initial_price / previous_position_upper_bound)) / (1 + k))))
            else:

                if previous_position_upper_bound <= current_price:
                    positions_value = amounts[0] + amounts[1] * weighted_average_price(granular_price_data, start_time=previous_position_start_time, end_time=current_time, initial_price=initial_price, lower_bound=previous_position_lower_bound, upper_bound=previous_position_upper_bound)
                    impermanent_loss_percent = (positions_value - HODL) / HODL
                else:
                    positions_value = (amounts[0] / weighted_average_price(granular_price_data, start_time=previous_position_start_time, end_time=current_time, initial_price=initial_price, lower_bound=previous_position_lower_bound, upper_bound=previous_position_upper_bound) + amounts[1]) * current_price
                    impermanent_loss_percent = (positions_value - HODL) / HODL
                    impermanent_loss_percent = (positions_value - HODL) / HODL


            return impermanent_loss_percent

                

     
        
        impermanent_loss = 0
        norm_impermanent_loss = 0
        
        
        if not self.start:
            




            # calculating change in initial investment from price divergence
            if action[1] >= 0.5:

                
                previous_position_lower_bound, previous_position_upper_bound = self.calculate_liquidity_position(self.previous_action, self.previous_price, self.previous_std_dev)

                end_of_positions_amounts = offline_fee_revenue_calculator.positions_current_amounts(self.previous_price, self.current_price, previous_position_lower_bound, previous_position_upper_bound, self.previous_amounts[0], self.previous_amounts[1])
                print(f'checkthese bits previous_price{self.previous_price},current_price{self.current_price},previous_position_lower_bound{previous_position_lower_bound},previous_position_upper_bound{previous_position_upper_bound}, previous_amounts{self.previous_amounts},')
                print("end_of_positions_amounts", end_of_positions_amounts)
                
                current_position_value = end_of_positions_amounts[0] + end_of_positions_amounts[1]*self.current_price


                self.change_in_initial_investment = current_position_value - self.amount_invested
                print("current_position_value",current_position_value)
                print("amount_invested",self.amount_invested)
                
                self.impermanent_loss_normaliser.update(self.change_in_initial_investment)
                norm_change_in_initial_investment = self.impermanent_loss_normaliser.normalise(self.change_in_initial_investment)
                self.state['unrealised_IL_norm'] = norm_change_in_initial_investment
                print('norm_unrealised_change', norm_change_in_initial_investment)
                
                
                print("change in investment", self.change_in_initial_investment)

                self.amount_invested = current_position_value
                print("amount_invested", self.amount_invested)
                
            else:
                
                
                
                previous_position_lower_bound, previous_position_upper_bound = self.calculate_liquidity_position(self.previous_action, self.previous_price, self.previous_std_dev)
                end_of_positions_amounts = offline_fee_revenue_calculator.positions_current_amounts(self.previous_price, self.current_price, previous_position_lower_bound, previous_position_upper_bound, self.previous_amounts[0], self.previous_amounts[1])
                current_position_value = end_of_positions_amounts[0] + end_of_positions_amounts[1]*self.current_price
                self.change_in_initial_investment = current_position_value - self.amount_invested
                
                # normalise unrealised and realised impermanent loss as they are all feasible for each other
                self.impermanent_loss_normaliser.update(self.change_in_initial_investment)
                norm_change_in_initial_investment = self.impermanent_loss_normaliser.normalise(self.change_in_initial_investment)
                self.state['unrealised_IL_norm'] = norm_change_in_initial_investment
                print('norm_unrealised_change', norm_change_in_initial_investment)
                print("amount_invested", self.amount_invested)
                
                self.change_in_initial_investment = 0
            



        
        '''fees earned calculation'''

        # Calculate the reward
        if action[1] < 0.5:
            self.lower_bound, self.upper_bound = self.last_bounds
            print('Current Liquidity Position:', self.lower_bound, self.upper_bound)
            
            
            
            
            
        else:
            self.lower_bound, self.upper_bound = self.calculate_liquidity_position(action, self.current_price, current_std_dev)
            print('Current Liquidity Position:', self.lower_bound, self.upper_bound)
            self.last_bounds = (self.lower_bound, self.upper_bound) # update last bounds
            self.previous_action = action # update previous action
            print('check this prev action', self.previous_action)
            self.previous_price = self.current_price # update previous price
            self.previous_std_dev = current_std_dev # update previous std dev
            self.previous_position_start_time = self.end_time_unix # update the time the previous position was created
            
            
        self.start=False
        
        print(f"start: {self.start_time_unix}")
        print(f"end: {self.end_time_unix}")
        print(f"current price: {self.current_price}")
        
        
        self.is_price_outside_range()

        
        # now collective fees earned during a position are provided at end of position, aim is that agent will learn
        # to try and create positions and hold them long enough so that the gas fees are covered
        

        
        fees_earned, amounts = offline_fee_revenue_calculator.calculate_fee_revenue(
            self.lower_bound,  # Lower bound
            self.upper_bound,  # Upper bound
            self.amount_invested,
            self.start_time_unix,  # Start time (in UNIX format)
            self.end_time_unix,
            swaps_data,
            CSV_Data_fee_revenue_data)  # End time (in UNIX format)
                      
        if action[1] > 0.5:
            
            self.previous_amounts = amounts # also need to update the position's initial required amounts 
        
      

        
        print("start time check", self.start_time_unix)
        print("end time check", self.end_time_unix)


        if action[1] < 0.5:
            
            self.overall_positions_fees = 0
            self.cumulative_fees += fees_earned  
                
        else:
            
                self.overall_positions_fees = 0
                self.overall_positions_fees = self.cumulative_fees
                self.cumulative_fees = 0
                self.cumulative_fees += fees_earned

        print("positions accumulated fees:", self.cumulative_fees)    
        
        # normalising fees to update the state
        self.fees_earned_normaliser.update(self.cumulative_fees)
        positions_accumulated_fees_norm = self.fees_earned_normaliser.normalise(self.cumulative_fees)
        self.state['positions_accumulated_fees_norm'] = positions_accumulated_fees_norm
            
            

                
        '''gas fee calculation'''

        if action[1] > 0.5:
            gas_fee = self.gas_cost_to_burn_then_mint['dollar_gas_cost'][self.current_step]
            norm_log_gas_fee = self.gas_cost_to_burn_then_mint['log_dollar_gas_cost_norm'][self.current_step]
        else:
            gas_fee = 0
            norm_log_gas_fee = 0
        
    # here we add the impermanent loss as it is currently negative
        reward = self.overall_positions_fees  - gas_fee + impermanent_loss
#         norm_reward = reward
        real_money_earned = self.overall_positions_fees  - gas_fee + impermanent_loss
    
    
        # reinvesting earned fees currently excluding collection gas fee as this is very low
        self.amount_invested += self.overall_positions_fees
    
    
    

        too_wide_penalty = 0
    
    
        if action[0] <= -150 or 150 < action[0]:
            
            
            too_wide_penalty = -self.too_wide_penalty
            
        
        print('too_wide_penalty', too_wide_penalty)
        reward += too_wide_penalty
        print('current_std_dev', current_std_dev)
    
    
    
    
    
        if action[1]>0.5:
            reward+=50
    

    

            
    
    
    
    

    
        print(f"new fees earned: {fees_earned}" ) 
        print(f"overall_previous_positions_fees: {self.overall_positions_fees}" ) 
        print(f"gas fee: -{gas_fee}") 
        print(f"Impermanent Loss: {impermanent_loss}") 


        
        gas_fee_for_penalty = self.gas_cost_to_burn_then_mint['dollar_gas_cost'][self.current_step]
        self.gas_prices.append(gas_fee_for_penalty)
        
        

        
        

        # to discourage agent from doing nothing when price leaves its range
        if self.current_price < self.lower_bound or self.upper_bound < self.current_price:
            self.consecutive_outside_zero += 1
            if len(self.gas_prices) > 0:
                avg_gas_price = sum(self.gas_prices) / len(self.gas_prices)
                penalty = self.outside_penalty * avg_gas_price * (1 + self.consecutive_outside_zero * 1)
                print("average_gas_fee", avg_gas_price)
                
        else:
            self.consecutive_outside_zero = 0
            penalty = 0


        reward -= penalty
        print("penalty:", penalty)

        
        
        # bonus for holding a position which rapidly decays incentivising holding short profitable positions
        
        held_position_bonus = 0
        
        if action[1] > 0.5:
#             self.consecutive_action_zero += 1
#             avg_gas_price = sum(self.gas_prices) / len(self.gas_prices)


            held_position_bonus = self.held_position_bonus_amount * self.cumulative_fees * (1.02 ** self.consecutive_action_zero)
            
            (0.6**(self.consecutive_action_zero))

        else:
            self.consecutive_action_zero = 0
        
        reward += held_position_bonus
        
        print("Reward_received:", reward)
        
        self.reward_normaliser.update(reward)
        norm_reward = self.reward_normaliser.normalise(reward)
    
        print('norm_reward:', norm_reward)
        print("gas_fee_for_avg", gas_fee)
        
        
        
        
        


        
        
            
        
        # Check if the episode is done
        # quicker done version for testing
        if self.current_step >= len(self.gas_cost_to_burn_then_mint) - 1 or self.step_count >= MAX_STEPS or self.amount_invested <=5000:
            self.done = True

        

        # increment the unix time by 3600 seconds (1 hour)
        self.start_time_unix += 3600
        self.end_time_unix += 3600
        
        
        
        print("real money earned:", real_money_earned)
        
        
        print('Ending step')
        
        return self.state, real_money_earned, reward, norm_reward, self.done, {}
    
        
        
        
    

    def is_price_outside_range(self):
        print("outside range current price and bounds:", self.current_price, self.lower_bound, self.upper_bound)
        print("is outside:", self.current_price > self.upper_bound or self.current_price < self.lower_bound)
        return self.current_price > self.upper_bound or self.current_price < self.lower_bound     

        



    '''continuous action space version'''
    def calculate_liquidity_position(self, action, current_price, std_dev):
        # Check if the switch is on (i.e., if the second value in the action array is 1)
        


        
        if action[1] == 1:
            center = current_price
            
            # as the normal distribution could result in 0 this would not actually be possible to we make it represent the minimum
            # width which is 2
            print(action[0])
            

            
            width = (action[0])+2 # action[0])*((1/8)*std_dev
            
            print('center', center)
            print('width', width)
            lower_bound = center - abs(width)
            upper_bound = center + abs(width)

            return np.array([lower_bound, upper_bound])

        else:
           # If the switch is off, keep the current position
            return self.state['liquidity_position_norm']
        



            




# In[ ]:





# In[17]:






# In[ ]:





# In[18]:


# import numpy as np
# np.random.seed(0)
# import tensorflow as tf


# In[19]:


device = torch.device("mps")
print(f"Using device: {device}")


# In[20]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import os


# In[21]:


from tensorflow.python.eager import profiler
from datetime import datetime
import objgraph

import objgraph
import weakref


# In[22]:


# !pip install tf-nightly


# In[ ]:





# In[ ]:





# In[23]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp


# In[24]:


from tensorflow.keras.layers import BatchNormalization
from keras.regularizers import l2
from tensorflow.keras import initializers


# In[25]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions


import torch

def truncated_normal(mean, std, lower_bound, upper_bound, size):
    """
    Generate samples from a truncated normal distribution.
    
    """
    mean = mean.view(-1).expand(*size)
    std = std.view(-1).expand(*size)
    samples = mean + std * torch.randn(*size).to(mean.device)
    
    while True:
        cond = (samples < lower_bound) | (samples > upper_bound)
        if not cond.any():
            break
        resample_indices = cond.nonzero(as_tuple=False).squeeze(0)
        if resample_indices.nelement() > 0:
            samples[resample_indices] = mean[resample_indices] + std[resample_indices] * torch.randn(resample_indices.nelement()).to(mean.device)

    return samples.view(*size)


class PolicyNetwork(nn.Module):
    """
    Neural network representing the policy.
    
    """
    def __init__(self, state_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.continuous_mean = nn.Linear(64, 1)
        self.continuous_std = nn.Linear(64, 1)
        self.discrete_output = nn.Linear(64, 1)

    def forward(self, state):
        state = state.to(device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        continuous_mean = self.continuous_mean(x)
        continuous_std = F.softplus(self.continuous_std(x))
        discrete_action_prob = torch.sigmoid(self.discrete_output(x))
        return continuous_mean, continuous_std, discrete_action_prob


class ValueNetwork(nn.Module):
    """
    Neural network to predict state value.
    
    """
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value_output = nn.Linear(64, 1)

    def forward(self, state):
        state = state.to(device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_output(x)
        return value


class PPOAgent:
    """
    Agent based on Proximal Policy Optimization (PPO) algorithm.
    
    """
    def __init__(self, state_size, action_size, update_epochs=10, lr=0.001, gamma=0.95, 
                 gae_lambda=0.55, entropy_beta=0.01, lr_decay_step=1000, lr_decay_gamma=0.99999):
        # Initialize neural networks and optimizer
        self.policy_network = PolicyNetwork(state_size).to(device)
        self.value_network = ValueNetwork(state_size).to(device)
        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr)
        self.optimizer_value = optim.Adam(self.value_network.parameters(), lr)
        
        # Initialize hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = 0.2
        self.entropy_beta = entropy_beta
        self.update_epochs = update_epochs
        self.lr_decay_step = lr_decay_step
        self.lr_decay_gamma = lr_decay_gamma

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        continuous_mean, continuous_std, discrete_action_prob = self.policy_network(state)

        # Define bounds for truncated normal
        lower_bound = -300.0
        upper_bound = 300.0
        
        continuous_action = truncated_normal(continuous_mean, continuous_std, lower_bound, upper_bound, size=(1,))
        discrete_distribution = distributions.Bernoulli(discrete_action_prob)
        discrete_action = discrete_distribution.sample()

        continuous_log_prob = -torch.log(continuous_std + 1e-8)  
        discrete_log_prob = discrete_distribution.log_prob(discrete_action)
        total_log_prob = continuous_log_prob + discrete_log_prob

        continuous_action = continuous_action.view(1, -1)
        action_tensor = torch.cat([continuous_action, discrete_action], dim=1)
        action_for_env = action_tensor.detach().cpu().numpy().flatten()
        return action_tensor, action_for_env, total_log_prob

    
    def compute_returns_and_advantages(self, rewards, dones, values):
        """
        Compute the returns and advantages
        
        """
        rewards, dones, values = rewards.to(device), dones.to(device), values.to(device)
        
        length = rewards.shape[0]
        returns = torch.empty(length + 1, 1, 1, device=device)
        advantages = torch.empty(length + 1, 1, 1, device=device)
        advantages[-1] = 0  # Initialize last advantage to 0
        not_done = 1 - dones
        next_value = 0

        # Append a dummy value to the end of values tensor
        dummy_value = torch.tensor([0], dtype=torch.float32, device=device).view(1, 1)
        values = torch.cat([values.view(-1, 1), dummy_value], dim=0)

        # Calculate returns and advantages in reverse
        for t in reversed(range(length)):
            returns[t] = rewards[t] + self.gamma * next_value * not_done[t]
            td_error = rewards[t] + self.gamma * values[t + 1] * not_done[t] - values[t]
            advantages[t] = td_error + self.gamma * self.gae_lambda * not_done[t] * advantages[t + 1]
            next_value = values[t]

        # Remove the last dummy value
        returns, advantages = returns[:-1], advantages[:-1]

        return returns, advantages

    def update(self, states, actions, log_probs, returns, advantages):
        """
        Perform policy and value network updates.
        
        """
        states, actions = states.to(device), actions.view(-1, 2).to(device)
        log_probs, returns, advantages = log_probs.clone().detach().view(-1, 1).to(device),returns.clone().detach().view(-1, 1).to(device),advantages.clone().detach().view(-1, 1).to(device)

        for _ in range(self.update_epochs):
            continuous_mean, continuous_std, discrete_action_probs = self.policy_network(states)
            
            # Ensure the standard deviation doesn't approach zero
            continuous_std += 1e-8
            if torch.any(continuous_std <= 0):
                print(f"Warning: non-positive standard deviation encountered: {continuous_std}")

            # Distributions
            continuous_distribution = distributions.Normal(continuous_mean, continuous_std)
            discrete_actions = (discrete_action_probs > 0.5).float()
            
            # Split actions into continuous and discrete
            old_continuous_actions, old_discrete_actions = torch.split(actions, [1, 1], dim=1)

            # New log probabilities
            logprobs_new_continuous = continuous_distribution.log_prob(old_continuous_actions)
            logprobs_new_discrete = old_discrete_actions * torch.log(discrete_action_probs + 1e-10) + (1 - old_discrete_actions) * torch.log(1 - discrete_action_probs + 1e-10)
            logprobs_new = logprobs_new_continuous + torch.sum(logprobs_new_discrete, dim=1, keepdim=True)

            # Proximal Policy Optimization (PPO) losses
            ratio = torch.exp(logprobs_new - log_probs)
            surrogate_loss = ratio * advantages
            clipped_surrogate_loss = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.mean(torch.min(surrogate_loss, clipped_surrogate_loss))

            # Entropy bonus
            entropy = -discrete_action_probs * torch.log(discrete_action_probs + 1e-10) - (1 - discrete_action_probs) * torch.log(1 - discrete_action_probs + 1e-10)
            entropy += continuous_distribution.entropy()
            policy_loss -= self.entropy_beta * torch.mean(entropy)

            # Value network loss
            value_preds = self.value_network(states)
            value_loss = torch.mean((value_preds - returns)**2)

            # Gradient update
            self.optimizer_policy.zero_grad()
            self.optimizer_value.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.optimizer_policy.step()
            self.optimizer_value.step()

        # Decay learning rate
        self.decay_learning_rate()

        return policy_loss.item(), value_loss.item()

    def decay_learning_rate(self):
        """
        Decays the learning rate for policy and value network optimizers.

        The decay is applied using a multiplicative factor and has a minimum threshold.
        """
        new_lr_policy = self.optimizer_policy.param_groups[0]['lr'] * (1 - (1 / self.lr_decay_step))
        new_lr_value = self.optimizer_value.param_groups[0]['lr'] * (1 - (1 / self.lr_decay_step))

        # Setting a minimum learning rate
        self.optimizer_policy.param_groups[0]['lr'] = max(new_lr_policy, 0.00001)
        self.optimizer_value.param_groups[0]['lr'] = max(new_lr_value, 0.00001)

        self.lr_decay_step *= self.lr_decay_gamma

    def load(self, name):
        self.policy_network.load_state_dict(torch.load(name + "_policy.pth"))
        self.value_network.load_state_dict(torch.load(name + "_value.pth"))

    def save(self, name):
        torch.save(self.policy_network.state_dict(), name + "_policy.pth")
        torch.save(self.value_network.state_dict(), name + "_value.pth")



import torch
import numpy as np
import os  # Importing os for the save directory

# Ensure the directory exists
save_dir = "PPO_parameters_0.30%"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def train_agent(agent, env, episodes, batch_size=64, device=torch.device("mps")):
    scores = []


    # Ensure the model is on the same device
    agent.policy_network = agent.policy_network.to(device)
    agent.value_network = agent.value_network.to(device)

    for e in range(episodes):
        state = env.reset()
        state = np.concatenate([v for v in state.values()]).reshape(1, -1)
        
        print("state", state.shape)
        done = False
        score = 0
        total_reward = 0
        states, actions, rewards, dones, next_states, log_probs = [], [], [], [], [], []
        
        for _ in range(MAX_STEPS):
            action_tensor, action_for_env, log_prob = agent.get_action(state)

            next_state, real_money_earned, reward, norm_reward, done, _ = env.step(action_for_env)

            next_state = np.concatenate([np.array(v).flatten() for v in next_state.values()])
            next_state = np.reshape(next_state, [1, agent.state_size])
            
            states.append(torch.tensor(state, dtype=torch.float32).to(device))
            actions.append(action_tensor.clone().detach().to(device))
            rewards.append(torch.tensor(norm_reward, dtype=torch.float32).to(device))
            dones.append(torch.tensor(float(done), dtype=torch.float32).to(device))
            log_probs.append(log_prob.to(device))

            state = next_state
            total_reward += reward
            score += real_money_earned  # increment the score

            if score < -30000:
                done = True
                print("Early stopping, real money fell below -30,000.")

            if len(states) >= batch_size:
                states_torch = torch.stack(states).to(device)
                actions_torch = torch.stack(actions).to(device)
                rewards_torch = torch.stack(rewards).to(device)
                dones_torch = torch.stack(dones).to(device)
                log_probs_torch = torch.stack(log_probs).to(device)

                values = agent.value_network(states_torch)

                returns, advantages = agent.compute_returns_and_advantages(rewards_torch, dones_torch, values)

                policy_loss, value_loss = agent.update(states_torch, actions_torch, log_probs_torch, returns, advantages)
                print(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}")

                states, actions, rewards, dones, log_probs = [], [], [], [], []
                
            if done:
                print(f"Episode: {e+1}/{episodes}, Total reward {total_reward}, Money earned: {score}")
                break
                
        scores.append(score)
        print(f"Episode: {e+1}/{episodes}, Money earned: {score}")

        if e % 100 == 0:
            save_dir = "./"  # Replace with your desired save directory
            model_path = os.path.join(save_dir, f"New_Hope_jupyter_PPO_{e}.pth")
            agent.save(model_path)

    return scores


# In[27]:




env = UniswapV3Env(uniswap_price_data=uniswap_price_data, gas_cost_to_burn_then_mint=gas_cost_to_burn_then_mint, 
                   uniswap_hourly_volume_data=uniswap_hourly_volume_data,
                  outside_penalty=2, held_position_bonus_amount=1, too_wide_penalty=200)

state_size = np.sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
action_size = env.action_space.shape[0]

agent = PPOAgent(state_size, action_size)



# Train the agent with the environment
MAX_STEPS = 2013
EPISODES = 100


scores = train_agent(agent, env, EPISODES)

# Plot the scores
plt.plot(range(len(scores)), scores)
plt.title('Score per episode')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.show()



# In[ ]:


import random

# Hyperparameter ranges
batch_size_options = [32, 64, 128]
held_position_bonus_options = [0.5, 1, 2]
too_wide_penalty_options = [50, 100, 500]
outside_penalty = [1, 3, 5]

lr_options = [0.001, 0.01, 0.1]
update_epochs_options = [5, 15, 25]
lr_decay_step_options = [200, 500, 1000]
lr_decay_gamma_options = [0.99, 0.9999, 0.99999]
gamma = [0.5, 0.99, 0.01]
gae_lambda = [0.9, 0.99, 0.01]
entropy_beta = [0.01, 0.05, 0.01]



# Number of random trials
num_trials = 10

# Store results
results = []


print_every = 2

for trial in range(num_trials):
    # Randomly sample hyperparameters
    batch_size = random.choice(batch_size_options)
    held_position_bonus_amount = random.choice(held_position_bonus_options)
    too_wide_penalty = random.choice(too_wide_penalty_options)
    outside_penalty = random.choice(outside_penalty)
    
    update_epochs = random.choice(update_epochs_options)
    lr = random.choice(lr_options)
    lr_decay_step = random.choice(lr_decay_step_options)
    lr_decay_gamma = random.choice(lr_decay_gamma_options)

    # Set the hyperparameters in your agent/environment
    # This will depend on how you've structured your code
    env = UniswapV3Env(uniswap_price_data=uniswap_price_data, 
                       gas_cost_to_burn_then_mint=gas_cost_to_burn_then_mint, 
                       uniswap_hourly_volume_data=uniswap_hourly_volume_data, outside_penalty=outside_penalty, 
                      held_position_bonus_amount=held_position_bonus_amount, too_wide_penalty=too_wide_penalty)
    
    state_size = np.sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
    action_size = env.action_space.shape[0]
    
    agent = PPOAgent(state_size, action_size, update_epochs, lr, lr_decay_step, lr_decay_gamma) 
    
    
        # Load the saved model here
    saved_model_path = "/Users/joshuahayes-powell/Documents/Dissertation/PPO_parameters_0.30%/New_Hope_jupyter_PPO_100.pth"  # Make sure the extension is correct
    agent.load(saved_model_path)





    # Train the agent
    scores = train_agent(agent, env, episodes=100, batch_size=batch_size)

    # Evaluate the agent (this could be the average score, final score, etc.)
    evaluation_metric = np.mean(scores)

    # Store the results
    results.append({
        'batch_size': batch_size,
        'held_position_bonus': held_position_bonus,
        'too_wide_penalty': too_wide_penalty,
        'learning_rate': learning_rate,
        'evaluation_metric': evaluation_metric
    })
    
    
    # Print the best trial so far
    if (trial + 1) % print_every == 0:
        best_trial_so_far = max(results, key=lambda x: x['evaluation_metric'])
        print(f"Best trial after {trial + 1} trials: {best_trial_so_far}")


# Find the best hyperparameters based on your evaluation metric
best_trial = max(results, key=lambda x: x['evaluation_metric'])
print("Best trial:", best_trial)

