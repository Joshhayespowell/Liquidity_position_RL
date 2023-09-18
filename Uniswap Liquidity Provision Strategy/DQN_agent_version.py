
import gym
from gym import spaces
import numpy as np
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
import optuna



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
start_date = pd.Timestamp("2022-04-01 21:00:00")

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



len(uniswap_price_data_test)



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


len(uniswap_hourly_volume_data) - 1

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
    
    def __init__(self, uniswap_price_data, gas_cost_to_burn_then_mint, uniswap_hourly_volume_data, outside_penalty, held_position_bonus_amount):
        super(UniswapV3Env, self).__init__()

        # Constants and general settings
        self.held_position_bonus_amount = held_position_bonus_amount
        self.outside_penalty = outside_penalty
        self.amount_invested = 10000
        self.granular_price_data = granular_price_data
        self.uniswap_price_data = uniswap_price_data
        self.uniswap_hourly_volume_data = uniswap_hourly_volume_data
        self.gas_cost_to_burn_then_mint = gas_cost_to_burn_then_mint

        # Trackers for actions and fees
        self.consecutive_action_zero = 0
        self.cumulative_fees = 0
        self.overall_positions_fees = 0
        self.new_position = True
        self.consecutive_outside_zero = 0

        # Current status variables
        self.current_price = None
        self.lower_bound = None
        self.upper_bound = None

        # Gas prices deque to track last 10 gas prices for dynamic penalty calculation
        self.gas_prices = deque(maxlen=10)

        # Counters and flags
        self.step_count = 0
        self.start = True
        self.previous_action = None
        self.previous_price = None
        self.previous_std_dev = None
        self.last_bounds = None
        self.initial_amounts = None
        self.previous_position_current_amounts_value = None

        # window related constants
        self.std_dev_window = 168
        self.price_window = 168
        self.largest_ta_window = 240

        # Normalisers for rewards and metrics
        self.impermanent_loss_normaliser = Normaliser()
        self.fees_earned_normaliser = Normaliser()
        self.penalty_normaliser = Normaliser()
        self.reward_normaliser = Normaliser()

        # Action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict({
            "current_price_diff_norm": spaces.Box(low=-6, high=6, shape=(1,), dtype=np.float32),
            "current_std_dev_diff_norm": spaces.Box(low=-6, high=6, shape=(1,), dtype=np.float32),
            "liquidity_position_norm": spaces.Box(low=-6, high=6, shape=(2,), dtype=np.float32),
            "MACD_norm": spaces.Box(low=-6, high=6, shape=(1,), dtype=np.float32),
            "RSI_norm": spaces.Box(low=-6, high=6, shape=(1,), dtype=np.float32),
            "hourly_volume_norm": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "gas_price_log_norm": spaces.Box(low=-6, high=6, shape=(1,), dtype=np.float32),
            "unrealised_IL_norm": spaces.Box(low=-6, high=6, shape=(1,), dtype=np.float32),
            "positions_accumulated_fees_norm": spaces.Box(low=-6, high=6, shape=(1,), dtype=np.float32)
        })

        # Current state and related information
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
    

    def reset(self, validation_mode=False):
        """
        Resets the environment to its initial state and returns an initial observation.
        """

        # Initialization messages
        print('Starting reset')

        # Reset basic properties
        self.amount_invested = 10000
        self.change_in_initial_investment = 0
        self.done = False
        self.step_count = 0
        self.start = True
        self.current_episode_reward = 0.0

        # Set current step and calculate initial time properties
        self.current_step = self.largest_ta_window
        fast_forward_time = self.current_step - 1
        first_date = self.uniswap_price_data['date'].iloc[0]  # Get the first date from the DataFrame
        unix_time = int(first_date.timestamp())
        self.start_time_unix = unix_time + 3600 * fast_forward_time
        self.end_time_unix = (unix_time + 7200) + 3600 * fast_forward_time

        # Set original liquidity position
        initial_action = self.action_space.sample()
        while initial_action == 0:
            # Avoid action 0 being chosen first
            initial_action = self.action_space.sample()

        # Compute various data points and derive state variables
        self.current_price = self.uniswap_price_data['price'][self.current_step]
        current_std_dev = standard_deviation(self.end_time_unix, self.granular_price_data['price'], self.std_dev_window)
        diff_norm_current_price = self.uniswap_price_data['price_diff_norm'][self.current_step]
        diff_norm_current_std_dev = standard_deviation(self.end_time_unix, self.granular_price_data['price_diff_norm'], self.std_dev_window)
        MACD_norm = self.uniswap_price_data['MACD_norm'][self.current_step]
        RSI_norm = self.uniswap_price_data['RSI_norm'][self.current_step]
        norm_hourly_volume = self.uniswap_hourly_volume_data['normalised_hourly_volume'][self.current_step]
        norm_log_gas_cost = self.gas_cost_to_burn_then_mint['log_dollar_gas_cost_norm'][self.current_step]

        # Construct the state dictionary
        self.state = {
            "current_price_diff_norm": diff_norm_current_price,
            "current_std_dev_diff_norm": diff_norm_current_std_dev,
            "liquidity_position_norm": self.calculate_liquidity_position(initial_action, diff_norm_current_price, diff_norm_current_std_dev),
            "MACD_norm": MACD_norm,
            "RSI_norm": RSI_norm,
            "hourly_volume_norm": norm_hourly_volume,
            "gas_price_log_norm": norm_log_gas_cost,
            "unrealised_IL_norm": 0,
            "positions_accumulated_fees_norm": 0
        }

        # Update boundaries and clip state
        self.last_bounds = self.calculate_liquidity_position(initial_action, self.current_price, current_std_dev)
        self.state = self.clip_state_to_space(self.state)

        # Compute offline fee revenue
        _, amounts = offline_fee_revenue_calculator.calculate_fee_revenue(
            self.last_bounds[0],  # Lower bound
            self.last_bounds[1],  # Upper bound
            self.amount_invested,
            self.start_time_unix,  # Start time (in UNIX format)
            self.end_time_unix,
            swaps_data,
            CSV_Data_fee_revenue_data)  # End time (in UNIX format)
        self.previous_amounts = amounts

        # Return the initial state of the environment
        return self.state, initial_action


    
    def step(self, action):
        """
        Execute one step in the environment.
        """

        # Increase the step count and provide an update
        self.step_count += 1
        print(f"Starting step number: {self.step_count}")

        # Check if the episode is finished
        if self.done:
            raise RuntimeError("Episode is done")

        # Move to the next step
        self.current_step += 1

        # Update the current price and related metrics
        self.current_price = self.uniswap_price_data['price'][self.current_step]
        current_std_dev = standard_deviation(self.end_time_unix, self.granular_price_data['price'], self.std_dev_window)
        diff_norm_current_price = self.uniswap_price_data['price_diff_norm'][self.current_step]
        diff_norm_current_std_dev = standard_deviation(self.end_time_unix, self.granular_price_data['price_diff_norm'], self.std_dev_window)

        # Update technical indicators
        MACD_norm = self.uniswap_price_data['MACD_norm'][self.current_step]
        RSI_norm = self.uniswap_price_data['RSI_norm'][self.current_step]
        norm_hourly_volume = self.uniswap_hourly_volume_data['normalised_hourly_volume'][self.current_step]
        norm_log_gas_cost = self.gas_cost_to_burn_then_mint['log_dollar_gas_cost_norm'][self.current_step]

        # Update state variables
        self.state['current_price_diff_norm'] = diff_norm_current_price
        self.state['current_std_dev_diff_norm'] = diff_norm_current_std_dev
        print('Step Liquidity Position: Action', action)
        self.state['liquidity_position_norm'] = self.calculate_liquidity_position(action, diff_norm_current_price, diff_norm_current_std_dev)
        self.state['MACD_norm'] = MACD_norm
        self.state['RSI_norm'] = RSI_norm
        self.state['hourly_volume_norm'] = norm_hourly_volume
        self.state['gas_price_log_norm'] = norm_log_gas_cost

        # Print state information
        print('Normalised Liquidity Position:', self.state['liquidity_position_norm'])
        print('diff_Normalised current price:', self.state['current_price_diff_norm'])

        
        '''Impermanent Loss calculation'''
        def calculate_impermanent_loss(df, previous_position_start_time, current_time, initial_price, current_price, previous_position_lower_bound,  previous_position_upper_bound, amounts):


            def weighted_average_price(df, start_time, end_time, initial_price, lower_bound, upper_bound):
                """Calculate the weighted average price."""
                # Convert Unix timestamps to datetime objects
                start_time = pd.to_datetime(start_time, unit='s')
                end_time = pd.to_datetime(end_time, unit='s')

                # Filter the data for the relevant time period
                df_subset = df.loc[start_time:end_time]

                # Determine weighted average based on price boundaries
                if current_price <= lower_bound:
                    df_subset = df_subset[(df_subset['price'] >= lower_bound) & (df_subset['price'] <= initial_price)]
                    weights = df_subset['amount0'] / df_subset['amount0'].sum()
                    weighted_avg = (df_subset['price'] * weights).sum()
                    weighted_avg = weighted_avg or (initial_price+lower_bound)/2
                elif upper_bound <= current_price:
                    df_subset = df_subset[(df_subset['price'] >= initial_price) & (df_subset['price'] <= upper_bound)]
                    weights = df_subset['amount0'] / df_subset['amount0'].sum()
                    weighted_avg = (df_subset['price'] * weights).sum()
                    weighted_avg = weighted_avg or (initial_price+upper_bound)/2
                return weighted_avg

            # Calculate the HODL value
            HODL = self.previous_amounts[0] + self.previous_amounts[1] * current_price

            # Calculate impermanent loss based on price boundaries and conditions
            if previous_position_lower_bound < current_price < previous_position_upper_bound:
                k = current_price / initial_price
                impermanent_loss_percent = (
                    (2 * math.sqrt(k)) / (1 + k) - 1
                ) * (
                    1 / (1 - (((math.sqrt(previous_position_lower_bound / initial_price) + k) * 
                               math.sqrt(initial_price / previous_position_upper_bound)) / (1 + k)))
                )
            else:
                if previous_position_upper_bound <= current_price:
                    positions_value = (
                        amounts[0] + 
                        amounts[1] * weighted_average_price(df, previous_position_start_time, current_time, initial_price, previous_position_lower_bound, previous_position_upper_bound)
                    )
                else:
                    weighted_avg_price = weighted_average_price(df, previous_position_start_time, current_time, initial_price, previous_position_lower_bound, previous_position_upper_bound)
                    positions_value = (amounts[0] / weighted_avg_price + amounts[1]) * current_price
                impermanent_loss_percent = (positions_value - HODL) / HODL

            return impermanent_loss_percent

        
        
        
        impermanent_loss = 0
        norm_impermanent_loss = 0
        

        if not self.start:
            # Calculate common variables once to avoid redundancy
            previous_position_lower_bound, previous_position_upper_bound = self.calculate_liquidity_position(self.previous_action, self.previous_price, self.previous_std_dev)
            end_of_positions_amounts = offline_fee_revenue_calculator.positions_current_amounts(self.previous_price, self.current_price, previous_position_lower_bound, previous_position_upper_bound, self.previous_amounts[0], self.previous_amounts[1])
            current_position_value = end_of_positions_amounts[0] + end_of_positions_amounts[1] * self.current_price

            # Calculate impermanent loss based on action
            if action == 0:
                impermanent_loss = 0
                unrealised_impermanent_loss = self.amount_invested * calculate_impermanent_loss(granular_price_data, self.previous_position_start_time, self.end_time_unix, self.previous_price, self.current_price, previous_position_lower_bound, previous_position_upper_bound, self.previous_amounts)
                self.impermanent_loss_normaliser.update(unrealised_impermanent_loss)
                norm_unrealised_impermanent_loss = self.impermanent_loss_normaliser.normalise(unrealised_impermanent_loss)
                self.state['unrealised_IL_norm'] = norm_unrealised_impermanent_loss
                self.change_in_initial_investment = 0
            else:
                impermanent_loss = self.amount_invested * calculate_impermanent_loss(granular_price_data, self.previous_position_start_time, self.end_time_unix, self.previous_price, self.current_price, previous_position_lower_bound, previous_position_upper_bound, self.previous_amounts)
                self.impermanent_loss_normaliser.update(impermanent_loss)
                norm_impermanent_loss = self.impermanent_loss_normaliser.normalise(impermanent_loss)
                self.state['unrealised_IL_norm'] = norm_impermanent_loss
                self.change_in_initial_investment = current_position_value - self.amount_invested
                self.amount_invested = current_position_value

        
        '''fees earned calculation'''

        # Gas Fee Calculation
        if action != 0:
            gas_fee = self.gas_cost_to_burn_then_mint['dollar_gas_cost'][self.current_step]
            norm_log_gas_fee = self.gas_cost_to_burn_then_mint['log_dollar_gas_cost_norm'][self.current_step]
        else:
            gas_fee = 0
            norm_log_gas_fee = 0

        # Calculate reward and actual money earned
        # Reward factors in impermanent loss and gas fees
        reward = self.overall_positions_fees - gas_fee + impermanent_loss
        real_money_earned = reward

        # Logging investment and various fees
        print('amount_invested', self.amount_invested)
        print(f"new fees earned: {fees_earned}")
        print(f"overall_previous_positions_fees: {self.overall_positions_fees}")
        print(f"gas fee: -{gas_fee}")
        print(f"Impermanent Loss: {impermanent_loss}")

        # Track gas fees for penalties
        gas_fee_for_penalty = self.gas_cost_to_burn_then_mint['dollar_gas_cost'][self.current_step]
        self.gas_prices.append(gas_fee_for_penalty)
        
        # Add penalty to reward if action is 0 and fee_earned is 0 
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

        # Calculate bonus for holding a position
        held_position_bonus = 0
        if action == 0:
            self.consecutive_action_zero += 1
            avg_gas_price = sum(self.gas_prices) / len(self.gas_prices)
            print("bonus test avg gas price")
            # Bonus decays with consecutive actions of doing nothing
            held_position_bonus = self.held_position_bonus_amount * avg_gas_price * (0.95 ** self.consecutive_action_zero)
            print("held position bonus:", held_position_bonus)
        else:
            self.consecutive_action_zero = 0

        reward += held_position_bonus
        print("Reward_received:", reward)

        # Normalize the reward
        self.reward_normaliser.update(reward)
        norm_reward = self.reward_normaliser.normalise(reward)
        print('norm_reward:', norm_reward)
        print("gas_fee_for_avg", gas_fee)

        # End the episode if certain conditions are met
        if self.current_step >= len(self.uniswap_price_data) - 1 or self.step_count >= MAX_STEPS or self.amount_invested <= 3000:
            self.done = True

        # Move forward in time by 1 hour (3600 seconds)
        self.start_time_unix += 3600
        self.end_time_unix += 3600

        print("real money earned:", real_money_earned)
        print('Ending step')

        return self.state, real_money_earned, reward, norm_reward, self.done, {}


    def is_price_outside_range(self):
        print("outside range current price and bounds:", self.current_price, self.lower_bound, self.upper_bound)
        print("is outside:", self.current_price > self.upper_bound or self.current_price < self.lower_bound)
        return self.current_price > self.upper_bound or self.current_price < self.lower_bound     



    '''smaller action space version'''
    def calculate_liquidity_position(self, action, current_price, std_dev):
        
        # calculate the segment boundaries
        segments = {
            'S-10': current_price - 3 * std_dev,
            'S-9': current_price - 2.7 * std_dev,
            'S-8': current_price - 2.4 * std_dev,
            'S-7': current_price - 2.1 * std_dev,
            'S-6': current_price - 1.8 * std_dev,
            'S-5': current_price - 1.5 * std_dev,
            'S-4': current_price - 1.2 * std_dev,
            'S-3': current_price - 0.9 * std_dev,
            'S-2': current_price - 0.6 * std_dev,
            'S-1': current_price - 0.3 * std_dev,
            'S0': current_price,
            'S1': current_price + 0.3 * std_dev,
            'S2': current_price + 0.6 * std_dev,
            'S3': current_price + 0.9 * std_dev,
            'S4': current_price + 1.2 * std_dev,
            'S5': current_price + 1.5 * std_dev,
            'S6': current_price + 1.8 * std_dev,
            'S7': current_price + 2.1 * std_dev,
            'S8': current_price + 2.4 * std_dev,
            'S9': current_price + 2.7 * std_dev,
            'S10': current_price + 3 * std_dev,
        }

        # map action to segment
        segment_mapping = {
            0: None,  # Do Nothing
            1: ('S-10', 'S10'),
            2: ('S-6', 'S6'),
            3: ('S-2', 'S2'),
            4: ('S-5', 'S10'),
            5: ('S-10', 'S+5'),  

        }
        

        if action in segment_mapping:
            if segment_mapping[action] is None:
                return self.state['liquidity_position_norm'] # this does nothing so keeps the liquidity position the same
            else:
                
                return np.array([segments[segment_mapping[action][0]], segments[segment_mapping[action][1]]])
        else:
            raise ValueError

device = torch.device("mps")
print(f"Using device: {device}")

class Net(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=50):
        super(Net, self).__init__()

        # Define the neural network layers
        self.fc_initial = nn.Linear(state_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, action_size)

        self.to(device)

    def forward(self, x):
        # Feed-forward through the layers
        x = F.relu(self.fc_initial(x))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size, env, learning_rate=0.001, gamma=0.93, epsilon=1.0, epsilon_decay=0.998, update_every=100, epsilon_min=0.01):
        # Initialization parameters for the agent
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.update_every = update_every
        self.t_step = 0

        # Initialize the models and optimizer
        self.model = Net(state_size, action_size).to(device)  
        self.target_model = Net(state_size, action_size).to(device)  
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.update_target_model()

        # Other variables
        self.memory = deque(maxlen=150000)
        self.previously_outside_range = False
        self.env = env
        self.previous_epsilon = self.epsilon

    def remember(self, state, action, reward, next_state, done):
        # Store experiences in the replay buffer
        self.memory.append((
            torch.FloatTensor(state).to(device),
            torch.tensor(action).to(device),
            torch.tensor(reward, dtype=torch.float32).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.tensor(done).to(device)
        ))

    def act(self, state):
        self.model.eval()  # Set to evaluation mode
        with torch.no_grad():
            if np.random.rand() <= self.epsilon:
                return random.randint(0, self.action_size - 1)
            q_values = self.model(torch.FloatTensor(state).to(device))
            return np.argmax(q_values.cpu().numpy())

    def replay(self, batch_size=32):
        # Train the network using experiences from the replay buffer
        self.model.train()  # Set to training mode
        
        minibatch = random.sample(self.memory, batch_size)
        # Unpack experiences into tensors
        states = torch.stack([exp[0] for exp in minibatch]).squeeze(1).to(device)
        actions = torch.tensor([exp[1] for exp in minibatch]).unsqueeze(-1).to(device)
        rewards = torch.tensor([exp[2] for exp in minibatch]).unsqueeze(-1).to(device)
        next_states = torch.stack([exp[3] for exp in minibatch]).squeeze(1).to(device)
        dones = torch.tensor([exp[4] for exp in minibatch]).unsqueeze(-1).to(device)

        # Calculate target Q-values
        with torch.no_grad():
            best_actions = self.model(next_states).argmax(1).unsqueeze(-1)
            target_values = self.target_model(next_states).gather(1, best_actions)
            targets = rewards + (self.gamma * target_values * (1 - dones.float()))
        
        # Calculate current Q-values
        current_values = self.model(states).gather(1, actions)
        loss = self.loss_fn(current_values, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        self.optimizer.step()

        # Epsilon decay logic
        if self.epsilon > self.epsilon_min:
            if not self.previously_outside_range and self.env.is_price_outside_range():
                self.previous_epsilon = self.epsilon
                self.epsilon *= 5
                self.previously_outside_range = True
            else:
                self.epsilon = self.previous_epsilon * self.epsilon_decay
                self.previously_outside_range = False

    def update_target_model(self):
        # Update the target model with weights from the local model
        if self.t_step % self.update_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=device))

    def save(self, name):
        torch.save(self.model.state_dict(), name)


# Ensure the directory exists
save_dir = "testing/new_price_data_DQN_FNN_parameters_0.30%"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
warm_up_length = 50


def train_agent(agent, env, episodes, batch_size, validation_mode, starting_episode=0):
    scores = []
    for e in range(starting_episode, episodes):
        

        print("Before reset: ")
        state_dict, initial_action = env.reset(validation_mode)

        state = np.concatenate([v for v in state_dict.values()]).reshape(1, -1)
        print("After reset: ")
        
        done = False
        score = 0  # Initialize score
        total_reward = 0
        
        agent.t_step = 0
        
        while not done:



            
            action = agent.act(state)
            next_state, real_money_earned, reward, norm_reward, done, _ = env.step(action)

            print("Action taken:", action)
            print("real_money_earned:", real_money_earned)
            print("Norm Reward received:", norm_reward)
            print("CHECK THIS REWARD:", reward)

            next_state = np.concatenate([np.array(v).flatten() for v in next_state.values()])
            next_state = np.reshape(next_state, [1, agent.state_size])
            next_state = torch.FloatTensor(next_state)  # convert next_state to tensor and move to device
            
                        # Only remember and learn from the experience if we're past the warm-up period
            if agent.t_step >= warm_up_length:
                agent.remember(state, action, norm_reward, next_state, done)
                
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)  
                    
                    
            agent.t_step += 1
            

            state = next_state
            total_reward += reward
            score += real_money_earned  # increment the score
            
            if score < -500000:
                done = True
                print("Early stopping, real money fell below -30,000.")
                
            
            if done:
                
                
                agent.update_target_model()
                scores.append(score)
                print(f"Episode: {e+1}/{episodes},Total reward {total_reward}, Money earned: {score}, Epsilon: {agent.epsilon:.2}")
                
                break
                
        if e % 50 == 0:
            model_path = os.path.join(save_dir, f"Institutional_New_Price_data_jupyter_dqn_FFN_{e}.pth")
            agent.save(model_path)
    return scores

env = UniswapV3Env(uniswap_price_data=uniswap_price_data, gas_cost_to_burn_then_mint=gas_cost_to_burn_then_mint, 
                   uniswap_hourly_volume_data=uniswap_hourly_volume_data, 
                   outside_penalty=2, held_position_bonus_amount=1.7)

state_size = np.sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size, env)

# Train the agent with the environment
MAX_STEPS = 5000
EPISODES = 110

# set a small number for testing
validation_mode = False
BATCH_SIZE = 32
scores = train_agent(agent, env, EPISODES, BATCH_SIZE, validation_mode)

'''turn on for validtion'''


uniswap_price_data = uniswap_price_data_validate.reset_index(drop=True)
uniswap_hourly_volume_data = uniswap_hourly_volume_data_validate.reset_index(drop=True)
gas_cost_to_burn_then_mint = gas_cost_to_burn_then_mint_validate.reset_index(drop=True)


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


print('optuna time')

MAX_STEPS = 5260


def objective(trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.1, log=True)
    gamma = trial.suggest_float("gamma", 0.5, 0.99, step=0.01)
    epsilon = trial.suggest_float("epsilon", 0.1, 1.0, step=0.1)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 0.999, step=0.001)
    epsilon_min = trial.suggest_float("epsilon_min", 0.01, 0.03, step=0.001)
    outside_penalty = trial.suggest_float('held_position_bonus_options',1, 5, step=0.3)    
    held_position_bonus_options = trial.suggest_float('held_position_bonus_options',0.5, 2, step=0.1)

    
    env = UniswapV3Env(uniswap_price_data=uniswap_price_data, 
                       gas_cost_to_burn_then_mint=gas_cost_to_burn_then_mint, 
                       uniswap_hourly_volume_data=uniswap_hourly_volume_data,outside_penalty=outside_penalty, 
                       held_position_bonus_amount=held_position_bonus_options)
    
    
    state_size = np.sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size, env, 
                     learning_rate=learning_rate, 
                     gamma=gamma, epsilon=epsilon, 
                     epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    
    # Load the saved model here
    saved_model_path = "/Users/joshuahayes-powell/Documents/Dissertation/testing/new_price_data_DQN_FNN_parameters_0.05%/New_Price_data_NO_HODL_jupyter_dqn_FFN_100.pth"
    agent.load(saved_model_path)
    
    BATCH_SIZE = 32
    EPISODES = 50
    scores = train_agent(agent, env, EPISODES, BATCH_SIZE, validation_mode=True)
    # return the final performance metric that you're interested in
    return scores[-1]


def print_callback(study, trial):
    if trial.number % 50 == 0:
        print(f"After trial {trial.number}, Best params so far: {study.best_params}")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, callbacks=[print_callback])

# Results
print(f"The optimized hyperparameters are {study.best_params}")
print(f"The optimized score is {study.best_value}")


# In[ ]:


'''turn on for training'''


uniswap_price_data = uniswap_price_data_test.reset_index(drop=True)
uniswap_hourly_volume_data = uniswap_hourly_volume_data_test.reset_index(drop=True)
gas_cost_to_burn_then_mint = gas_cost_to_burn_then_mint_test.reset_index(drop=True)


uniswap_hourly_volume_data, uniswap_price_data, gas_cost_to_burn_then_mint = preprocess_data(uniswap_hourly_volume_data, uniswap_price_data, gas_cost_to_burn_then_mint)


# In[ ]:



# Define the start date
start_date = uniswap_price_data['date'].iloc[0]

# Filter each DataFrame based on the start date
uniswap_price_data = uniswap_price_data[uniswap_price_data['date'] >= start_date].copy()
uniswap_hourly_volume_data = uniswap_hourly_volume_data[uniswap_hourly_volume_data['date'] >= start_date].copy()
gas_cost_to_burn_then_mint = gas_cost_to_burn_then_mint[gas_cost_to_burn_then_mint['date'] >= start_date].copy()


uniswap_price_data.reset_index(inplace=True)
uniswap_hourly_volume_data.reset_index(inplace=True)
gas_cost_to_burn_then_mint.reset_index(inplace=True)


# In[ ]:


import matplotlib.pyplot as plt

def test_agent(agent, env, episodes, starting_episode=0):
    scores = []
    money_earned_over_steps = []  # to store money earned over steps for plotting
    
    # Set epsilon to its minimum value to favor exploitation over exploration
    original_epsilon = agent.epsilon
    agent.epsilon = agent.epsilon_min

    for e in range(starting_episode, episodes):
        state_dict, _ = env.reset(False)
        state = np.concatenate([v for v in state_dict.values()]).reshape(1, -1)
        
        done = False
        score = 0  # Initialize score
        stepwise_money = []  # to store money earned at each step
        
        while not done:
            action = agent.act(state)
            next_state, real_money_earned, _, _, done, _ = env.step(action)
            next_state = np.concatenate([np.array(v).flatten() for v in next_state.values()])
            next_state = np.reshape(next_state, [1, agent.state_size])
            
            state = next_state
            score += real_money_earned  # increment the score
            
            stepwise_money.append(score)  # append money earned at this step

            if done:
                scores.append(score)
                money_earned_over_steps.append(stepwise_money)
                print(f"Episode: {e+1}/{episodes}, Money earned: {score}")
                break
                
    # Restore the original epsilon value
    agent.epsilon = original_epsilon
    
    # Plotting
    for idx, money in enumerate(money_earned_over_steps):
        plt.plot(money, label=f'Episode {idx+1}')
    plt.title('Money Earned Over Steps')
    plt.xlabel('Step')
    plt.ylabel('Money Earned')
    plt.legend()
    plt.show()

    return scores


# In[ ]:


# Load your best performing model here (after validation)
best_model_path = "/Users/joshuahayes-powell/Documents/Dissertation/testing/new_price_data_DQN_FNN_parameters_0.05%/New_Price_data_NO_HODL_jupyter_dqn_FFN_5.pth"
agent.load(best_model_path)

# Initialize the environment with your testing data
env = UniswapV3Env(uniswap_price_data=uniswap_price_data, gas_cost_to_burn_then_mint=gas_cost_to_burn_then_mint, 
                   uniswap_hourly_volume_data=uniswap_hourly_volume_data, 
                   outside_penalty=2, held_position_bonus_amount=1.7)

# Run the test
EPISODES_TO_TEST = 10  # Or however many episodes you wish to test over
test_scores = test_agent(agent, env, EPISODES_TO_TEST)
