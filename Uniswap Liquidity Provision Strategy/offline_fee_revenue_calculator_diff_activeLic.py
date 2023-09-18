from gql import gql, Client
import pandas as pd
import numpy as np


def CSV_Data_df(start_date, end_date, CSV_Data_fee_revenue_data):
    # Load the provided fee revenue data into a variable
    CSV_data = CSV_Data_fee_revenue_data

    # Convert all the columns in the data to numeric, setting non-numeric values to NaN
    CSV_data = CSV_data.apply(pd.to_numeric, errors = 'coerce')

    # Reverse the order of rows in the data
    CSV_data = CSV_data[::-1]

    # Filter the data to only include rows within the provided start and end dates
    CSV_data = CSV_data[(CSV_data['periodStartUnix'] >= start_date) & (CSV_data['periodStartUnix']<=end_date)]

    # Reset the index of the filtered data
    CSV_data = CSV_data.reset_index()

    # Return the processed data
    return CSV_data





def get_amount_ETH(sqrt_lower_bound,sqrt_upper_bound,liquidity,decimals):
    '''
    Calculate the amount of ETH based on the given parameters.
    '''
    # Swap bounds if they are in the wrong order
    if (sqrt_lower_bound > sqrt_upper_bound):
        (sqrt_lower_bound,sqrt_upper_bound) = (sqrt_upper_bound,sqrt_lower_bound)

    # Compute the amount of ETH
    amount_ETH = ((liquidity*2**96*(sqrt_upper_bound-sqrt_lower_bound)/sqrt_upper_bound/sqrt_lower_bound)/10**decimals)
    
    return amount_ETH

def get_amount_USDC(sqrt_lower_bound,sqrt_upper_bound,liquidity,decimals):
    '''
    Calculate the amount of USDC based on the given parameters.
    '''
    # Swap bounds if they are in the wrong order
    if (sqrt_lower_bound > sqrt_upper_bound):
        (sqrt_lower_bound,sqrt_upper_bound) = (sqrt_upper_bound,sqrt_lower_bound)

    # Compute the amount of USDC
    amount_USDC = liquidity*(sqrt_upper_bound-sqrt_lower_bound)/2**96/10**decimals
    
    return amount_USDC

def get_amounts(current_price,price_lower_bound,price_upper_bound,liquidity,decimal_USDC,decimal_ETH):
    '''
    Compute the amounts of ETH and USDC based on the given parameters.
    '''
    # Convert price values to their square root representation
    sqrt_current_price = (np.sqrt(current_price*10**(decimal_ETH-decimal_USDC)))*(2**96)
    sqrt_lower_bound = np.sqrt(price_lower_bound*10**(decimal_ETH-decimal_USDC))*(2**96)
    sqrt_upper_bound = np.sqrt(price_upper_bound*10**(decimal_ETH-decimal_USDC))*(2**96)

    # Swap bounds if they are in the wrong order
    if (sqrt_lower_bound > sqrt_upper_bound):
        (sqrt_lower_bound,sqrt_upper_bound) = (sqrt_upper_bound,sqrt_lower_bound)

    # Calculate amounts based on where the current price lies in relation to the bounds
    if sqrt_current_price <= sqrt_lower_bound:
        amount_ETH = get_amount_ETH(sqrt_lower_bound,sqrt_upper_bound,liquidity,decimal_USDC)
        return 0, amount_ETH

    elif sqrt_current_price < sqrt_upper_bound and sqrt_current_price > sqrt_lower_bound:
        amount_ETH = get_amount_ETH(sqrt_current_price,sqrt_upper_bound,liquidity,decimal_USDC)
        amount_USDC = get_amount_USDC(sqrt_lower_bound,sqrt_current_price,liquidity,decimal_ETH)
        return amount_USDC, amount_ETH

    else:
        amount_USDC = get_amount_USDC(sqrt_lower_bound,sqrt_upper_bound,liquidity,decimal_ETH)
        return amount_USDC, 0 





def get_liquidity_USDC(sqrt_lower_bound, sqrt_upper_bound, amount_ETH, decimals):
    '''
    Calculate the required USDC liquidity based on the bounds and amount of ETH.
    '''
    # Ensure the lower bound is less than the upper bound
    if sqrt_lower_bound > sqrt_upper_bound:
        sqrt_lower_bound, sqrt_upper_bound = sqrt_upper_bound, sqrt_lower_bound

    # Compute the USDC liquidity
    liquidity = amount_ETH / ((2**96 * (sqrt_upper_bound - sqrt_lower_bound) / sqrt_upper_bound / sqrt_lower_bound) / 10**decimals)
    return liquidity

def get_liquidity_ETH(sqrt_lower_bound, sqrt_upper_bound, amount_USDC, decimals):
    '''
    Calculate the required ETH liquidity based on the bounds and amount of USDC.
    '''
    # Ensure the lower bound is less than the upper bound
    if sqrt_lower_bound > sqrt_upper_bound:
        sqrt_lower_bound, sqrt_upper_bound = sqrt_upper_bound, sqrt_lower_bound

    # Compute the ETH liquidity
    liquidity = amount_USDC / ((sqrt_upper_bound - sqrt_lower_bound) / 2**96 / 10**decimals)
    return liquidity

def get_liquidity(current_price, price_lower_bound, price_upper_bound, amount_USDC, amount_ETH, decimal_USDC, decimal_ETH):
    '''
    Determine the liquidity based on the current price, bounds, and amounts of USDC and ETH.
    '''
    # Convert prices to square root representations
    sqrt_current_price = (np.sqrt(current_price * 10**(decimal_ETH - decimal_USDC))) * (2**96)
    sqrt_lower_bound = np.sqrt(price_lower_bound * 10**(decimal_ETH - decimal_USDC)) * (2**96)
    sqrt_upper_bound = np.sqrt(price_upper_bound * 10**(decimal_ETH - decimal_USDC)) * (2**96)

    # Ensure the lower bound is less than the upper bound
    if sqrt_lower_bound > sqrt_upper_bound:
        sqrt_lower_bound, sqrt_upper_bound = sqrt_upper_bound, sqrt_lower_bound

    # Calculate the required liquidity based on where the current price is in relation to the bounds
    if sqrt_current_price <= sqrt_lower_bound:
        liquidity_USDC = get_liquidity_USDC(sqrt_lower_bound, sqrt_upper_bound, amount_ETH, decimal_USDC)
        return liquidity_USDC
    elif sqrt_current_price < sqrt_upper_bound and sqrt_current_price > sqrt_lower_bound:
        liquidity_USDC = get_liquidity_USDC(sqrt_current_price, sqrt_upper_bound, amount_ETH, decimal_USDC)
        liquidity_ETH = get_liquidity_ETH(sqrt_lower_bound, sqrt_current_price, amount_USDC, decimal_ETH)
        
        # Return the lower of the two liquidity values for best representation
        liquidity = liquidity_USDC if liquidity_USDC < liquidity_ETH else liquidity_ETH
        return liquidity
    else:
        liquidity_ETH = get_liquidity_ETH(sqrt_lower_bound, sqrt_upper_bound, amount_USDC, decimal_ETH)
        return liquidity_ETH




def output(CSV_data):
    '''
    Function to process and output relevant information from the data frame
    '''

    # Calculate the value of fees
    CSV_data['feeValue']= (CSV_data['myfee0'])+ (CSV_data['myfee1']*CSV_data['close'])
    # Calculate the value of amounts
    CSV_data['amountValue']= (CSV_data['amount0']) + (CSV_data['amount1']*CSV_data['close'])

    # Calculate the value of fee growth
    CSV_data['feeGrowthValue']= (CSV_data['fee_for_USDC'])+ (CSV_data['fee_for_ETH']*CSV_data['close'])
    # Convert the 'periodStartUnix' to a datetime format
    CSV_data['date']=pd.to_datetime(CSV_data['periodStartUnix'], unit='s')

    # Select only the relevant columns from the data frame
    data=CSV_data[['date', 'myfee0', 'myfee1', 'feeGrowthValue','feeValue', 'amountValue', 'amount0', 'amount1', 'close']]
    # Fill any NA or NaN values with 0
    data=data.fillna(0)

    # Print the fees earned
    fees_earned = data['feeValue'].iloc[0]

    return fees_earned



def value_traded_in_range(end_timestamp, swaps_data, lower_bound, upper_bound):
    '''
    This function calculates what percentage of the value was traded in a chosen range,
    within a specific one-hour window.
    '''

    df = swaps_data.copy()  

    # Initialize epsilon (a very small constant)
    epsilon = 1e-10

    # Define the start timestamp as one hour before the end timestamp.
    start_timestamp = end_timestamp - 3600  #3600 = 1hour

    # Format timestamps as strings for using them in DataFrame's loc method.
    hour_start = pd.to_datetime(start_timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')
    hour_end = pd.to_datetime(end_timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')

    # Select the data within this date range.
    selected_data = df.loc[hour_start:hour_end].copy()

    # Calculate the ratio of amount_USDC to amount_ETH for each transaction in the selected hour.
    selected_data['ratio'] = abs(selected_data['amount0'] / selected_data['amount1'])

    # Calculate total inward amounts (i.e., negative amounts represent inward trades).
    total_inward_amount_USDC = selected_data.loc[selected_data['amount0'] < 0, 'amount0'].sum() * -1
    total_inward_amount_ETH = (selected_data[selected_data['amount1'] < 0]['amount1'] * selected_data['ratio']).sum() * -1


    # Select the trades where the ratio is within the specified range.
    ratio_selected_data = selected_data[(abs(selected_data['ratio']) >= lower_bound) & (abs(selected_data['ratio']) <= upper_bound)]

    # Calculate inward amounts in the selected ratio range.
    in_range_inward_amount_USDC = ratio_selected_data.loc[ratio_selected_data['amount0'] < 0, 'amount0'].sum() * -1
    in_range_inward_amount_ETH = (ratio_selected_data[ratio_selected_data['amount1'] < 0]['amount1'] * ratio_selected_data['ratio']).sum() * -1


    # Calculate the percentage of value traded in the range.
    percentage_value_of_in_range_trades = (
        (total_inward_amount_USDC / (total_inward_amount_USDC + total_inward_amount_ETH + epsilon)) * 
        (in_range_inward_amount_USDC / (total_inward_amount_USDC + epsilon)) +
        (total_inward_amount_ETH / (total_inward_amount_USDC + total_inward_amount_ETH + epsilon)) * 
        (in_range_inward_amount_ETH / (total_inward_amount_ETH + epsilon))
    )

    # Return the calculated percentage.
    return percentage_value_of_in_range_trades





def calculate_fee_revenue(liquidity_lower_bound, liquidity_upper_bound, total_amount_to_invest, period_start_date, period_end_date, swaps_data, CSV_Data_fee_revenue_data):

    
    


    # Define the Unix timestamps for the start and end of the time period of interest.
    # The specific time period chosen here is from 2022-01-01 00:00:00 to 2022-01-01 02:00:00.
    start_date = period_start_date  
    end_date = period_end_date    


    CSV_data = CSV_Data_df(start_date, end_date, CSV_Data_fee_revenue_data)

    # The 'decimals' field in the pool data specifies the precision of each token's value.
    # Retrieve these decimal values for both tokens.
    decimal_USDC = CSV_data.iloc[0]['pool.token0.decimals']  # Decimal precision for token0
    decimal_ETH = CSV_data.iloc[0]['pool.token1.decimals']  # Decimal precision for token1
    
    # Calculate the difference in decimal precision between the two tokens.
    decimal = decimal_ETH - decimal_USDC
    
    # Compute the fee growth for each token.
    # 'feeGrowthGlobal0X128' and 'feeGrowthGlobal1X128' are fields retrieved from the CSV_Data_dfQL API.
    # They represent the accumulated fees for each token, scaled by 2^128 to maintain precision.
    # Dividing these fields by 2^128 and then by 10^decimals gives the fee growth in terms of the actual token amounts.
    CSV_data['feeGrowth0'] = ((CSV_data['feeGrowthGlobal0X128']) / (2**128)) / (10**decimal_USDC)
    CSV_data['feeGrowth1'] = ((CSV_data['feeGrowthGlobal1X128']) / (2**128)) / (10**decimal_ETH)

    # Define the range (in terms of token price ratio) in which to invest liquidity.
    upper_bound = liquidity_upper_bound
    lower_bound = liquidity_lower_bound

    # Define the total amount to be invested.
    total_amount_to_invest = total_amount_to_invest

    # Calculate the fee earned by an unbound unit of liquidity within the specified hour.
    # 'feeGrowth0_previous' and 'feeGrowth1_previous' represent the fee growth values from the previous hour.
    CSV_data['feeGrowth0_previous'] = CSV_data['feeGrowth0'].iloc[1]
    CSV_data['feeGrowth1_previous'] = CSV_data['feeGrowth0'].iloc[1]

    # Calculate the fee difference between the current and previous hour for both tokens.
    # These fields represent the actual fees earned by a liquidity provider in the pool.
    CSV_data['fee_for_USDC'] = CSV_data['feeGrowth0'].iloc[0] - CSV_data['feeGrowth0'].iloc[-1]
    CSV_data['fee_for_ETH'] = CSV_data['feeGrowth1'].iloc[0] - CSV_data['feeGrowth1'].iloc[-1]

    
    # Compute the square roots of the upper and lower price bounds, scaled by the token's decimal precision.
    # These square roots represent the price bounds in the "sqrtPrice" format used in Uniswap V3.
    sqrt_upper_bound = np.sqrt(upper_bound * 10 ** (decimal))  
    sqrt_lower_bound = np.sqrt(lower_bound * 10 ** (decimal)) 

    # Compute the square root of the closing price of the last transaction, scaled by the token's decimal precision.
    # This represents the current price in the "sqrtPrice" format.
    sqrt_current_close = np.sqrt(CSV_data['close'].iloc[0]* 10 ** (decimal))

    # Copy the closing prices to a new column 'price0'.
    CSV_data['price0'] = CSV_data['close']

    # Depending on the current price 'sqrt_current_close', calculate the liquidity 'deltaL' to be provided, 
    # and the resulting amounts of token0 'amount_USDC' and token1 'amount_ETH' that would be added to the pool.

    # Case 1: Current price is within the bounds
    if sqrt_current_close > sqrt_lower_bound and sqrt_current_close < sqrt_upper_bound:
        # The formula for 'deltaL' comes from the liquidity equation in Uniswap V3, rearranged to solve for liquidity.
        # This calculates how much liquidity to add to the pool to invest the desired amount at the current price.
        deltaL = -total_amount_to_invest * sqrt_upper_bound/ (-2*sqrt_current_close*sqrt_upper_bound+(CSV_data['price0'].iloc[0]* 10 ** (decimal))+sqrt_lower_bound*sqrt_upper_bound)
        # Calculate the amounts of token0 and token1 to be added to the pool.
        amount_USDC = deltaL * (sqrt_current_close - sqrt_lower_bound) 
        amount_ETH = deltaL * ((1 / sqrt_current_close) - (1 / sqrt_upper_bound)) * 10 ** (decimal)  

    # Case 2: Current price is below the lower bound
    elif sqrt_current_close < sqrt_lower_bound:
        deltaL = total_amount_to_invest / (((1 / sqrt_lower_bound) - (1 / sqrt_upper_bound)) * (CSV_data['price0'].iloc[0]))
        # If the current price is below the lower bound, all of the investment would be in token1.
        amount_USDC = 0  
        amount_ETH = total_amount_to_invest / (CSV_data['price0'].iloc[0])

    # Case 3: Current price is above the upper bound
    else:
        deltaL = total_amount_to_invest / (sqrt_upper_bound - sqrt_lower_bound) 
        # If the current price is above the upper bound, all of the investment would be in token0.
        amount_USDC = total_amount_to_invest
        amount_ETH = 0  

    position_liquidity = get_liquidity(CSV_data['price0'].iloc[0],lower_bound,upper_bound,amount_USDC,amount_ETH,decimal_USDC,decimal_ETH)

    # due to rounding errors and slight discrepency in square root calculations we re-calculate the current amounts with the get_amounts function so that its constant with future amounts calculated through the get_amount function 
    amounts = get_amounts(CSV_data['price0'].iloc[0],lower_bound,upper_bound,position_liquidity,decimal_USDC,decimal_ETH)

    Amounts = amounts[0], amounts[1]

    ActiveLiquidity = 0
    CSV_data['amount0'] = 0
    CSV_data['amount1'] = 0
    
    ActiveLiquidity = value_traded_in_range(end_date, swaps_data, liquidity_lower_bound, liquidity_upper_bound)

    CSV_data['myfee0'] = CSV_data['fee_for_USDC'] * position_liquidity * ActiveLiquidity
    CSV_data['myfee1'] = CSV_data['fee_for_ETH'] * position_liquidity * ActiveLiquidity

    fees_earned = output(CSV_data)
    
    return fees_earned, Amounts

def positions_current_amounts(previous_price, current_price, lower_bound, upper_bound, previous_amount_USDC, previous_amount_ETH):


    # The 'decimals' field in the pool data specifies the precision of each token's value.
    # Retrieve these decimal values for both tokens.
    decimal_USDC = 6  # Decimal precision for token0
    decimal_ETH = 18  # Decimal precision for token1

    position_liquidity = get_liquidity(previous_price,lower_bound, upper_bound,previous_amount_USDC,previous_amount_ETH,decimal_USDC,decimal_ETH)

    amounts = get_amounts(current_price,lower_bound, upper_bound,position_liquidity,decimal_USDC,decimal_ETH)

    return amounts

