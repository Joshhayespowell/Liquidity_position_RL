import json
import requests
from requests.exceptions import ChunkedEncodingError

url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"

# Define the base query with placeholders for timestamp_gte
query = """
query($timestamp_gte: BigInt!) {
  swaps(
    first: 1000
    orderBy: timestamp
    orderDirection: asc
    where: {
      pool:"0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
      timestamp_gte: $timestamp_gte
      timestamp_lt: 1690844400
    }
  ) {
    transaction{
      timestamp
      gasUsed
      gasPrice
    }
  }
}
"""

headers = {"Content-Type": "application/json"}

timestamp_gte = 0  # Initialize as you like, just as an example
timestamp_lt = 1690844400  # Initialize as you like, just as an example

# Open the file in append mode
with open("0.30_ERC20_gas_fee_data.json", "a") as file:
    max_retries = 3
    last_swap = None  # Changed from last_mint to last_swap
    repetition_count = 0

    while True:
        retry_count = 0

        while retry_count < max_retries:
            try:
                data = {"query": query, "variables": {"timestamp_gte": timestamp_gte}}
                response = requests.post(url, headers=headers, json=data)
                result = json.loads(response.text)

                if "errors" in result:
                    print("Error in response:", result)
                    break

                if response.status_code == 200:
                    print(result)
                    swaps = result["data"].get("swaps", [])  # Changed from mints to swaps

                    if not swaps:
                        break  # No more results

                    # Write the JSON data to the file
                    for swap in swaps:  # Changed from mint to swap
                        file.write(json.dumps(swap) + "\n")

                    if last_swap and last_swap == swaps[-1]:  # Changed from last_mint to last_swap
                        repetition_count += 1
                        if repetition_count >= 10:
                            break  # The same last transaction was repeated 10 times
                    else:
                        repetition_count = 0  # Reset the counter if the last transaction is not the same

                    last_swap = swaps[-1]  # Changed from last_mint to last_swap
                    last_timestamp = int(last_swap["transaction"]["timestamp"])  # Changed from last_mint to last_swap

                    if last_timestamp < timestamp_lt:
                        timestamp_gte = last_timestamp
                    else:
                        break
                else:
                    print("Error:", response.status_code)
                    break

                break  # Successfully executed, exit the inner loop
            except ChunkedEncodingError as e:
                print(f"ChunkedEncodingError occurred: {e}")
                retry_count += 1
                print(f"Retrying... (attempt {retry_count})")

            if repetition_count >= 10:
                break  # The same last transaction was repeated 10 times

        if retry_count >= max_retries or repetition_count >= 10:
            print("Max retries reached or data repeated too many times. Exiting.")
            break

        if not swaps or last_timestamp >= timestamp_lt:  # Changed from mints to swaps
            break

print("Data saved to file.")
