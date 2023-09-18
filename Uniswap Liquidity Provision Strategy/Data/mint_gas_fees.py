import json
import requests
from requests.exceptions import ChunkedEncodingError

url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"

# Define the base query with placeholders for timestamp_gte
query = """
query($timestamp_gte: BigInt!) {
  mints(
    first: 1000
    orderBy: timestamp
    orderDirection: asc
    where: {
      pool:"0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640" #0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640 <-0.05 | 0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8
      timestamp_gte: $timestamp_gte
      timestamp_lt: 1690844400 # 2023-01-01 00:00
    }
  ) {
    transaction{
      blockNumber
      timestamp
      gasUsed
      gasPrice
      mints{
        id
      }
    }
  }
}
"""

headers = {"Content-Type": "application/json"}

timestamp_gte = 0  # 2023-01-01 00:00
timestamp_lt = 1640997885  # 2023-08-01 00:00

# Open the file in append mode
with open("0.05_mints_gas_fee_data.json", "a") as file:
    max_retries = 3
    last_mint = None
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
                    mints = result["data"].get("mints", [])

                    if not mints:
                        break  # No more results

                    # Write the JSON data to the file
                    for mint in mints:
                        file.write(json.dumps(mint) + "\n")

                    if last_mint and last_mint == mints[-1]:
                        repetition_count += 1
                        if repetition_count >= 10:
                            break  # The same last transaction was repeated 10 times
                    else:
                        repetition_count = 0  # Reset the counter if the last transaction is not the same

                    last_mint = mints[-1]
                    last_timestamp = int(last_mint["transaction"]["timestamp"])

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

        if not mints or last_timestamp >= timestamp_lt:
            break

print("Data saved to file.")
