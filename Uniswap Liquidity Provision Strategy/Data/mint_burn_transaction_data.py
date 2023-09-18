import json
import requests
from requests.exceptions import ChunkedEncodingError

url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"

mint_query = """
query($timestamp_gte: BigInt!) {
  mints(
    first: 1000
    orderBy: timestamp
    orderDirection: asc
    where: {
      pool: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640" #0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640 <-0.05 | 0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8 <-0.30
      timestamp_gte: $timestamp_gte
      timestamp_lt: 1690844400
    }
  ) {
    id
    owner
    origin
    timestamp
    amount
    amount0
    amount1
    amountUSD
    tickLower
    tickUpper
  }
}
"""

burn_query = """
query($timestamp_gte: BigInt!) {
  burns(
    first: 1000
    orderBy: timestamp
    orderDirection: asc
    where: {
      pool: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640" #0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640 <-0.05
      timestamp_gte: $timestamp_gte
      timestamp_lt: 1690844400
    }
  ) {
    id
    owner
    origin
    timestamp
    amount
    amount0
    amount1
    amountUSD
    tickLower
    tickUpper
  }
}
"""

headers = {"Content-Type": "application/json"}

mint_timestamp_gte = 0
burn_timestamp_gte = 0
timestamp_gte = 1640995200  # 2023-01-01 00:00
timestamp_lt = 1690844400  # 2023-08-01 00:00


# Open the file in append mode
with open("0.05_mints_data.json", "a") as mints_file, open("0.05_burns_data.json", "a") as burns_file:
    max_retries = 3

            # Variables to track the last transaction and repetition count
    last_mint = None
    last_burn = None
    mint_repetition_count = 0
    burn_repetition_count = 0

    while True:
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Fetch mint transactions
                data = {"query": mint_query, "variables": {"timestamp_gte": mint_timestamp_gte}}
                response = requests.post(url, headers=headers, json=data)
                result = json.loads(response.text)

                if "errors" in result:
                    print("Error in mint response:", result)
                    break

                if response.status_code == 200:
                    print(result)
                    mints = result["data"].get("mints", [])

                    if mints:
                        for mint in mints:
                            mints_file.write(json.dumps(mint) + "\n")

                        # Check if the last mint is the same as the previous last mint
                        if last_mint == mints[-1]:
                            mint_repetition_count += 1
                        else:
                            last_mint = mints[-1]
                            mint_timestamp_gte = int(last_mint["timestamp"])
                            mint_repetition_count = 0  # Reset the repetition count

                        if mint_repetition_count >= 10:
                            print("Repeated data 10 times. Breaking.")
                            break

                    if mint_timestamp_gte >= timestamp_lt:
                        break
                else:
                    print("Error:", response.status_code)
                    break

                # Fetch burn transactions
                data = {"query": burn_query, "variables": {"timestamp_gte": burn_timestamp_gte}}
                response = requests.post(url, headers=headers, json=data)
                result = json.loads(response.text)

                if "errors" in result:
                    print("Error in burn response:", result)
                    break

                if response.status_code == 200:
                    print(result)
                    burns = result["data"].get("burns", [])

                    if burns:
                        for burn in burns:
                            burns_file.write(json.dumps(burn) + "\n")

                        # Check if the last burn is the same as the previous last burn
                        if last_burn == burns[-1]:
                            burn_repetition_count += 1
                        else:
                            last_burn = burns[-1]
                            burn_timestamp_gte = int(last_burn["timestamp"])
                            burn_repetition_count = 0  # Reset the repetition count

                        if burn_repetition_count >= 10:
                            print("Repeated data 10 times. Breaking.")
                            break

                    if burn_timestamp_gte >= timestamp_lt:
                        break
                else:
                    print("Error:", response.status_code)
                    break

                break  # Successfully executed, exit the inner loop
            except ChunkedEncodingError as e:
                print(f"ChunkedEncodingError occurred: {e}")
                retry_count += 1
                print(f"Retrying... (attempt {retry_count})")

        if retry_count >= max_retries or mint_repetition_count >= 10 or burn_repetition_count >= 10:
            print("Max retries reached or data repeated too many times. Exiting.")
            break

        if (mints and mint_timestamp_gte >= timestamp_lt and mint_repetition_count < 10) and (burns and burn_timestamp_gte >= timestamp_lt and burn_repetition_count < 10):
            break

    print("Data saved to file.")