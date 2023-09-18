import json
import requests
from requests.exceptions import ChunkedEncodingError

url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"

# Define the base query with placeholders for timestamp_gte
query = """
query($timestamp_gte: Int!) {
  poolHourDatas(
    first: 1000
    orderBy: periodStartUnix
    orderDirection: asc
    where: {
      pool:"0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8" #0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640 <-0.05 | 0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8 <-0.30
      periodStartUnix_gte: $timestamp_gte
      periodStartUnix_lt: 1690844400
    }
  ) {
    periodStartUnix
    liquidity
    high
    low
    pool {
      token0 {
        decimals
      }
      token1 {
        decimals
      }
    }
    close
    feeGrowthGlobal0X128
    feeGrowthGlobal1X128
  }
}
"""

headers = {"Content-Type": "application/json"}

# timestamp_gte = 1640995200  # 2023-01-01 00:00
timestamp_gte = 0
timestamp_lt = 1690844400  # 2023-08-01 00:00


# Variables to track the last data and repetition count
last_data = None
data_repetition_count = 0

# Open the file in append mode
with open("0.30_fee_revenue_data.json", "a") as file:
    max_retries = 3

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
                    poolHourDatas = result["data"].get("poolHourDatas", [])

                    if poolHourDatas:
                        for poolHourData in poolHourDatas:
                            file.write(json.dumps(poolHourData) + "\n")

                        # Check if the last data is the same as the previous last data
                        if last_data == poolHourDatas[-1]:
                            data_repetition_count += 1
                        else:
                            last_data = poolHourDatas[-1]
                            timestamp_gte = int(last_data["periodStartUnix"])
                            data_repetition_count = 0  # Reset the repetition count

                        if data_repetition_count >= 10:
                            print("Repeated data 10 times. Breaking.")
                            break

                    if timestamp_gte >= timestamp_lt:
                        break
                else:
                    print("Error:", response.status_code)
                    break

                break  # Successfully executed, exit the inner loop
            except ChunkedEncodingError as e:
                print(f"ChunkedEncodingError occurred: {e}")
                retry_count += 1
                print(f"Retrying... (attempt {retry_count})")

        if retry_count >= max_retries or data_repetition_count >= 10:
            print("Max retries reached or data repeated too many times. Exiting.")
            break

        if poolHourDatas and timestamp_gte >= timestamp_lt and data_repetition_count < 10:
            break

print("Data saved to file.")