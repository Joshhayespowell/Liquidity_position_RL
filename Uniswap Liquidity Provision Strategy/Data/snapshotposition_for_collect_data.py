import json
import requests
from requests.exceptions import ChunkedEncodingError

url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"

# Define the new query for positionSnapshots with timestamp_gte and timestamp_lt
query = """
query($timestamp_gte: BigInt!, $timestamp_lt: BigInt!) {
  positionSnapshots(
    first: 1000
    where: {
      pool: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640" #0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640 <-0.05 | 0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8 <-0.30
      timestamp_gte: $timestamp_gte
      timestamp_lt: $timestamp_lt
    }
  ) {
    owner
    timestamp
    collectedFeesToken0
    collectedFeesToken1
  }
}
"""

headers = {"Content-Type": "application/json"}

timestamp_gte = 0  # 2022-01-01 00:00
timestamp_lt = 1690844400  # 2023-01-01 00:00

# Open the file in append mode
with open("0.05_position_snapshots_data.json", "a") as file:
    max_retries = 3

    while True:
        retry_count = 0

        while retry_count < max_retries:
            try:
                data = {"query": query, "variables": {"timestamp_gte": timestamp_gte, "timestamp_lt": timestamp_lt}}
                response = requests.post(url, headers=headers, json=data)
                result = json.loads(response.text)

                if "errors" in result:
                    print("Error in response:", result)
                    break

                if response.status_code == 200:
                    print(result)
                    position_snapshots = result["data"].get("positionSnapshots", [])

                    if not position_snapshots:
                        break  # No more results

                    # Write the JSON data to the file
                    for snapshot in position_snapshots:
                        file.write(json.dumps(snapshot) + "\n")

                    last_snapshot = position_snapshots[-1]
                    last_timestamp = int(last_snapshot["timestamp"])

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

        if retry_count >= max_retries:
            print("Max retries reached. Exiting.")
            break

        if not position_snapshots or last_timestamp >= timestamp_lt:
            break

print("Data saved to file.")
