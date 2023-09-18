import json
import requests

url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"

query = """
query {
  poolHourDatas(
    first: 1000
    orderBy: periodStartUnix
    orderDirection: asc
    where: {
      pool: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640" #0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640 <-0.05 | 0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8 <-0.30
      periodStartUnix_gte: 1690840800
      periodStartUnix_lt: 1690844400
    }
  ) {
    periodStartUnix
    feesUSD
    tvlUSD
    close
  }
}
"""
timestamp_gte = 0
timestamp_lt = 1690844400  # 2023-08-01 00:00


headers = {"Content-Type": "application/json"}

# Initialize pool_hour_datas to avoid NameError
pool_hour_datas = []

# Make the API request
data = {"query": query}
response = requests.post(url, headers=headers, json=data)
result = json.loads(response.text)

# Debug prints
print("Data being sent:", json.dumps(data))
print("Response:", response.text)

if "errors" in result:
    print("GraphQL Errors:", result["errors"])
else:
    if response.status_code == 200:
        pool_hour_datas = result["data"].get("poolHourDatas", [])

        # Open the file in append mode
        with open("0.05_pool_hour_data.json", "a") as file:
            # Write the JSON data to the file
            for pool_hour_data in pool_hour_datas:
                file.write(json.dumps(pool_hour_data) + "\n")

        print("Data saved to file.")
    else:
        print("Error:", response.status_code)
