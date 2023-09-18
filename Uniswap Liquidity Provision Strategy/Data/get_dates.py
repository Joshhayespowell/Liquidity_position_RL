import datetime

date_string = "2022-06-20 02:00"
dt = datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M")
timestamp = dt.timestamp()

print(int(timestamp))





timestamp = 1640995200
dt = datetime.datetime.fromtimestamp(timestamp)
formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")

print(formatted_date)



# 0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8 0.3

# 0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640 0.05