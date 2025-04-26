# -*- coding = utf-8 -*-
# @Time: 2025/4/13 14:20
# @Author: Zhihang Yi
# @File: __init__.py.py
# @Software: PyCharm

from datetime import datetime, timedelta

# time range to obtain data from a year from today to today
end = datetime.now()
start = end - timedelta(days=365)

end_str = end.strftime("%Y-%m-%d")
start_str = start.strftime("%Y-%m-%d")

# time interval to obtain the data
interval_str = '1h'

# proportion of training set to validation set
proportion = 5

# extent to send buy / sell signals
extent = 0.5 * (10 ** -2)

dimension = 28
