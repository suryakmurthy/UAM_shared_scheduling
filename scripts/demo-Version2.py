'''
#@Time      :5/19/22 08:17
#@Author    :Chelsea Li
#@File      :Scenario_gen.py
#@Software  :PyCharm
'''

import numpy as np
from random import expovariate

f=open("version2.scn","w")

f.write("00:00:00.00>TRAILS ON \n")
f.write("\n")
# f.write("00:00:00.00>CIRCLE a, 32.7783,-96.8046 0.2 \n")
# f.write("00:00:00.00>CIRCLE b, 33.1559,-96.872 0.2 \n")
f.write("0:00:00.00>PAN 32.77173371	-96.83678249 \n")
f.write("0:00:00.00>ZOOM 2 \n")
f.write("00:00:00.00>TAXI OFF 4\n")
# f.write("0:00:00.00>FF\n")
f.write("\n")



# Generate the demand based on poisson distribution
# simulate the 'number' events in a poisson process with an arrival rate(lambda_x) of 'interval' arrivals per second
def generate_interval(interval,number):
    # seed(100)
    lambda_x = 1/interval
    ac_demand_interval = [int(expovariate(lambda_x)) for i in range(number)]
    depart_time = np.cumsum(ac_demand_interval)
    depart_time_ori = depart_time.copy()
    return depart_time,depart_time_ori


ORIG_list = ['K','R','Q','L','F','G']
ROUTE = {"K":20, "R":16, "Q":10,"L":16,"F":14,"G":20}

n= 10
inv = (12*3600)/58
AC_nums = [n,n,n,n,n,n]   #number of total flights each route
AC_intervals =[(12*3600)/102,(12*3600)/150,(12*3600)/90,(12*3600)/75,(12*3600)/200,inv]  # arrival rate pre second for each route

### Generate departure time for six routes under exponential distribution
dep_time = []
dep_time_ori = []
dep_time_dict = {}
for iters in range(len(AC_nums)):
    dep_time.append([])
    dep_time_ori.append([])


for i in range(len(AC_nums)):
    route_id=ORIG_list[i]
    dep_time[i],dep_time_ori[i] = generate_interval(AC_intervals[i],AC_nums[i])
    curr_dep = dep_time[i]

    for idx in range(len(curr_dep)):
        aircraft_No = route_id + str(idx)
        dep_time_dict[aircraft_No] = curr_dep[idx]
dep_dict_ordered = dict(sorted(dep_time_dict.items(), key=lambda item: item[1]))
# print(dep_dict_ordered)

###  generate scn demo
for item in dep_dict_ordered.items():
    route_id = item[0][0]
    route_length = ROUTE[route_id]
    depart_time = item[1]
    plane = "P" + item[0]
    time = "00:00:" + str(depart_time) + ".00"
    f.write(time + ">CRE " + plane + ",Mavic," + route_id + "1,0,0" + "\n")
    # f.write(time+">LISTRTE "+plane+"\n")
    f.write(time + ">ORIG " + plane + " " + route_id + "1\n")
    f.write(time + ">DEST " + plane + " " + route_id + str(route_length) + "\n")
    f.write(time + ">SPD " + plane + " 30" + "\n")
    f.write(time + ">ALT " + plane + " 400" + "\n")
    for j in range(1, route_length + 1):
        wpt = route_id + str(j)
        f.write(time + ">ADDWPT " + plane + " " + wpt + " 400 40" + "\n")
    f.write(time + ">" + plane + " VNAV on \n")
    f.write("\n")

f.close()