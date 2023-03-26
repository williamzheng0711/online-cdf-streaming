import numpy as np 
import matplotlib.pyplot as pyplot 
  
# useful_Mbps_18 = [5.7698, 12.1930, 14.08764, 15.42866, 15.9156, 16.4002]
# max_Mbps_18    = [17.823771, 17.948281, 17.852223, 17.81206, 17.79159, 17.77351]
# avg_Delay_18 = [1.46847, 2.09748, 2.805153, 3.62868, 4.49, 5.36468]
# defective_rate_18 = [0.054722, 0.054268, 0.05222, 0.0523256, 0.051849, 0.0495977]


# useful_Mbps_15 = [5.176910, 10.899363, 13.0132,  13.7327, 13.83467, 13.825932]
# max_Mbps_15    = [14.43862, 14.292970, 14.21338, 14.19831, 14.196598, 14.20023]
# avg_Delay_15 = [1.52045, 2.23425, 3.09702, 4.0945, 5.0782, 6.04662]
# defective_rate_15 = [0.0495597, 0.046923, 0.048908, 0.0476571, 0.04863636, 0.0490909]


# X = ['Buffer=0','1/FPS','2/FPS','3/FPS', "4/FPS", "5/FPS"]
# X_axis = np.arange(len(X))

# fig, ax1 = pyplot.subplots()  
# ax2 = ax1.twinx()

# ax1.bar(X_axis - 0.1, useful_Mbps_18, 0.2, label = 'Actual', color="purple")
# ax1.bar(X_axis + 0.1, max_Mbps_18, 0.2, label = 'Maximal', color="blue")
# ax1.legend(["Actual", "Maximal"])

# ax2.plot(X_axis, avg_Delay_18, 'orange')

# ax1.set_xlabel('How much buffer applied?')
# ax1.set_ylabel('Throughput during testing phase (in Mbps)', color='blue')
# ax2.set_ylabel('Avg. delay for non-defective frames (in 1/FPS)', color='orange')

# pyplot.xticks(X_axis, X)

# pyplot.title("Trace data No. 18: throughput graph")
# pyplot.show()




## Dataset 18

x_axis_ours = [0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
ours = [0.56, 0.75, 0.923, 0.932, 0.913, 0.842, 0.762, 0.625, 0.511]

x_axis = [0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
RLS = [0.05, 0.226, 0.623, 0.798, 0.797, 0.751, 0.6712, 0.5869, 0.48]


## Dataset 8
# x_axis = [0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
# RLS = [0.02, 0.04, 0.375, 0.683, 0.872, 0.85, 0.8061, 0.7146, 0.6184]

# x_axis_ours = [0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
# ours = [0.245, 0.33 , 0.79, 0.87, 0.96, 0.97, 0.92, 0.87, 0.82, 0.71, 0.62]

pyplot.plot(x_axis_ours, ours, 'o-')
pyplot.plot(x_axis, RLS, 'o-')
pyplot.xlabel("Target")
pyplot.ylabel("Bandwidth utilization")
pyplot.legend(["Ours", "OnRLS"])
pyplot.show()