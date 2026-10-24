import pandas as pd
import matplotlib.pyplot as plt

'''
This will provide the line graph for Rel_brightness v/s Time_Days
So as to visualize the data.
'''

stac1 = pd.read_csv('stac1_brightness.csv')
stac2 = pd.read_csv('stac2_brightness.csv')
stac3 = pd.read_csv('stac3_brightness.csv')
stell_obv = pd.read_csv('stellar_observations_data.csv')

plt.plot(stac1["Time_days"], stac1["Rel_brightness"])
plt.title("STAC-1 Brightness Observations")
plt.xlabel("Time (days)")
plt.ylabel("Relative Brightness")
plt.grid(True)
plt.show()

plt.plot(stac2["Time_days"], stac2["Rel_brightness"])
plt.title("STAC-2 Brightness Observations")
plt.xlabel("Time (days)")
plt.ylabel("Relative Brightness")
plt.grid(True)
plt.show()

plt.plot(stac3["Time_days"], stac3["Rel_brightness"])
plt.title("STAC-3 Brightness Observations")
plt.xlabel("Time (days)")
plt.ylabel("Relative Brightness")
plt.grid(True)
plt.show()