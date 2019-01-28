import numpy as np
import matplotlib.pyplot as plt


def get_linear_fit(x_data, y_data):
	xSum=0
	ySum=0
	xxSum=0
	xySum=0
	for i in range(len(x_data)):
		xSum += x_data[i]
		ySum += y_data[i]
		xxSum += x_data[i] ** 2
		xySum += x_data[i] * y_data[i]

	slope = (len(x_data) * xySum - xSum * ySum) / (len(x_data) * xxSum - xSum * xSum)
	intercept = (ySum - slope * xSum) / len(x_data)

	return slope, intercept


def calc_r_squared(slope, intercept, x, y):
	p = np.poly1d([slope, intercept])
	# fit values, and mean
	yhat = p(x) # or [p(z) for z in x]
	ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
	ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
	sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
	r_squared = ssreg / sstot

	return r_squared



encoder_poses = np.array([180, 650, 1180, 1770, 2420, 3120, 4020, 5030, 6150, 7450, 9020])
approxs_freq = np.array([55.0, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00])
approxs_freq_squared =  np.multiply(approxs_freq, approxs_freq)

# Take regression
slope, intercept = get_linear_fit(encoder_poses, approxs_freq_squared)
x = np.arange(0., encoder_poses[-1] + 1000, 1)

# Calculate R Value
r_squared = calc_r_squared(slope, intercept, encoder_poses, approxs_freq_squared)
print("slope: " + str(slope) + " intercept: " + str(intercept) + " r_squared: " + str(r_squared))


plt.figure(1)

plt.scatter(encoder_poses, approxs_freq)
plt.title("Encoder Position vs. Frequency")
plt.xlabel("Encoder Position")
plt.ylabel("Frequency")

plt.figure(2)
plt.scatter(encoder_poses, approxs_freq_squared)
plt.plot(x, slope*x + intercept)
plt.title("Encoder Position vs. Frequency Squared")
plt.xlabel("Encoder Position")
plt.ylabel("Frequency Squared")
plt.legend(["r_squared={}".format(r_squared)], loc=2)



plt.show()