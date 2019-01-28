import numpy as np
import matplotlib.pyplot as plt


def calc_r_squared(coeffs, x, y):
	p = np.poly1d(coeffs)
	# fit values, and mean
	yhat = p(x) # or [p(z) for z in x]
	ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
	ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
	sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
	r_squared = ssreg / sstot

	return r_squared



real_freq3 = [34.65, 51.91, 58.27, 77.78, 87.3, 98, 110]
measured_freq3 = [22.05, 18.2, 39.8, 24.96, 29.7, 33.81, 38.2]


coeffs3 = np.polyfit(measured_freq3, real_freq3, 2)
print(coeffs3)
x3 = np.arange(0., measured_freq3[-1] + 100, 1)

plt.figure(1)
plt.title("ACF Curve Fit with Manual Tuning and Whamola Strikers")
plt.scatter(measured_freq3, real_freq3)
plt.plot(x3, (coeffs3[0]*pow(x3, 2) + coeffs3[1]*x3 + coeffs3[2]))
plt.xlim(0, 155)
plt.ylim(0, 155)
plt.xlabel("Real Frequency (Hz)")
plt.ylabel("Whamola Strike ACF Frequency (Hz)")


# With filtering 

filtered_measured3 = [24.96, 29.7, 33.81, 39.2]
real_filtered_freq3 = [77.78, 87.3, 98, 110]

coeffs3_f = np.polyfit(filtered_measured3, real_filtered_freq3, 2)
print(coeffs3_f)
x3_f = np.arange(0., filtered_measured3[-1] + 100, 1)


plt.figure(2)
plt.title("Filtered ACF Curve Fit with Manual Tuning and Whamola Strikers")
plt.scatter(filtered_measured3, real_filtered_freq3)
plt.plot(x3_f, (coeffs3_f[0]*pow(x3_f, 2) + coeffs3_f[1]*x3_f + coeffs3_f[2]))
plt.xlim(0, 155)
plt.ylim(0, 155)
plt.xlabel("Real Frequency (Hz)")
plt.ylabel("Whamola Strike ACF Frequency (Hz)")



plt.show()
