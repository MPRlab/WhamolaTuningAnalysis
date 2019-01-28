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




# This uses FS=200, LENGTH=4096, PD_FUDGE=0.15, filteredfreq*250
real_freq = range(20, 100, 10)
real_freq2 = range(25, 146, 5)
print(real_freq)
measured_freq = [12, 15, 18.75, 23, 27, 31, 36, 40, 45]
measured_freq2 = [13.5, 15.0, 17.2, 19.5, 21.4, 24.0, 25.8, 28.3, 30.7, 32.8, 35.3, 37.7, 39.8, 42.2, 44.5, 47.2, 49.5, 51.7, 54.0, 56.5, 59.0, 61.1, 63.9, 65.8, 68.1]
measured_freq3 = [22.05, 18.2, 39.8, 24.96, 29.7, 33.81, 38.2]
print
print(measured_freq)

if (len(real_freq2) != len(measured_freq2)):
		print("Real Freq is " + str(len(real_freq2)) + "long.")
		print("Measured Freq is " + str(len(measured_freq2)) + "long.")


#coeffs = np.polyfit(measured_freq, real_freq, 2)
coeffs2 = np.polyfit(measured_freq2, real_freq2, 2)
#print(coeffs)
print(coeffs2)
# print("r_squared: " + calc_r_squared(coeffs, measured_freq, real_freq))
# p = np.poly1d(z)
#x = np.arange(0., measured_freq[-1] + 1000, 1)
x2 = np.arange(0., measured_freq2[-1] + 1000, 1)

#plt.figure(1)
#plt.title("ACF Curve Fit")
#plt.scatter(measured_freq, real_freq)
#plt.plot(x, (coeffs[0]*pow(x, 2) + coeffs[1]*x + coeffs[2]))
#plt.xlim(0, 110)
#plt.ylim(0, 110)
#plt.xlabel("Real Frequency (Hz)")
#plt.ylabel("ACF Frequency (Hz)")

plt.figure(2)
plt.title("Extended ACF Curve Fit")
plt.scatter(measured_freq2, real_freq2)
plt.plot(x2, (coeffs2[0]*pow(x2, 2) + coeffs2[1]*x2 + coeffs2[2]))
plt.xlim(0, 155)
plt.ylim(0, 155)
plt.xlabel("Real Frequency (Hz)")
plt.ylabel("ACF Frequency (Hz)")

plt.show()