import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from utils import gaussNormalDensityFcn

# This code works with the image doramar.jpg, but it can be used for any other image analysis with certain changes that
# have to be determined experimentally.

# load the image
image = imread("images/doramar.jpg")
imShape = image.shape

# show the image
plt.figure(figsize=[10,10])
plt.imshow(image)
plt.axis('off')

plt.show()
############################
# Segments of the image are determined experimentally
s1 = image[48:120, 20:300, :]
s2 = image[54:450, 370:450, :]
s3 = image[480:520, 130:220, :]
s5 = image[345:460, 150:185, :]

# RED SEGMENT - s1
s1Shape = (s1.shape[0], s1.shape[1])
print("Resolution of red segment: " + str(s1Shape))

plt.figure()
plt.imshow(s1)
plt.title("Segment s1")
plt.axis('off')

# channels
s1RedCh = s1[:, :, 0]
s1GreenCh = s1[:, :, 1]
s1BlueCh = s1[:, :, 2]

# statistical parameters of s1
s1RedChMean = np.mean(s1RedCh)
s1GreenChMean = np.mean(s1GreenCh)
s1BlueChMean = np.mean(s1BlueCh)

s1RedChVar = np.var(s1RedCh)
s1GreenChVar = np.var(s1GreenCh)
s1BlueChVar = np.var(s1BlueCh)

# s1 histograms

plt.figure(figsize = [7, 18])
plt.subplot(3, 1, 1)
plt.title('Histogrami i estimirane pdf RGB kanala za segment 1')
_, _, _ = plt.hist(s1RedCh.flatten(), bins = 40, density = True, color = 'lightcoral', label = 'histogram')
# number of bins and xlim have to be determined empirically
plt.grid()
plt.xlim(200, 250)
x = np.arange(0, 256, 0.1)
y = gaussNormalDensityFcn(x, s1RedChMean, s1RedChVar)
plt.plot(x, y, color = 'red', label = 'estimirana funkcija gustine raspodjele')
plt.legend()

plt.subplot(3, 1, 2)
_, _, _ = plt.hist(s1GreenCh.flatten(), bins = 35, density = True, color = 'aquamarine', label = 'histogram')
plt.grid()
plt.xlim(0, 50)
x = np.arange(0, 256, 0.1)
y = gaussNormalDensityFcn(x, s1GreenChMean, s1GreenChVar)
plt.plot(x, y, color = 'green', label = 'estimirana funkcija gustine raspodjele')
plt.legend()

plt.subplot(3, 1, 3)
_, _, _ = plt.hist(s1BlueCh.flatten(), bins = 30, density = True, color = 'lightskyblue', label = 'histogram')
plt.grid()
plt.xlim(0, 50)
x = np.arange(0, 256, 0.1)
y = gaussNormalDensityFcn(x, s1BlueChMean, s1BlueChVar)
plt.plot(x, y, color = 'blue', label = 'estimirana funkcija gustine raspodjele')
plt.xlabel('intenzitet boje')
plt.legend()

plt.show()

# ORANGE SEGMENT - s2
s2Shape = (s2.shape[0], s2.shape[1])
print("Resolution of orange segment: " + str(s2Shape))

plt.figure()
plt.imshow(s2)
plt.title("Segment s2")
plt.axis('off')

# channels
s2RedCh = s2[:, :, 0]
s2GreenCh = s2[:, :, 1]
s2BlueCh = s2[:, :, 2]

# statistical parameters of s2
s2RedChMean = np.mean(s2RedCh)
s2GreenChMean = np.mean(s2GreenCh)
s2BlueChMean = np.mean(s2BlueCh)

s2RedChVar = np.var(s2RedCh)
s2GreenChVar = np.var(s2GreenCh)
s2BlueChVar = np.var(s2BlueCh)

# s2 histograms

plt.figure(figsize = [7, 18])
plt.subplot(3, 1, 1)
plt.title('Histogrami i estimirane pdf RGB kanala za segment 2')
_, _, _ = plt.hist(s2RedCh.flatten(), bins = 30, density = True, color = 'lightcoral', label = 'histogram')
plt.grid()
plt.xlim(150, 250)
x = np.arange(0, 256, 0.1)
y = gaussNormalDensityFcn(x, s2RedChMean, s2RedChVar)
plt.plot(x, y, color = 'red', label = 'estimirana funkcija gustine raspodjele')
plt.legend()

plt.subplot(3, 1, 2)
_, _, _ = plt.hist(s2GreenCh.flatten(), bins = 30, density = True, color = 'aquamarine', label = 'histogram')
plt.grid()
x = np.arange(0, 256, 0.1)
y = gaussNormalDensityFcn(x, s2GreenChMean, s2GreenChVar)
plt.plot(x, y, color = 'green', label = 'estimirana funkcija gustine raspodjele')
plt.legend()

plt.subplot(3, 1, 3)
_, _, _ = plt.hist(s2BlueCh.flatten(), bins = 25, density = True, color = 'lightskyblue', label = 'histogram')
plt.grid()
plt.xlim(0, 50)
x = np.arange(0, 256, 0.1)
y = gaussNormalDensityFcn(x, s2BlueChMean, s2BlueChVar)
plt.plot(x, y, color = 'blue', label = 'estimirana funkcija gustine raspodjele')
plt.xlabel('intenzitet boje')
plt.legend()
plt.show()

# PURPLE SEGMENT - s3
s3Shape = (s3.shape[0], s3.shape[1])
print("Resolution of purple segment: " + str(s3Shape))

plt.figure()
plt.imshow(s3)
plt.title("Segment s3")
plt.axis('off')

# channels
s3RedCh = s3[:, :, 0]
s3GreenCh = s3[:, :, 1]
s3BlueCh = s3[:, :, 2]

# statistical parameters of s3
s3RedChMean = np.mean(s3RedCh)
s3GreenChMean = np.mean(s3GreenCh)
s3BlueChMean = np.mean(s3BlueCh)

s3RedChVar = np.var(s3RedCh)
s3GreenChVar = np.var(s3GreenCh)
s3BlueChVar = np.var(s3BlueCh)

# s3 histograms
plt.figure(figsize = [7, 18])
plt.subplot(3, 1, 1)
plt.title('Histogrami i estimirane pdf RGB kanala za segment 3')
_, _, _ = plt.hist(s3RedCh.flatten(), bins = 30, density = True, color = 'lightcoral', label = 'histogram')
plt.grid()
x = np.arange(0, 256, 0.1)
y = gaussNormalDensityFcn(x, s3RedChMean, s3RedChVar)
plt.plot(x, y, color = 'red', label = 'estimirana funkcija gustine raspodjele')
plt.legend()

plt.subplot(3, 1, 2)
_, _, _ = plt.hist(s3GreenCh.flatten(), bins = 30, density = True, color = 'aquamarine', label = 'histogram')
plt.grid()
x = np.arange(0, 256, 0.1)
y = gaussNormalDensityFcn(x, s3GreenChMean, s3GreenChVar)
plt.plot(x, y, color = 'green', label = 'estimirana funkcija gustine raspodjele')
plt.legend()

plt.subplot(3, 1, 3)
_, _, _ = plt.hist(s3BlueCh.flatten(), bins = 25, density = True, color = 'lightskyblue', label = 'histogram')
plt.grid()
x = np.arange(0, 256, 0.1)
y = gaussNormalDensityFcn(x, s3BlueChMean, s3BlueChVar)
plt.plot(x, y, color = 'blue', label = 'estimirana funkcija gustine raspodjele')
plt.xlabel('intenzitet boje')
plt.legend()
plt.show()
# YELLOW - s5
s5Shape = (s5.shape[0], s5.shape[1])
print("Resolution of yellow segment: " + str(s5Shape))

plt.figure()
plt.imshow(s5)
plt.title("Segment s5")
plt.axis('off')

# channels
s5RedCh = s5[:, :, 0]
s5GreenCh = s5[:, :, 1]
s5BlueCh = s5[:, :, 2]

# statistical parameters of s5
s5RedChMean = np.mean(s5RedCh)
s5GreenChMean = np.mean(s5GreenCh)
s5BlueChMean = np.mean(s5BlueCh)

s5RedChVar = np.var(s5RedCh)
s5GreenChVar = np.var(s5GreenCh)
s5BlueChVar = np.var(s5BlueCh)

# s5 histograms
plt.figure(figsize = [7, 18])
plt.subplot(3, 1, 1)
plt.title('Histogrami i estimirane pdf RGB kanala za segment 5')
_, _, _ = plt.hist(s5RedCh.flatten(), bins = 25, density = True, color = 'lightcoral', label = 'histogram')
plt.grid()
plt.xlim(150, 250)
x = np.arange(0, 256, 0.1)
y = gaussNormalDensityFcn(x, s5RedChMean, s5RedChVar)
plt.plot(x, y, color = 'red', label = 'estimirana funkcija gustine raspodjele')
plt.legend()

plt.subplot(3, 1, 2)
_, _, _ = plt.hist(s5GreenCh.flatten(), bins = 30, density = True, color = 'aquamarine', label = 'histogram')
plt.grid()
x = np.arange(0, 256, 0.1)
y = gaussNormalDensityFcn(x, s5GreenChMean, s5GreenChVar)
plt.plot(x, y, color = 'green', label = 'estimirana funkcija gustine raspodjele')
plt.legend()

plt.subplot(3, 1, 3)
_, _, _ = plt.hist(s5BlueCh.flatten(), bins = 40, density = True, color = 'lightskyblue', label = 'histogram')
plt.grid()
plt.xlim(0, 50)
x = np.arange(0, 256, 0.1)
y = gaussNormalDensityFcn(x, s5BlueChMean, s5BlueChVar)
plt.plot(x, y, color = 'blue', label = 'estimirana funkcija gustine raspodjele')
plt.xlabel('intenzitet boje')
plt.legend()
plt.show()

# Data synthesis
s1Mean = np.array([[s1RedChMean, s1GreenChMean, s1BlueChMean]]).T
x = np.vstack((s1RedCh.flatten(), s1GreenCh.flatten(), s1BlueCh.flatten()))
s1Var = np.cov(x)
s1Synth = np.random.multivariate_normal(s1Mean.flatten(), s1Var, s1Shape)/256
# scaled with 256 to form interval [0, 1] for imshow's representation of RGB data for floats


s2Mean = np.array([[s2RedChMean, s2GreenChMean, s2BlueChMean]]).T
x = np.vstack((s2RedCh.flatten(), s2GreenCh.flatten(), s2BlueCh.flatten()))
s2Var = np.cov(x)
s2Synth = np.random.multivariate_normal(s2Mean.flatten(), s2Var, s2Shape)/256

s3Mean = np.array([[s3RedChMean, s3GreenChMean, s3BlueChMean]]).T
x = np.vstack((s3RedCh.flatten(), s3GreenCh.flatten(), s3BlueCh.flatten()))
s3Var = np.cov(x)
s3Synth = np.random.multivariate_normal(s3Mean.flatten(), s3Var, s3Shape)/256

s5Mean = np.array([[s5RedChMean, s5GreenChMean, s5BlueChMean]]).T
x = np.vstack((s5RedCh.flatten(), s5GreenCh.flatten(), s5BlueCh.flatten()))
s5Var = np.cov(x)
s5Synth = np.random.multivariate_normal(s5Mean.flatten(), s5Var, s5Shape)/256

# Comparison of real colors and colors from estimated model
plt.figure(figsize = ([5, 3]))
plt.subplot(2, 1, 1)
plt.title("stvarno")
plt.imshow(s1)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.title("sintetizirano")
plt.imshow(s1Synth)
plt.axis('off')

plt.figure(figsize = ([3, 5]))
plt.subplot(1, 2, 1)
plt.title("stvarno")
plt.imshow(s2)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title("sintetizirano")
plt.imshow(s2Synth)
plt.axis('off')

plt.show()

# Palette of colors for segments 1, 2, 3 and 5
s1Palette = np.random.multivariate_normal(s1Mean.flatten(), s1Var, (200, 200))/256
s2Palette = np.random.multivariate_normal(s2Mean.flatten(), s2Var, (200, 200))/256
s3Palette = np.random.multivariate_normal(s3Mean.flatten(), s3Var, (200, 200))/256
s5Palette = np.random.multivariate_normal(s5Mean.flatten(), s5Var, (200, 200))/256

plt.figure(figsize = ([5, 5]))
plt.subplot(2, 2, 1)
plt.imshow(s1Palette)
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(s2Palette)
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(s3Palette)
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(s5Palette)
plt.axis('off')

# Histogram and empirical function
plt.figure(figsize=[5, 3])
plt.subplot(1, 2, 1)
cnt, bins, _ = plt.hist(s2GreenCh.flatten(), bins = 30, density = True, color = 'green')
x = np.arange(0, 256, 0.1)
y = gaussNormalDensityFcn(x, s2GreenChMean, s2GreenChVar)
plt.plot(x, y, color = 'lime')
plt.xlim([50, 150])
plt.title("Histogram")
plt.grid()

plt.show()

# histogram with different number of bins is needed for good estimation of cdf, smaller number of bins is good for
# better representation of concepts

cnt, bins = np.histogram(s2GreenCh.flatten(), bins = 256, density=True)
plt.subplot(1, 2, 2)
plt.title("Empirijska funkcija")
plt.plot(np.cumsum(cnt)*np.diff(bins)[0], 'g.' )
plt.plot(x, np.cumsum(y)*0.1, color = 'lime')
plt.grid()
plt.show()