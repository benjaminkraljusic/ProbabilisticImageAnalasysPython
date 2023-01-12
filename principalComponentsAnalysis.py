import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA, IncrementalPCA

image = cv2.imread("images/tesla.jpg", 0)

print(image.shape)
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')

# Kompresija
pca = PCA()
pca.fit(image)

varCumuPerc = np.cumsum(pca.explained_variance_ratio_)*100
k = np.argmax(varCumuPerc > 95)
print(pca.components_.shape)
print(str(k) + " glavnih komponenti sadrzava 95% varijanse.")

fig, ax = plt.subplots(figsize=[7,5])
ax.set_title('Kumulativna varijansa sadržana u glavnim komponentama')
ax.set_ylabel('Kumulativna varijansa')
ax.set_xlabel('Glavne komponente')
ax.axvline(x=k, color="k", linestyle="--")
ax.axhline(y=95, color="r", linestyle="--")
ax.grid()


ax = plt.plot(varCumuPerc)

# Rekonstrukcija
ipca = IncrementalPCA(n_components=k)
imageRec = ipca.inverse_transform(ipca.fit_transform(image))

fig, ax = plt.subplots()
ax.imshow(imageRec, cmap=plt.cm.gray)
ax.axis("off")

print(ipca.components_.shape)

fig, ax = plt.subplots(2, 2, figsize = [5, 5])
n = 0
m = 0
for i in [50, 150, 200, 250]:
    ipca = IncrementalPCA(n_components=i)
    tf = ipca.fit_transform(image)
    print(ipca.components_.shape)
    print(ipca.components_.shape)
    imageRec = ipca.inverse_transform(tf)
    ax[n, m].imshow(imageRec, cmap=plt.cm.gray)
    ttl = str(i) + " glavnih komponenti"
    ax[n, m].set_title(ttl)
    ax[n, m].axis("off")
    m += 1
    if m == 2:
        n = 1
        m = 0

fig, ax = plt.subplots(figsize=[7,5])
ax.bar(range(1, 101), pca.explained_variance_ratio_[0:100]*100, align='center')
ax.grid()
ax.set_title("Varijansa sadrzana u individualnim glavnim komponentama")
ax.set_xlabel("Glavne komponente")
plt.show()