import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import spams

PATCH_SIZE=(11, 11)

def build_dict(x_len):

    def haar_wavelet(size, j, k, width):
        base = np.zeros((size, 1))
        amp = 2**(j/2.0 - 1)
        level = 0.5/(2**j)
        space = int(round(level*width))
        base[k:k+space] = amp
        base[k+space:k+2*space] = -amp
        return base

    waves = []
    width = 4
    for k in range(0, x_len, width//2):
        waves.append(haar_wavelet(x_len, 0, k, width))
    
    width = 2
    for k in range(0, x_len, width//2):
        waves.append(haar_wavelet(x_len, 1, k, width))
    
    return np.hstack(waves)


D = build_dict(PATCH_SIZE[0]*PATCH_SIZE[1])
print(f"Generated Haar dictionary ({D.shape})")
D = np.asfortranarray(D)

img = cv.imread('input_image.png')[:, :, 1]
blur = cv.GaussianBlur(img, ksize=(5, 5), sigmaX=0)

patches = []
for i in range(0, blur.shape[0]-PATCH_SIZE[0], PATCH_SIZE[0]):
    for j in range(0, blur.shape[1]-PATCH_SIZE[1], PATCH_SIZE[1]):
        patch = blur[i:i+PATCH_SIZE[0], j:j+PATCH_SIZE[1]].flatten()
        mu = patch.mean()
        sigma = patch.std()
        X = np.asfortranarray((patch-mu)/sigma)
        
        alpha = spams.lasso(np.asfortranarray(X[:, None]), D, lambda1=0.25)
        patches.append(alpha)
        

patches = np.array(patches)
print(patches.shape)