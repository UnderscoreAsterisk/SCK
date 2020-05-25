import cv2 as cv
import numpy as np
import spams
import sys

PATCH_SIZE=(11, 11)
lower_CM = 7
upper_CM = 150

INPUT_FNAME = sys.argv[1]
OUTPUT_FNAME = f"{INPUT_FNAME.split('.')[0]}_keypoints.png"

class KeyPoint:
    def __init__(self, X, Y, alpha):
        self.X = X
        self.Y = Y
        self.alpha = alpha

    @property
    def SM(self):
        return np.count_nonzero(self.alpha)+sum(abs(self.alpha))

    @property
    def CM(self):
        return np.count_nonzero(self.alpha)

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

# Build Dictionary
D = build_dict(PATCH_SIZE[0]*PATCH_SIZE[1])
print(f"Generated Haar dictionary {D.shape}")
D = np.asfortranarray(D)

# Step1: Mean blur
img = cv.imread(INPUT_FNAME)
blur = cv.blur(img[:, :, 1], ksize=(11, 11))

# Step2: blockwise sparse representation learning
key_pts = []
for i in range(0, blur.shape[0]-PATCH_SIZE[0], PATCH_SIZE[0]):
    kp_row = []
    for j in range(0, blur.shape[1]-PATCH_SIZE[1], PATCH_SIZE[1]):
        patch = blur[i:i+PATCH_SIZE[0], j:j+PATCH_SIZE[1]].flatten()
        mu = patch.mean()
        sigma = patch.std()
        if sigma == 0:
            X = np.asfortranarray(np.zeros(patch.shape))
        else:
            X = np.asfortranarray((patch-mu)/sigma)
        
        alpha = spams.lasso(np.asfortranarray(X[:, None]), D, lambda1=0.125)

        kp_row.append(KeyPoint(i+PATCH_SIZE[0]//2, j+PATCH_SIZE[1]//2, alpha.toarray()))
    key_pts.append(kp_row)

print(f"Keypoint matrix: ({len(key_pts)}, {len(key_pts[0])}) [Total = {len(key_pts)*len(key_pts[0])}]")

max_y = len(key_pts)-1
max_x = len(key_pts[0])-1
suppresed_kp = []

NEIGHBOURHOOD = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
# Step3: Non maxima suppression
for y, row in enumerate(key_pts):
    for x, kp in enumerate(row):
        if y == 0 or y == max_y or x == 0 or x == max_x:
            continue
        elif kp.CM < lower_CM or kp.CM > upper_CM:
            continue
        else:
            max_SM = kp.SM
            for (dx, dy) in NEIGHBOURHOOD:
                max_SM = max(max_SM, key_pts[y+dy][x+dx].SM)
            if max_SM == kp.SM:
                suppresed_kp.append(kp)

print(f"{len(suppresed_kp)} Keypoints remain after suppression")

#Draw the obtained keypoints on the image
kp_size = int(np.sqrt((PATCH_SIZE[0]**2 + PATCH_SIZE[1]**2)/2))
for kp in suppresed_kp:
    cv.circle(img, (kp.Y, kp.X), kp_size//2, (0, 255, 0), 2)
cv.imwrite(OUTPUT_FNAME, img)