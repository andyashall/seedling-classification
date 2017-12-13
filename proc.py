from keras.preprocessing import image
from PIL import ImageFilter
from skimage import filters
import numpy as np

img = image.load_img('./data/test/00c47e980.png')

# img = img.filter(ImageFilter.FIND_EDGES)

img = np.array(image.img_to_array(img))

# edges = filters.sobel(img)
# img = img.imshow(edges)
# io.show()

img = np.array(image.img_to_array(img))
img *= 255.0/img.max()

print(img.shape)


img[img[:, :, 1] < 80] = 0
img[img[:, :, 2] > 50] = 0

# for c in range(img.shape[0]):
#   for r in range(img.shape[1]):
#     r,g,b = img[c,r]
#     print(r)
#     if b < 50 and g > 100:
#       img[c,r] = r,g,b
#     else:
#       img[c,r] = 0,0,0

img = image.array_to_img(img)

img = img.filter(ImageFilter.FIND_EDGES)

img.save('./data/proc.png', 'png')