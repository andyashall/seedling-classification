from keras.preprocessing import image
from PIL import ImageFilter
import numpy as np

img = image.load_img('./data/test/0ad9e7dfb.png')

# img = img.filter(ImageFilter.FIND_EDGES)

img = np.array(image.img_to_array(img))

# edges = filters.sobel(img)
# img = img.imshow(edges)
# io.show()

img = np.array(image.img_to_array(img))
# Normalize
img *= 255.0/img.max()

# if green is less than 80 then 0
# img[np.all([img[:, :, 1] < 70, img[:, :, 2] > 60])] = 0

# Link for pre implemented

green = img[:, :, 1] < 100

blue = img[:, :, 2] > 20

g2 = img[:, :, 1] > 100

b2 = img[:, :, 2] > 100

print(green.shape)

new = np.zeros(shape=green.shape, dtype=bool)

for i in range(0, len(green)):
  for n in range(0, len(green)):
    if (green[i,n] and blue[i,n]) or (g2[i,n] and b2[i,n]):
      new[i,n] = True
      print(new[i,n])
    else:
      new[i,n] = False
      print(new[i,n])

print(new)

print(blue)

img[new] = 0

# if blue is more than 50 then 0
# img[img[:, :, 2] > 60] = 0

# for c in range(img.shape[0]):
#   for r in range(img.shape[1]):
#     r,g,b = img[c,r]
#     print(r)
#     if b < 50 and g > 100:
#       img[c,r] = r,g,b
#     else:
#       img[c,r] = 0,0,0

img = image.array_to_img(img)

# img = img.filter(ImageFilter.FIND_EDGES)

img.save('./data/proc.png', 'png')