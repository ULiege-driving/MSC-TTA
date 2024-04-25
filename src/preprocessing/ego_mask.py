from PIL import Image
import numpy as np

img = Image.open('')

data = np.array(img)

mask = np.all(data[:, :, :3] == [0, 0, 142], axis=-1)

data[mask] = [255, 255, 255, 255]
data[~mask] = [0, 0, 0, 255]

new_img = Image.fromarray(data)

new_img.save('')
