from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt


def parse_img(filename, shape):
    img_original = Image.open(filename).convert("L")
    img_original = img_original.resize(shape, Image.ANTIALIAS)
    img = ImageOps.expand(img_original, border=int(
        shape[0] * 0.25), fill='black')
    img = np.asarray(img, dtype='float64')
    img_original = np.asarray(img_original)
    return img_original, img


if __name__ == "__main__":
    img = parse_img("./img/CT_ScoutView-large.jpg", (320, 320))

    circle = plt.Circle(
        (img.size[0] / 2, img.size[0] / 2), img.size[0] / 2, color='r', fill=False)
    plt.gcf().gca().add_artist(circle)
    plt.show()
