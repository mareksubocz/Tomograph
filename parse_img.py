from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt


def parse_img(filename, shape):
    img = Image.open(filename).convert("L")
    img = img.resize(shape, Image.ANTIALIAS)
    img = ImageOps.expand(
        img, border=int(shape[0] * 0.25), fill='black')
    arr = np.asarray(img)
    return arr


if __name__ == "__main__":
    img = parse_img("./img/CT_ScoutView-large.jpg", (320, 320))

    circle = plt.Circle(
        (img.size[0] / 2, img.size[0] / 2), img.size[0] / 2, color='r', fill=False)
    plt.gcf().gca().add_artist(circle)
    plt.show()
