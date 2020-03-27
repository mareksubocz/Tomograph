from parse_img import parse_img
from bresenham_line import BresenhamLine
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sin, cos, pi
import numpy as np
from numba import jit
import time
import os


@jit
def clamp(vals, minval, maxval):

    for i, val in enumerate(vals):
        if val < minval:
            vals[i] = minval
        if val > maxval:
            vals[i] = maxval
    return vals


def draw_plot(img, emiter, detectors, radon):
    plt.subplot(1, 2, 1)
    plt.cla()
    plt.imshow(img, cmap='gray'),
    plt.xticks([]), plt.yticks([])
    plt.scatter(*emiter, c='g')
    plt.plot(*zip(*detectors), 'r.')

    plt.subplot(1, 2, 2)
    plt.cla()
    plt.imshow(radon, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()


def compute_step(i_step, img, center, r, ro, n_det, radon, n_steps):
    global alpha
    emiter = [int(center[0] + r * cos(alpha)),
              int(center[1] - r * sin(alpha))]

    detectors = [
        (int(center[0] + r * cos(alpha + pi - ro / 2 + i * ro / (n_det - 1))),
         int(center[0] - r * sin(alpha + pi - ro / 2 + i * ro / (n_det - 1))))
        for i in range(n_det)]
    end = time.time()

    # W razie błedu wyjścia poza obraz
    # detectors = list(map(lambda x: np.clip(x, 0, img.shape[0] - 1), detectors))

    lines = [np.clip(BresenhamLine(*emiter, *detector), 0,
                     img.shape[0] - 1) for detector in detectors]

    # start = time.time()
    # lines = find_lines(emiter, detectors, img.shape)
    # end = time.time()

    # print('new:', end - start)

    lines = [list(map(lambda p: img[p[0]][p[1]], line)) for line in lines]

    radon[:, i_step] = list(map(sum, lines))
    end = time.time()
    alpha += 2 * pi / n_steps
    # i_step += 1

    draw_plot(img, emiter, detectors, radon)
    return alpha


@jit(nopython=True, nogil=True, fastmath=True)
def find_lines(emiter, detectors, shape, img):
    lines = np.array([clamp(BresenhamLine(*emiter, *detector),
                            0, shape[0] - 1) for detector in detectors])
    lines = np.zeros(())
    # for detector in detectors:
    #     for
    return lines


def start_computing(SETTINGS):
    img = parse_img(SETTINGS['img_name'], SETTINGS['shape'])
    center = tuple(map(lambda x: x / 2, img.shape))
    r = center[0]
    ro = SETTINGS['range_angle']
    n_det = SETTINGS['n_det']
    n_steps = SETTINGS['n_steps']

    radon = np.zeros((SETTINGS['n_det'], SETTINGS['n_steps']), dtype='float64')

    ani = FuncAnimation(plt.gcf(),
                        lambda x: compute_step(
                            x, img, center, r, ro, n_det, radon, n_steps),
                        frames=SETTINGS['n_steps'],
                        repeat=False)
    plt.show()
    print("elo")


alpha = 0


def run():
    SETTINGS = {
        'img_name': "./img/Kwadraty2.jpg",
        'shape': (240, 240),
        'n_det': 100,
        'range_angle': pi / 4,
        'n_steps': 200
    }

    start_computing(SETTINGS)


if __name__ == "__main__":
    run()
