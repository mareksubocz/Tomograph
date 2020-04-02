from parse_img import parse_img
from bresenham_line import BresenhamLine
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from math import sin, cos, pi
import numpy as np
from numba import jit
import os


def draw_plot(img, emiter, detectors, radon, inv_radon):
    plt.subplot(2, 2, 1)
    plt.cla()
    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.scatter(*emiter, c='g')
    plt.xlim(0, img.shape[0])
    plt.ylim(img.shape[1], 0)
    plt.plot(*zip(*detectors), 'r.')

    plt.subplot(2, 2, 2)
    plt.cla()
    plt.imshow(radon, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    plt.cla()
    plt.imshow(inv_radon, cmap='gray')


def compute_step(i_step, img, center, r, ro, n_det, radon, inv_radon, n_steps):
    global alpha
    emiter = [int(center[0] + r * cos(alpha)),
              int(center[1] - r * sin(alpha))]
    detectors = [
        (int(center[0] + r * cos(alpha + pi - ro / 2 + i * ro / (n_det - 1))),
         int(center[0] - r * sin(alpha + pi - ro / 2 + i * ro / (n_det - 1))))
        for i in range(n_det)]

    # W razie błedu wyjścia poza obraz
    # detectors = list(map(lambda x: np.clip(x, 0, img.shape[0] - 1), detectors))

    # Radon
    chosen_lines = [np.clip(BresenhamLine(*emiter, *detector), 0,
                            img.shape[0] - 1) for detector in detectors]
    lines = [list(map(lambda p: img[p[0]][p[1]], line))
             for line in chosen_lines]

    result_list = list(map(sum, lines))
    radon[:, i_step] = result_list

    # Inverse Radon
    for i_line, line in enumerate(chosen_lines):
        for p in line:
            inv_radon[p[0]][p[1]] += result_list[i_line]

    alpha += 2 * pi / n_steps
    draw_plot(img, emiter, detectors, radon, inv_radon)
    return alpha


def start_computing(SETTINGS):
    img = parse_img(SETTINGS['img_name'], SETTINGS['shape'])
    center = tuple(map(lambda x: x / 2, img.shape))
    r = center[0]
    ro = SETTINGS['range_angle']
    n_det = SETTINGS['n_det']
    n_steps = SETTINGS['n_steps']

    radon = np.zeros((SETTINGS['n_det'], SETTINGS['n_steps']), dtype='float64')
    inv_radon = np.zeros(img.shape, dtype='float64')

    fig, ax = plt.subplots(2, 2)
    plt.subplot(2, 2, 2)
    plt.imshow(radon, cmap='gray')
    plt.colorbar()
    ani = FuncAnimation(fig,
                        lambda x: compute_step(
                            x, img, center, r, ro, n_det, radon, inv_radon, n_steps),
                        frames=SETTINGS['n_steps'],
                        repeat=True)

    # HTML(ani.to_html5_video())

    plt.show()
    print("koniec")


alpha = 0
if __name__ == "__main__":
    SETTINGS = {
        'img_name': "./img/Shepp_logan.jpg",
        'shape': (240, 240),
        'n_det': 200,
        'range_angle': pi * 1.5,
        'n_steps': 400
    }

    start_computing(SETTINGS)
