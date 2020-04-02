from parse_img import parse_img
from bresenham_line import BresenhamLine
from math import sin, cos, pi
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import os


def compute(SETTINGS, img):
    center = tuple(map(lambda x: x / 2, img.shape))
    r = center[0]
    ro = SETTINGS['range_angle']
    n_det = SETTINGS['n_det']
    n_steps = SETTINGS['n_steps']

    radon = np.zeros((SETTINGS['n_det'], SETTINGS['n_steps']), dtype='float64')
    inv_radon = np.zeros(img.shape, dtype='float64')

    alpha = 0
    inv_radons = []
    det_emits = []
    for i_step in tqdm(range(SETTINGS['n_steps'])):
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
        inv_radons.append(deepcopy(inv_radon))
        det_emits.append((emiter, detectors))

    return radon, inv_radon, inv_radons, det_emits


def run(SETTINGS):
    img_original, img = parse_img(SETTINGS['img_name'], SETTINGS['shape'])
    radon, inv_radon, inv_radons = compute(SETTINGS, img)
    plt.subplot(2, 3, 1)
    plt.imshow(img_original)
    plt.subplot(2, 3, 4)
    plt.imshow(radon)
    plt.subplot(2, 3, 5)
    plt.imshow(inv_radon)
    plt.subplot(2, 3, 6)
    part0 = int((1 / 6) * inv_radon.shape[0])
    part1 = int((1 / 6) * inv_radon.shape[1])
    inv_radon = inv_radon[part0:-part0, part1:-part1]
    plt.imshow(inv_radon)
    plt.show()


if __name__ == "__main__":
    SETTINGS = {
        'img_name': "./img/Shepp_logan.jpg",
        'shape': (240, 240),
        'n_det': 200,
        'range_angle': pi * 1.5,
        'n_steps': 400
    }
    run(SETTINGS)
