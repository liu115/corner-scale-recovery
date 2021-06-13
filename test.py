import numpy as np
import matplotlib.pyplot as plt
from scale_recovery import estimate_by_searching_in_range


def basic():
    np.random.seed(1111)
    SCALE_1 = 2 * 100
    SCALE_2 = 3 * 100
    NOISE_LEVEL = 0
    base = np.random.rand(15, 2)
    corner1 = base * SCALE_1
    corner2 = base * SCALE_2 + np.random.rand(15, 2) * NOISE_LEVEL
    estimate_by_searching_in_range(
        corner1, corner2,
        initial_scale=1, max_scale=3,
        scale_step=0.1, delta=0,
        plot=True, plot_fn='no_delta_no_noise.png')
    plt.clf()

def basic_with_noise():
    np.random.seed(1111)
    SCALE_1 = 2 * 100
    SCALE_2 = 3 * 100
    NOISE_LEVEL = 1
    base = np.random.rand(15, 2)
    corner1 = base * SCALE_1
    corner2 = base * SCALE_2 + np.random.rand(15, 2) * NOISE_LEVEL
    estimate_by_searching_in_range(
        corner1, corner2,
        initial_scale=1, max_scale=3,
        scale_step=0.1, delta=0,
        plot=True, plot_fn='no_delta_with_noise.png')
    plt.clf()


def basic_with_delta():
    np.random.seed(1111)
    SCALE_1 = 2 * 100
    SCALE_2 = 3 * 100
    NOISE_LEVEL = 0
    base = np.random.rand(15, 2)
    corner1 = base * SCALE_1
    corner2 = base * SCALE_2 + np.random.rand(15, 2) * NOISE_LEVEL
    estimate_by_searching_in_range(
        corner1, corner2,
        initial_scale=1, max_scale=3,
        scale_step=0.1, delta=1e-8,
        plot=True, plot_fn='with_delta_no_noise.png')
    plt.clf()


def noise_and_delta():
    np.random.seed(1111)
    SCALE_1 = 2 * 100
    SCALE_2 = 3 * 100
    NOISE_LEVEL = 1
    base = np.random.rand(15, 2)
    corner1 = base * SCALE_1
    corner2 = base * SCALE_2 + np.random.rand(15, 2) * NOISE_LEVEL
    estimate_by_searching_in_range(
        corner1, corner2,
        initial_scale=1, max_scale=3,
        scale_step=0.1, delta=1e-8,
        plot=True, plot_fn='with_delta_with_noise.png')
    plt.clf()



def main():
    basic()
    basic_with_noise()
    basic_with_delta()
    noise_and_delta()


main()