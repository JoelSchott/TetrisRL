import matplotlib.pyplot as plt


def plot_epsilon(steps: int, epsilon_max: float, epsilon_min: float, epsilon_decay: float):
    epsilons = []
    for i in range(steps):
        epsilons.append(epsilon_min + (epsilon_max - epsilon_min) * (epsilon_decay ** i))
    plt.plot(epsilons)
    plt.show()


def main():
    plot_epsilon(10000, 1, 0.1, 0.9995)


if __name__ == '__main__':
    main()