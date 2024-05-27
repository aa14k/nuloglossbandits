import matplotlib.pyplot as plt

def plot(x, log, sqr, name):
    plt.xlabel('Training data size (timesteps)')
    plt.ylabel('Average failure rate')
    plt.plot(x, log.mean(1), '.--', label='Log Loss')
    plt.plot(x, sqr.mean(1), '.--', label='Squared Loss')
    plt.legend()
    plt.savefig(name)
