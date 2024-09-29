import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

def sim_diff(counts, binomial_probs, normal_probs):
    diff_binomial = counts - binomial_probs
    diff_normal = counts - normal_probs

    # Set up the bin edges for plotting
    bins = np.arange(len(counts))

    # Create a figure and axes
    plt.figure(figsize=(10, 6))

    # Plot the difference between simulation and binomial probabilities
    plt.bar(bins - 0.2, diff_binomial, width=0.4, label="Difference vs Binomial", align="center")

    # Plot the difference between simulation and normal probabilities
    plt.bar(bins + 0.2, diff_normal, width=0.4, label="Difference vs Normal", align="center")

    # Add labels and title
    plt.xlabel('Bins')
    plt.ylabel('Difference')
    plt.title('Difference Between Simulated and Theoretical Distributions')
    plt.legend()

    # Show the plot
    plt.show()


# Function to simulate the Galton board
def simulate_galton_board(n, N):
    # Initialize the counts for balls landing in each position
    counts = np.zeros(n + 1)
    
    for _ in range(N):
        position = 0  # Start at the top
        for _ in range(n):
            position += np.random.choice([0, 1])  # Move left or right
        counts[position] += 1  # Increment the count for the final position

    return counts

# Function to calculate the binomial probabilities
def binomial_probabilities(n):
    return binom.pmf(np.arange(n + 1), n, 0.5)

# Function to calculate normal distribution probabilities
def normal_probabilities(n):
    mu = n / 2
    sigma = np.sqrt(n / 4)
    x = np.arange(n + 1)
    return norm.pdf(x, mu, sigma)


#n = 10  # Number of levels
#N = 1000  # Number of balls

def simulate(n=10, N=1000):
    
    counts = simulate_galton_board(n, N)
    binomial_probs = binomial_probabilities(n) * N
    normal_probs = normal_probabilities(n) * N
    sim_diff(counts, binomial_probs, normal_probs)


    # Plotting the results
    plt.bar(np.arange(n + 1) - 0.2, counts, width=0.4, label='Simulated', color='blue', alpha=0.7)
    plt.bar(np.arange(n + 1) + 0.2, binomial_probs, width=0.4, label='Binomial', color='orange', alpha=0.7)
    plt.plot(np.arange(n + 1), normal_probs, label='Normal', color='black', linewidth=2)

    plt.xlabel('Position')
    plt.ylabel('Number of Balls')
    plt.title('Galton Board Simulation')
    plt.legend()
    #plt.show()
    plt.savefig(f'galton_board_n_{n}_10000.jpg', format='png')
    plt.close()  # Close the figure to avoid display

simulate(1000, 10000)


