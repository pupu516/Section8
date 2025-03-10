import numpy as np

# Parameters
L = 4  # Lattice size (L x L)
J = 1.0  # Coupling constant
B = 0.0  # External magnetic field
T = 1.0  # Temperature
beta = 1.0 / T  # Inverse temperature

# Initialize random spin configuration
np.random.seed(42)
spins = np.random.choice([-1, 1], size=(L, L))

# Function to compute Hamiltonian
def hamiltonian(spins, J, B):
    energy = 0.0
    for i in range(L):
        for j in range(L):
            # Sum over nearest neighbors (periodic boundary conditions)
            energy -= J * spins[i, j] * (spins[(i+1)%L, j] + spins[i, (j+1)%L])
            energy -= B * spins[i, j]
    return energy

# Compute Hamiltonian
H = hamiltonian(spins, J, B)
print("Hamiltonian:", H)



from itertools import product

# Generate all possible spin configurations
all_spins = list(product([-1, 1], repeat=L*L))

# Compute partition function Z
Z = sum(np.exp(-beta * hamiltonian(np.array(spins).reshape(L, L), J, B)) for spins in all_spins)

# Compute probabilities for each configuration
probabilities = [np.exp(-beta * hamiltonian(np.array(spins).reshape(L, L), J, B)) / Z for spins in all_spins]

# Visualize one configuration
import matplotlib.pyplot as plt
plt.imshow(spins, cmap="binary")
plt.title("Random Spin Configuration")
plt.colorbar()
plt.savefig("part_2.png")



# Gibbs sampling function
def gibbs_sampler(spins, J, B, beta, n_steps=1000):
    for _ in range(n_steps):
        for i in range(L):
            for j in range(L):
                # Compute local field
                local_field = J * (spins[(i+1)%L, j] + spins[(i-1)%L, j] + spins[i, (j+1)%L] + spins[i, (j-1)%L]) + B
                # Compute probability of flipping
                p_flip = 1 / (1 + np.exp(-2 * beta * local_field * spins[i, j]))
                # Flip spin with probability p_flip
                if np.random.rand() < p_flip:
                    spins[i, j] *= -1
    return spins

# Run Gibbs sampler
gibbs_spins = gibbs_sampler(spins.copy(), J, B, beta, n_steps=1000)

# Visualize Gibbs-sampled configuration
plt.imshow(gibbs_spins, cmap="binary")
plt.title("Gibbs-Sampled Spin Configuration")
plt.colorbar()
plt.savefig("part_3.png")


# Function to compute magnetization
def magnetization(spins):
    return np.mean(spins)

# Study phase transition
temperatures = np.linspace(0.1, 5.0, 50)
magnetizations = []

for T in temperatures:
    beta = 1.0 / T
    gibbs_spins = gibbs_sampler(spins.copy(), J, B, beta, n_steps=1000)
    magnetizations.append(magnetization(gibbs_spins))

# Plot magnetization vs temperature
plt.plot(temperatures, magnetizations, marker="o")
plt.xlabel("Temperature (T)")
plt.ylabel("Magnetization (M)")
plt.title("Phase Transition in 2D Ising Model")
plt.axvline(x=2.269, color="red", linestyle="--", label="Critical Temperature")
plt.legend()
plt.savefig("part_4.png")
