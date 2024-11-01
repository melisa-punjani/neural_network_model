# Methodology

## Overview

To investigate the interactions among GPe, STN, and SNC cells, conductance-based models of the subthalamopallidal network were simulated using Python. Simulations involved adjusting intrinsic and synaptic properties of the cells to uncover mechanisms behind their interactions. Two distinct networks were employed to explore structured connectivity patterns at varying interconnection levels.

## Modeling the STN

The STN model comprises various currents and their kinetics. The model includes:

- Spike-generating currents (<i>I<sub>K</sub></i> and <i>I<sub>Na</sub></i>)
- Low-threshold T-type <i>Ca<sup>2+</sup></i> current (<i>I<sub>T</sub></i>)
- High-threshold <i>Ca<sup>2+</sup></i> current (<i>I<sub>Ca</sub></i>)
- Calcium-activated and voltage-independent afterhyperpolarization <i>K<sup>+</sup></i> current (<i>I<sub>AHP</sub></i>)
- Leak current (<i>I<sub>L</sub></i>)

The membrane potential of each STN neuron follows:

<i>C<sub>m</sub> = -I<sub>L</sub> - I<sub>K</sub> - I<sub>Na</sub> - I<sub>Ca</sub> - I<sub>T</sub> - I<sub>AHP</sub> - I<sub>G→S</sub></i>

### Gating Variables

Gating variables are split into:
- **Slowly operating (n, h, r)**
- **Rapidly activating (m, a, s)**

The activation gating for rapidly activating channels is instantaneous, while slowly operating gating variables depend on both time and voltage. Differential equations describe these variables' kinetics.

### Calcium Dynamics

The intracellular <i>Ca<sup>2+</sup></i> concentration [Ca] is governed by:

<i>[Ca]' = ε(-I<sub>Ca</sub> - I<sub>T</sub> - k<sub>Ca</sub>[Ca])</i>

## Modeling the GPe

The GPe model, similar to the STN model, follows:

<i>C<sub>m</sub> = -I<sub>L</sub> - I<sub>K</sub> - I<sub>Na</sub> - I<sub>Ca</sub> - I<sub>T</sub> - I<sub>AHP</sub> - I<sub>S→S</sub> - I<sub>G→G</sub> + I<sub>app</sub></i>

### Synaptic Currents

Two synaptic currents are incorporated:
- **<i>I<sub>S→G</sub></i>**: Excitatory input from the STN
- **<i>I<sub>G→G</sub></i>**: Inhibitory influence from other GPe cells

## Synaptic Conductivity

Details of STN and GPe cell connectivity are modeled with varying levels of interconnection (sparse vs. tight). The models affect network dynamics, leading to clustering, wave propagation, and spiking activity.

## Example Results

### Figure 2

Firing behaviors of GPe and STN neurons in a sparsely connected model across various <i>g<sub>T</sub></i> levels (0.5, 10, and 100 <i>nS/m<sup>2</sup></i>).

### Figure 3

Firing behaviors of GPe and STN neurons in a tightly connected model with different <i>g<sub>T</sub></i> levels (0.5, 10, and 100 <i>nS/m<sup>2</sup></i>).
