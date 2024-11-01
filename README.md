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

The voltage-dependent currents are described by Hodgkin-Huxley formalisms as:

<img width="174" alt="Screenshot 2024-11-01 at 09 57 55" src="https://github.com/user-attachments/assets/73c83c2b-422d-459e-b71b-63adb9820cd0">

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

where I<sub>app</sub> is a constant external current and the voltage-dependent currents are as described above, except for I<sub>T</sub>. It follows a simpler form as:

<img width="130" alt="Screenshot 2024-11-01 at 10 00 48" src="https://github.com/user-attachments/assets/24552077-7623-4046-81b8-43ef0ebd1a77">

### Synaptic Currents

Two synaptic currents are incorporated:
- **<i>I<sub>S→G</sub></i>**: Excitatory input from the STN
- **<i>I<sub>G→G</sub></i>**: Inhibitory influence from other GPe cells

## Synaptic Conductivity

Details of STN and GPe cell connectivity are modeled with varying levels of interconnection (sparse vs. tight). The models affect network dynamics, leading to clustering, wave propagation, and spiking activity. Simplified depictions of the arrangements of modeled networks can be visualized as:

<img width="519" alt="Screenshot 2024-11-01 at 09 52 11" src="https://github.com/user-attachments/assets/fe257776-7541-46d4-b200-ab597ee29071">

a) Sparsely connected model: Each GPe neuron inhibits its two closest GPe neighbors and also inhibits two STN neurons, skipping the three nearest ones. Each STN and SNC cell sends excitation only to its nearest GPe cell in line. Spatially periodic boundary conditions were enforced. 

b) Tightly connected model. Each GPe neuron interacts with the five nearest STN neurons and six nearest GPe cells. Each STN and SNC cell provides excitation to the three closest GPe cells. Spatially periodic boundary conditions were applied.

## GPe and STN parameter values used for modeling

<img width="333" alt="Screenshot 2024-11-01 at 10 09 08" src="https://github.com/user-attachments/assets/a7563da0-d290-4ead-b6b2-4b91958bab33">

Parameters in nanosiemans per square micrometer represent the maximal conductances of the respective currents in the STN and GPe current balance equations. Parameters in millivolts (mV) correspond to the reversal potentials for these currents. Parameters in milliseconds (msec) denote the time constants related to the time evolution of the gating variables in the Hodgkin-Huxley models of these currents, while other parameters pertain to associated constants. The membrane potentials in between STN and GPe cells were kept as constants with V<sub>S→G</sub>=0 mV, V<sub>G→S</sub>=-85 mV, and V<sub>G→G</sub>=-85 mV, while the conductances (g) were varied. For the simulations where the conductances were not the center of focus, they were held at a standard level as g<sub>S→G</sub>=0.5 nS/m2, g<sub>G→S</sub>=3.0 nS/m2, and g<sub>G→G</sub>=0.02 nS/m2. g<sub>T</sub> and g<sub>AHP</sub> parameters were also manipulated in the result collection stage.

## Example Results

### Figure 1
<img width="610" alt="Screenshot 2024-11-01 at 09 54 56" src="https://github.com/user-attachments/assets/c9bb403e-5e3d-4216-819d-5eddf8ba4b68">

Firing behaviors of GPe and STN neurons in a sparsely connected model across various <i>g<sub>T</sub></i> levels (0.5, 10, and 100 <i>nS/m<sup>2</sup></i>).

### Figure 2
<img width="609" alt="Screenshot 2024-11-01 at 09 55 22" src="https://github.com/user-attachments/assets/c0aabd37-72b2-4edd-90ce-164d081c96d8">

Firing behaviors of GPe and STN neurons in a tightly connected model with different <i>g<sub>T</sub></i> levels (0.5, 10, and 100 <i>nS/m<sup>2</sup></i>).
