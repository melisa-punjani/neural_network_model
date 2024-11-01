import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

C_m = 1

gLSTN = 2.25
gKSTN = 45
gNaSTN = 37.5
gTSTN = 0.5
gCaSTN = 0.5
gAHPSTN = 9

gLGPE = 0.1
gKGPE = 30
gNaGPE = 120
gTGPE = 0.5
gCaGPE = 0.15
gAHPGPE = 30

vLSTN = -60
vKSTN = -80
vNaSTN = 55
vCaSTN = 140

vLGPE = -55
vKGPE = -80
vNaGPE = 55
vCaGPE = 120

tauh1STN= 500
taun1STN = 100
taur1STN = 17.5

tauh1GPE = 0.27
taun1GPE = 0.27
taurGPE = 30

tauh0STN = 1
taun0STN = 1
taur0STN = 40

tauh0GPE = 0.05
taun0GPE = 0.05

phihSTN = 0.75
phinSTN = 0.75
phirSTN = 0.2

phihGPE = 0.05
phinGPE = 0.05
phirGPE = 1.0

k1STN = 15
kCaSTN = 22.5

k1GPE = 30
kCaGPE = 20

epsilonSTN = 3.75 * (10 ** -5)
epsilonGPE = 1 * (10 ** -4)

thetamSTN = -30
thetahSTN = -39
thetanSTN = -32
thetarSTN = -67
thetaaSTN = -63
thetabSTN = 0.4
thetasSTN = -39

thetamGPE = -37
thetahGPE = -58
thetanGPE = -50
thetarGPE = -70
thetaaGPE = -57
thetasGPE = -35

thetaTauhSTN = -57
thetaTaunSTN = -80
thetaTaurSTN= 68
thetaHgSTN = -39
thetagSTN = 30

thetaTauhGPE = -40
thetaTaunGPE = -40
thetaHgGPE = -57
thetagGPE = 20

alphaSTN = 5
alphaGPE = 2

sigmamSTN = 15
sigmahSTN = -3.1
sigmanSTN = 8
sigmarSTN = -2
sigmaaSTN = 7.8
sigmabSTN = -0.1
sigmasSTN = 8
sigmaTauhSTN = -3
sigmaTaunSTN = -26.0
sigmaTaurSTN = -2.2
sigmaHgSTN = 8

sigmamGPE = 10
sigmahGPE = -12
sigmanGPE = 14
sigmarGPE = -2
sigmaaGPE = 2
sigmasGPE = 2
sigmaTauhGPE = -12
sigmaTaunGPE = -12
sigmaHgGPE = 2

betaSTN = 1
betaGPE = 0.04

vGPE_to_STN = -85
vGPE_to_GPE = -85
vSTN_to_GPE = 0

g_GPE_to_STN = 3 # need to reduce to mimic hyperexcitability
g_STN_to_GPE = 25 # need to ramp up to mimic hyperexcitability
g_GPE_to_GPE = 0.0002 # need to reduce to mimic hyperexcitability

I_app = -1

# did not include VGG yet

def INTtoX_DE2(time, solution):
    VSTN1 =solution[0]
    nSTN1 = solution[1]
    hSTN1 =solution[2]
    rSTN1 = solution[3]
    sSTN1 = solution[4]
    CaSTN1 = solution[5]

    ninfSTN1 = 1/(1+ np.exp( -(VSTN1-thetanSTN)/sigmanSTN))
    minfSTN1 = 1/(1+ np.exp(-(VSTN1-thetamSTN)/sigmamSTN))
    hinfSTN1 = 1/(1+ np.exp(-(VSTN1-thetahSTN)/sigmahSTN))
    ainfSTN1 = 1/(1+ np.exp(-(VSTN1-thetaaSTN)/sigmaaSTN))
    rinfSTN1 = 1/(1+ np.exp(-(VSTN1-thetarSTN)/sigmarSTN))
    sinfSTN1 = 1/(1+ np.exp(-(VSTN1-thetasSTN)/sigmasSTN))
    binfSTN1 = 1/(1+ np.exp((rSTN1 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSTN1 = gLSTN * (VSTN1 - vLSTN)
    IKSTN1 = gKSTN * (nSTN1 ** 4) * (VSTN1 - vKSTN)
    INaSTN1 = gNaSTN * (minfSTN1 ** 3) * hSTN1 * (VSTN1 - vNaSTN)
    ITSTN1 = gTSTN * (ainfSTN1 ** 3) * (binfSTN1 ** 2) * (VSTN1 - vCaSTN)
    ICaSTN1 = gCaSTN * (sinfSTN1 ** 2) * (VSTN1 - vCaSTN)
    IAHPSTN1 = gAHPSTN * (VSTN1 - vKSTN) * (CaSTN1/(CaSTN1 + k1STN))

    taunSTN1 = taun0STN + taun1STN/(1 + np.exp(-(VSTN1 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSTN1 = tauh0STN + tauh1STN/(1 + np.exp(-(VSTN1 - thetaTauhSTN)/sigmaTauhSTN))
    taurSTN1 = taur0STN + taur1STN/(1 + np.exp(-(VSTN1 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSTN1 = 1/(1 + np.exp(-((VSTN1-30) + 39)/8))

    VSTN2 =solution[6]
    nSTN2 = solution[7]
    hSTN2 =solution[8]
    rSTN2 = solution[9]
    sSTN2 = solution[10]
    CaSTN2 = solution[11]

    ninfSTN2 = 1/(1+ np.exp( -(VSTN2-thetanSTN)/sigmanSTN))
    minfSTN2 = 1/(1+ np.exp(-(VSTN2-thetamSTN)/sigmamSTN))
    hinfSTN2 = 1/(1+ np.exp(-(VSTN2-thetahSTN)/sigmahSTN))
    ainfSTN2 = 1/(1+ np.exp(-(VSTN2-thetaaSTN)/sigmaaSTN))
    rinfSTN2 = 1/(1+ np.exp(-(VSTN2-thetarSTN)/sigmarSTN))
    sinfSTN2 = 1/(1+ np.exp(-(VSTN2-thetasSTN)/sigmasSTN))
    binfSTN2 = 1/(1+ np.exp((rSTN2 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSTN2 = gLSTN * (VSTN2 - vLSTN)
    IKSTN2 = gKSTN * (nSTN2 ** 4) * (VSTN2 - vKSTN)
    INaSTN2 = gNaSTN * (minfSTN2 ** 3) * hSTN2 * (VSTN2 - vNaSTN)
    ITSTN2 = gTSTN * (ainfSTN2 ** 3) * (binfSTN2 ** 2) * (VSTN2 - vCaSTN)
    ICaSTN2 = gCaSTN * (sinfSTN2 ** 2) * (VSTN2 - vCaSTN)
    IAHPSTN2 = gAHPSTN * (VSTN2 - vKSTN) * (CaSTN2/(CaSTN2 + k1STN))

    taunSTN2 = taun0STN + taun1STN/(1 + np.exp(-(VSTN2 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSTN2 = tauh0STN + tauh1STN/(1 + np.exp(-(VSTN2 - thetaTauhSTN)/sigmaTauhSTN))
    taurSTN2 = taur0STN + taur1STN/(1 + np.exp(-(VSTN2 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSTN2 = 1/(1 + np.exp(-((VSTN2-30) + 39)/8))

    VSTN3 =solution[12]
    nSTN3 = solution[13]
    hSTN3 =solution[14]
    rSTN3 = solution[15]
    sSTN3 = solution[16]
    CaSTN3 = solution[17]

    ninfSTN3 = 1/(1+ np.exp( -(VSTN3-thetanSTN)/sigmanSTN))
    minfSTN3 = 1/(1+ np.exp(-(VSTN3-thetamSTN)/sigmamSTN))
    hinfSTN3 = 1/(1+ np.exp(-(VSTN3-thetahSTN)/sigmahSTN))
    ainfSTN3 = 1/(1+ np.exp(-(VSTN3-thetaaSTN)/sigmaaSTN))
    rinfSTN3 = 1/(1+ np.exp(-(VSTN3-thetarSTN)/sigmarSTN))
    sinfSTN3 = 1/(1+ np.exp(-(VSTN3-thetasSTN)/sigmasSTN))
    binfSTN3 = 1/(1+ np.exp((rSTN3 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSTN3 = gLSTN * (VSTN3 - vLSTN)
    IKSTN3 = gKSTN * (nSTN3 ** 4) * (VSTN3 - vKSTN)
    INaSTN3 = gNaSTN * (minfSTN3 ** 3) * hSTN3 * (VSTN3 - vNaSTN)
    ITSTN3 = gTSTN * (ainfSTN3 ** 3) * (binfSTN3 ** 2) * (VSTN3 - vCaSTN)
    ICaSTN3 = gCaSTN * (sinfSTN3 ** 2) * (VSTN3 - vCaSTN)
    IAHPSTN3 = gAHPSTN * (VSTN3 - vKSTN) * (CaSTN3/(CaSTN3 + k1STN))

    taunSTN3 = taun0STN + taun1STN/(1 + np.exp(-(VSTN3 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSTN3 = tauh0STN + tauh1STN/(1 + np.exp(-(VSTN3 - thetaTauhSTN)/sigmaTauhSTN))
    taurSTN3 = taur0STN + taur1STN/(1 + np.exp(-(VSTN3 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSTN3 = 1/(1 + np.exp(-((VSTN3-30) + 39)/8))

    VSTN4 =solution[18]
    nSTN4 = solution[19]
    hSTN4 =solution[20]
    rSTN4 = solution[21]
    sSTN4 = solution[22]
    CaSTN4 = solution[23]

    ninfSTN4 = 1/(1+ np.exp( -(VSTN4 -thetanSTN)/sigmanSTN))
    minfSTN4 = 1/(1+ np.exp(-(VSTN4 -thetamSTN)/sigmamSTN))
    hinfSTN4 = 1/(1+ np.exp(-(VSTN4-thetahSTN)/sigmahSTN))
    ainfSTN4 = 1/(1+ np.exp(-(VSTN4-thetaaSTN)/sigmaaSTN))
    rinfSTN4 = 1/(1+ np.exp(-(VSTN4-thetarSTN)/sigmarSTN))
    sinfSTN4 = 1/(1+ np.exp(-(VSTN4-thetasSTN)/sigmasSTN))
    binfSTN4 = 1/(1+ np.exp((rSTN4 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSTN4 = gLSTN * (VSTN4 - vLSTN)
    IKSTN4 = gKSTN * (nSTN4 ** 4) * (VSTN4 - vKSTN)
    INaSTN4 = gNaSTN * (minfSTN4 ** 3) * hSTN4 * (VSTN4 - vNaSTN)
    ITSTN4 = gTSTN * (ainfSTN4 ** 3) * (binfSTN4 ** 2) * (VSTN4 - vCaSTN)
    ICaSTN4 = gCaSTN * (sinfSTN4 ** 2) * (VSTN4 - vCaSTN)
    IAHPSTN4 = gAHPSTN * (VSTN4 - vKSTN) * (CaSTN4/(CaSTN4 + k1STN))

    taunSTN4 = taun0STN + taun1STN/(1 + np.exp(-(VSTN4 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSTN4 = tauh0STN + tauh1STN/(1 + np.exp(-(VSTN4 - thetaTauhSTN)/sigmaTauhSTN))
    taurSTN4 = taur0STN + taur1STN/(1 + np.exp(-(VSTN4 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSTN4 = 1/(1 + np.exp(-((VSTN4-30) + 39)/8))

    VSTN5 =solution[24]
    nSTN5 = solution[25]
    hSTN5 =solution[26]
    rSTN5 = solution[27]
    sSTN5 = solution[28]
    CaSTN5 = solution[29]

    ninfSTN5 = 1/(1+ np.exp( -(VSTN5-thetanSTN)/sigmanSTN))
    minfSTN5 = 1/(1+ np.exp(-(VSTN5-thetamSTN)/sigmamSTN))
    hinfSTN5 = 1/(1+ np.exp(-(VSTN5-thetahSTN)/sigmahSTN))
    ainfSTN5 = 1/(1+ np.exp(-(VSTN5-thetaaSTN)/sigmaaSTN))
    rinfSTN5 = 1/(1+ np.exp(-(VSTN5-thetarSTN)/sigmarSTN))
    sinfSTN5 = 1/(1+ np.exp(-(VSTN5-thetasSTN)/sigmasSTN))
    binfSTN5 = 1/(1+ np.exp((rSTN5 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSTN5 = gLSTN * (VSTN5 - vLSTN)
    IKSTN5 = gKSTN * (nSTN5 ** 4) * (VSTN5 - vKSTN)
    INaSTN5 = gNaSTN * (minfSTN5 ** 3) * hSTN5 * (VSTN5 - vNaSTN)
    ITSTN5 = gTSTN * (ainfSTN5 ** 3) * (binfSTN5 ** 2) * (VSTN5 - vCaSTN)
    ICaSTN5 = gCaSTN * (sinfSTN5 ** 2) * (VSTN5 - vCaSTN)
    IAHPSTN5 = gAHPSTN * (VSTN5 - vKSTN) * (CaSTN5/(CaSTN5 + k1STN))

    taunSTN5 = taun0STN + taun1STN/(1 + np.exp(-(VSTN5 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSTN5 = tauh0STN + tauh1STN/(1 + np.exp(-(VSTN5 - thetaTauhSTN)/sigmaTauhSTN))
    taurSTN5 = taur0STN + taur1STN/(1 + np.exp(-(VSTN5 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSTN5 = 1/(1 + np.exp(-((VSTN5-30) + 39)/8))

    VSTN6 =solution[30]
    nSTN6 = solution[31]
    hSTN6 =solution[32]
    rSTN6 = solution[33]
    sSTN6 = solution[34]
    CaSTN6 = solution[35]

    ninfSTN6 = 1/(1+ np.exp( -(VSTN6-thetanSTN)/sigmanSTN))
    minfSTN6 = 1/(1+ np.exp(-(VSTN6-thetamSTN)/sigmamSTN))
    hinfSTN6 = 1/(1+ np.exp(-(VSTN6-thetahSTN)/sigmahSTN))
    ainfSTN6 = 1/(1+ np.exp(-(VSTN6-thetaaSTN)/sigmaaSTN))
    rinfSTN6 = 1/(1+ np.exp(-(VSTN6-thetarSTN)/sigmarSTN))
    sinfSTN6 = 1/(1+ np.exp(-(VSTN6-thetasSTN)/sigmasSTN))
    binfSTN6 = 1/(1+ np.exp((rSTN6 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSTN6 = gLSTN * (VSTN6 - vLSTN)
    IKSTN6 = gKSTN * (nSTN6 ** 4) * (VSTN6 - vKSTN)
    INaSTN6 = gNaSTN * (minfSTN6 ** 3) * hSTN6 * (VSTN6 - vNaSTN)
    ITSTN6 = gTSTN * (ainfSTN6 ** 3) * (binfSTN6 ** 2) * (VSTN6 - vCaSTN)
    ICaSTN6 = gCaSTN * (sinfSTN6 ** 2) * (VSTN6 - vCaSTN)
    IAHPSTN6 = gAHPSTN * (VSTN6 - vKSTN) * (CaSTN6/(CaSTN6 + k1STN))

    taunSTN6 = taun0STN + taun1STN/(1 + np.exp(-(VSTN6 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSTN6 = tauh0STN + tauh1STN/(1 + np.exp(-(VSTN6 - thetaTauhSTN)/sigmaTauhSTN))
    taurSTN6 = taur0STN + taur1STN/(1 + np.exp(-(VSTN6 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSTN6 = 1/(1 + np.exp(-((VSTN6-30) + 39)/8))

    VSTN7 =solution[36]
    nSTN7 = solution[37]
    hSTN7 =solution[38]
    rSTN7 = solution[39]
    sSTN7 = solution[40]
    CaSTN7 = solution[41]

    ninfSTN7 = 1/(1+ np.exp( -(VSTN7-thetanSTN)/sigmanSTN))
    minfSTN7 = 1/(1+ np.exp(-(VSTN7-thetamSTN)/sigmamSTN))
    hinfSTN7 = 1/(1+ np.exp(-(VSTN7-thetahSTN)/sigmahSTN))
    ainfSTN7 = 1/(1+ np.exp(-(VSTN7-thetaaSTN)/sigmaaSTN))
    rinfSTN7 = 1/(1+ np.exp(-(VSTN7-thetarSTN)/sigmarSTN))
    sinfSTN7 = 1/(1+ np.exp(-(VSTN7-thetasSTN)/sigmasSTN))
    binfSTN7 = 1/(1+ np.exp((rSTN7 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSTN7 = gLSTN * (VSTN7 - vLSTN)
    IKSTN7 = gKSTN * (nSTN7 ** 4) * (VSTN7 - vKSTN)
    INaSTN7 = gNaSTN * (minfSTN7 ** 3) * hSTN7 * (VSTN7 - vNaSTN)
    ITSTN7 = gTSTN * (ainfSTN7 ** 3) * (binfSTN7 ** 2) * (VSTN7 - vCaSTN)
    ICaSTN7 = gCaSTN * (sinfSTN7 ** 2) * (VSTN7 - vCaSTN)
    IAHPSTN7 = gAHPSTN * (VSTN7 - vKSTN) * (CaSTN7/(CaSTN7 + k1STN))

    taunSTN7 = taun0STN + taun1STN/(1 + np.exp(-(VSTN7 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSTN7 = tauh0STN + tauh1STN/(1 + np.exp(-(VSTN7 - thetaTauhSTN)/sigmaTauhSTN))
    taurSTN7 = taur0STN + taur1STN/(1 + np.exp(-(VSTN7 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSTN7 = 1/(1 + np.exp(-((VSTN7-30) + 39)/8))

    VSTN8 =solution[42]
    nSTN8 = solution[43]
    hSTN8 = solution[44]
    rSTN8 = solution[45]
    sSTN8 = solution[46]
    CaSTN8 = solution[47]

    ninfSTN8 = 1/(1+ np.exp( -(VSTN8-thetanSTN)/sigmanSTN))
    minfSTN8 = 1/(1+ np.exp(-(VSTN8-thetamSTN)/sigmamSTN))
    hinfSTN8 = 1/(1+ np.exp(-(VSTN8-thetahSTN)/sigmahSTN))
    ainfSTN8 = 1/(1+ np.exp(-(VSTN8-thetaaSTN)/sigmaaSTN))
    rinfSTN8 = 1/(1+ np.exp(-(VSTN8-thetarSTN)/sigmarSTN))
    sinfSTN8 = 1/(1+ np.exp(-(VSTN8-thetasSTN)/sigmasSTN))
    binfSTN8 = 1/(1+ np.exp((rSTN8 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSTN8 = gLSTN * (VSTN8 - vLSTN)
    IKSTN8 = gKSTN * (nSTN8 ** 4) * (VSTN8 - vKSTN)
    INaSTN8 = gNaSTN * (minfSTN8 ** 3) * hSTN8 * (VSTN8 - vNaSTN)
    ITSTN8 = gTSTN * (ainfSTN8 ** 3) * (binfSTN8 ** 2) * (VSTN8 - vCaSTN)
    ICaSTN8 = gCaSTN * (sinfSTN8 ** 2) * (VSTN8 - vCaSTN)
    IAHPSTN8 = gAHPSTN * (VSTN8 - vKSTN) * (CaSTN8/(CaSTN8 + k1STN))

    taunSTN8 = taun0STN + taun1STN/(1 + np.exp(-(VSTN8 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSTN8 = tauh0STN + tauh1STN/(1 + np.exp(-(VSTN8 - thetaTauhSTN)/sigmaTauhSTN))
    taurSTN8 = taur0STN + taur1STN/(1 + np.exp(-(VSTN8 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSTN8 = 1/(1 + np.exp(-((VSTN8-30) + 39)/8))

    VSTN9=solution[48]
    nSTN9 = solution[49]
    hSTN9 =solution[50]
    rSTN9 = solution[51]
    sSTN9 = solution[52]
    CaSTN9 = solution[53]

    ninfSTN9 = 1/(1+ np.exp( -(VSTN9-thetanSTN)/sigmanSTN))
    minfSTN9 = 1/(1+ np.exp(-(VSTN9-thetamSTN)/sigmamSTN))
    hinfSTN9 = 1/(1+ np.exp(-(VSTN9-thetahSTN)/sigmahSTN))
    ainfSTN9 = 1/(1+ np.exp(-(VSTN9-thetaaSTN)/sigmaaSTN))
    rinfSTN9 = 1/(1+ np.exp(-(VSTN9-thetarSTN)/sigmarSTN))
    sinfSTN9 = 1/(1+ np.exp(-(VSTN9-thetasSTN)/sigmasSTN))
    binfSTN9 = 1/(1+ np.exp((rSTN9 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSTN9 = gLSTN * (VSTN9 - vLSTN)
    IKSTN9 = gKSTN * (nSTN9 ** 4) * (VSTN9 - vKSTN)
    INaSTN9 = gNaSTN * (minfSTN9 ** 3) * hSTN9 * (VSTN9 - vNaSTN)
    ITSTN9 = gTSTN * (ainfSTN9 ** 3) * (binfSTN9 ** 2) * (VSTN9 - vCaSTN)
    ICaSTN9 = gCaSTN * (sinfSTN9 ** 2) * (VSTN9 - vCaSTN)
    IAHPSTN9 = gAHPSTN * (VSTN9 - vKSTN) * (CaSTN9/(CaSTN9 + k1STN))

    taunSTN9 = taun0STN + taun1STN/(1 + np.exp(-(VSTN9 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSTN9 = tauh0STN + tauh1STN/(1 + np.exp(-(VSTN9- thetaTauhSTN)/sigmaTauhSTN))
    taurSTN9 = taur0STN + taur1STN/(1 + np.exp(-(VSTN9 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSTN9 = 1/(1 + np.exp(-((VSTN9-30) + 39)/8))

    VGPE1 = solution[54] # GPE
    nGPE1 = solution[55]
    hGPE1 = solution[56]
    rGPE1 = solution[57]
    sGPE1 = solution[58]
    CaGPE1 = solution[59]

    ninfGPE1 = 1/(1+ np.exp(-(VGPE1-thetanGPE)/sigmanGPE))
    minfGPE1 = 1/(1+ np.exp(-(VGPE1-thetamGPE)/sigmamGPE))
    hinfGPE1 = 1/(1+ np.exp(-(VGPE1-thetahGPE)/sigmahGPE))
    ainfGPE1 = 1/(1+ np.exp(-(VGPE1-thetaaGPE)/sigmaaGPE))
    rinfGPE1 = 1/(1+ np.exp(-(VGPE1-thetarGPE)/sigmarGPE))
    sinfGPE1 =1/(1+ np.exp(-(VGPE1-thetasGPE)/sigmasGPE))

    ILGPE1 = gLGPE * (VGPE1 - vLGPE)
    IKGPE1 = gKGPE * (nGPE1 ** 4) * (VGPE1 - vKGPE)
    INaGPE1 = gNaGPE * (minfGPE1 ** 3) * hGPE1 * (VGPE1 - vNaGPE)
    ITGPE1 = gTGPE * (ainfGPE1 ** 3) * rGPE1 * (VGPE1 - vCaGPE)
    ICaGPE1 = gCaGPE * (sinfGPE1 ** 2) * (VGPE1 - vCaGPE)
    IAHPGPE1 = gAHPGPE * (VGPE1 - vKGPE) * (CaGPE1/(CaGPE1 + k1GPE))

    taunGPE1 = taun0GPE + taun1GPE/(1 + np.exp(-(VGPE1 - thetaTaunGPE)/sigmaTaunGPE))
    tauhGPE1 = tauh0GPE + tauh1GPE/(1 + np.exp(-(VGPE1 - thetaTauhGPE)/sigmaTauhGPE))

    HinfGPE1 = 1/(1 + np.exp(-((VGPE1-20) + 57)/2))

    VGPE2 = solution[60] # GPE
    nGPE2 = solution[61]
    hGPE2 = solution[62]
    rGPE2 = solution[63]
    sGPE2 = solution[64]
    CaGPE2 = solution[65]

    ninfGPE2 = 1/(1+ np.exp(-(VGPE2-thetanGPE)/sigmanGPE))
    minfGPE2 = 1/(1+ np.exp(-(VGPE2-thetamGPE)/sigmamGPE))
    hinfGPE2 = 1/(1+ np.exp(-(VGPE2-thetahGPE)/sigmahGPE))
    ainfGPE2 = 1/(1+ np.exp(-(VGPE2-thetaaGPE)/sigmaaGPE))
    rinfGPE2 = 1/(1+ np.exp(-(VGPE2-thetarGPE)/sigmarGPE))
    sinfGPE2 =1/(1+ np.exp(-(VGPE2-thetasGPE)/sigmasGPE))

    ILGPE2 = gLGPE * (VGPE2 - vLGPE)
    IKGPE2 = gKGPE * (nGPE2 ** 4) * (VGPE2 - vKGPE)
    INaGPE2 = gNaGPE * (minfGPE2 ** 3) * hGPE2 * (VGPE2 - vNaGPE)
    ITGPE2 = gTGPE * (ainfGPE2 ** 3) * rGPE2 * (VGPE2 - vCaGPE)
    ICaGPE2 = gCaGPE * (sinfGPE2 ** 2) * (VGPE2 - vCaGPE)
    IAHPGPE2 = gAHPGPE * (VGPE2 - vKGPE) * (CaGPE2/(CaGPE2 + k1GPE))

    taunGPE2 = taun0GPE + taun1GPE/(1 + np.exp(-(VGPE2 - thetaTaunGPE)/sigmaTaunGPE))
    tauhGPE2 = tauh0GPE + tauh1GPE/(1 + np.exp(-(VGPE2 - thetaTauhGPE)/sigmaTauhGPE))

    HinfGPE2 = 1/(1 + np.exp(-((VGPE2-20) + 57)/2))

    VGPE3 = solution[66] # GPE
    nGPE3 = solution[67]
    hGPE3 = solution[68]
    rGPE3 = solution[69]
    sGPE3 = solution[70]
    CaGPE3= solution[71]

    ninfGPE3 = 1/(1+ np.exp(-(VGPE3-thetanGPE)/sigmanGPE))
    minfGPE3 = 1/(1+ np.exp(-(VGPE3-thetamGPE)/sigmamGPE))
    hinfGPE3 = 1/(1+ np.exp(-(VGPE3-thetahGPE)/sigmahGPE))
    ainfGPE3 = 1/(1+ np.exp(-(VGPE3-thetaaGPE)/sigmaaGPE))
    rinfGPE3 = 1/(1+ np.exp(-(VGPE3-thetarGPE)/sigmarGPE))
    sinfGPE3 =1/(1+ np.exp(-(VGPE3-thetasGPE)/sigmasGPE))

    ILGPE3 = gLGPE * (VGPE3 - vLGPE)
    IKGPE3 = gKGPE * (nGPE3 ** 4) * (VGPE3 - vKGPE)
    INaGPE3 = gNaGPE * (minfGPE3 ** 3) * hGPE3 * (VGPE3 - vNaGPE)
    ITGPE3 = gTGPE * (ainfGPE3 ** 3) * rGPE3 * (VGPE3 - vCaGPE)
    ICaGPE3 = gCaGPE * (sinfGPE3 ** 2) * (VGPE3 - vCaGPE)
    IAHPGPE3 = gAHPGPE * (VGPE3 - vKGPE) * (CaGPE3/(CaGPE3 + k1GPE))

    taunGPE3 = taun0GPE + taun1GPE/(1 + np.exp(-(VGPE3 - thetaTaunGPE)/sigmaTaunGPE))
    tauhGPE3 = tauh0GPE + tauh1GPE/(1 + np.exp(-(VGPE3 - thetaTauhGPE)/sigmaTauhGPE))

    HinfGPE3 = 1/(1 + np.exp(-((VGPE3-20) + 57)/2))

    VGPE4 = solution[72] # GPE
    nGPE4 = solution[73]
    hGPE4 = solution[74]
    rGPE4 = solution[75]
    sGPE4 = solution[76]
    CaGPE4 = solution[77]

    ninfGPE4 = 1/(1+ np.exp(-(VGPE4-thetanGPE)/sigmanGPE))
    minfGPE4 = 1/(1+ np.exp(-(VGPE4-thetamGPE)/sigmamGPE))
    hinfGPE4 = 1/(1+ np.exp(-(VGPE4-thetahGPE)/sigmahGPE))
    ainfGPE4 = 1/(1+ np.exp(-(VGPE4-thetaaGPE)/sigmaaGPE))
    rinfGPE4 = 1/(1+ np.exp(-(VGPE4-thetarGPE)/sigmarGPE))
    sinfGPE4 =1/(1+ np.exp(-(VGPE4-thetasGPE)/sigmasGPE))

    ILGPE4 = gLGPE * (VGPE4 - vLGPE)
    IKGPE4 = gKGPE * (nGPE4 ** 4) * (VGPE4 - vKGPE)
    INaGPE4 = gNaGPE * (minfGPE4 ** 3) * hGPE4 * (VGPE4 - vNaGPE)
    ITGPE4 = gTGPE * (ainfGPE4 ** 3) * rGPE4 * (VGPE4 - vCaGPE)
    ICaGPE4 = gCaGPE * (sinfGPE4 ** 2) * (VGPE4 - vCaGPE)
    IAHPGPE4 = gAHPGPE * (VGPE4 - vKGPE) * (CaGPE4/(CaGPE4 + k1GPE))

    taunGPE4 = taun0GPE + taun1GPE/(1 + np.exp(-(VGPE4 - thetaTaunGPE)/sigmaTaunGPE))
    tauhGPE4 = tauh0GPE + tauh1GPE/(1 + np.exp(-(VGPE4 - thetaTauhGPE)/sigmaTauhGPE))
    
    HinfGPE4 = 1/(1 + np.exp(-((VGPE4-20) + 57)/2))

    VGPE5 = solution[78] # GPE
    nGPE5 = solution[79]
    hGPE5 = solution[80]
    rGPE5 = solution[81]
    sGPE5 = solution[82]
    CaGPE5 = solution[83]

    ninfGPE5 = 1/(1+ np.exp(-(VGPE5-thetanGPE)/sigmanGPE))
    minfGPE5 = 1/(1+ np.exp(-(VGPE5-thetamGPE)/sigmamGPE))
    hinfGPE5 = 1/(1+ np.exp(-(VGPE5-thetahGPE)/sigmahGPE))
    ainfGPE5 = 1/(1+ np.exp(-(VGPE5-thetaaGPE)/sigmaaGPE))
    rinfGPE5 = 1/(1+ np.exp(-(VGPE5-thetarGPE)/sigmarGPE))
    sinfGPE5 =1/(1+ np.exp(-(VGPE5-thetasGPE)/sigmasGPE))

    ILGPE5 = gLGPE * (VGPE5 - vLGPE)
    IKGPE5 = gKGPE * (nGPE5 ** 4) * (VGPE5 - vKGPE)
    INaGPE5 = gNaGPE * (minfGPE5 ** 3) * hGPE5 * (VGPE5 - vNaGPE)
    ITGPE5 = gTGPE * (ainfGPE5 ** 3) * rGPE5 * (VGPE5 - vCaGPE)
    ICaGPE5 = gCaGPE * (sinfGPE5 ** 2) * (VGPE5 - vCaGPE)
    IAHPGPE5 = gAHPGPE * (VGPE5 - vKGPE) * (CaGPE5/(CaGPE5 + k1GPE))

    taunGPE5 = taun0GPE + taun1GPE/(1 + np.exp(-(VGPE5 - thetaTaunGPE)/sigmaTaunGPE))
    tauhGPE5 = tauh0GPE + tauh1GPE/(1 + np.exp(-(VGPE5 - thetaTauhGPE)/sigmaTauhGPE))

    HinfGPE5 = 1/(1 + np.exp(-((VGPE5-20) + 57)/2))

    VGPE6 = solution[84] # GPE
    nGPE6 = solution[85]
    hGPE6 = solution[86]
    rGPE6 = solution[87]
    sGPE6 = solution[88]
    CaGPE6 = solution[89]

    ninfGPE6 = 1/(1+ np.exp(-(VGPE6-thetanGPE)/sigmanGPE))
    minfGPE6 = 1/(1+ np.exp(-(VGPE6-thetamGPE)/sigmamGPE))
    hinfGPE6 = 1/(1+ np.exp(-(VGPE6-thetahGPE)/sigmahGPE))
    ainfGPE6 = 1/(1+ np.exp(-(VGPE6-thetaaGPE)/sigmaaGPE))
    rinfGPE6 = 1/(1+ np.exp(-(VGPE6-thetarGPE)/sigmarGPE))
    sinfGPE6 =1/(1+ np.exp(-(VGPE6-thetasGPE)/sigmasGPE))

    ILGPE6 = gLGPE * (VGPE6 - vLGPE)
    IKGPE6 = gKGPE * (nGPE6 ** 4) * (VGPE6 - vKGPE)
    INaGPE6 = gNaGPE * (minfGPE6 ** 3) * hGPE6 * (VGPE6 - vNaGPE)
    ITGPE6 = gTGPE * (ainfGPE6 ** 3) * rGPE6 * (VGPE6 - vCaGPE)
    ICaGPE6 = gCaGPE * (sinfGPE6 ** 2) * (VGPE6 - vCaGPE)
    IAHPGPE6 = gAHPGPE * (VGPE6 - vKGPE) * (CaGPE6/(CaGPE6 + k1GPE))

    taunGPE6 = taun0GPE + taun1GPE/(1 + np.exp(-(VGPE6 - thetaTaunGPE)/sigmaTaunGPE))
    tauhGPE6 = tauh0GPE + tauh1GPE/(1 + np.exp(-(VGPE6 - thetaTauhGPE)/sigmaTauhGPE))

    HinfGPE6 = 1/(1 + np.exp(-((VGPE6-20) + 57)/2))

    VGPE7 = solution[90] # GPE
    nGPE7 = solution[91]
    hGPE7 = solution[92]
    rGPE7 = solution[93]
    sGPE7 = solution[94]
    CaGPE7 = solution[95]

    ninfGPE7 = 1/(1+ np.exp(-(VGPE7-thetanGPE)/sigmanGPE))
    minfGPE7 = 1/(1+ np.exp(-(VGPE7-thetamGPE)/sigmamGPE))
    hinfGPE7 = 1/(1+ np.exp(-(VGPE7-thetahGPE)/sigmahGPE))
    ainfGPE7 = 1/(1+ np.exp(-(VGPE7-thetaaGPE)/sigmaaGPE))
    rinfGPE7 = 1/(1+ np.exp(-(VGPE7-thetarGPE)/sigmarGPE))
    sinfGPE7 = 1/(1+ np.exp(-(VGPE7-thetasGPE)/sigmasGPE))

    ILGPE7 = gLGPE * (VGPE7 - vLGPE)
    IKGPE7 = gKGPE * (nGPE7 ** 4) * (VGPE7 - vKGPE)
    INaGPE7 = gNaGPE * (minfGPE7 ** 3) * hGPE7 * (VGPE7 - vNaGPE)
    ITGPE7 = gTGPE * (ainfGPE7 ** 3) * rGPE7 * (VGPE7 - vCaGPE)
    ICaGPE7 = gCaGPE * (sinfGPE7 ** 2) * (VGPE7 - vCaGPE)
    IAHPGPE7 = gAHPGPE * (VGPE7 - vKGPE) * (CaGPE7/(CaGPE7 + k1GPE))

    taunGPE7 = taun0GPE + taun1GPE/(1 + np.exp(-(VGPE7 - thetaTaunGPE)/sigmaTaunGPE))
    tauhGPE7 = tauh0GPE + tauh1GPE/(1 + np.exp(-(VGPE7 - thetaTauhGPE)/sigmaTauhGPE))

    HinfGPE7 = 1/(1 + np.exp(-((VGPE7-20) + 57)/2))
   
    VGPE8 = solution[96] # GPE
    nGPE8 = solution[97]
    hGPE8 = solution[98]
    rGPE8 = solution[99]
    sGPE8 = solution[100]
    CaGPE8 = solution[101]

    ninfGPE8 = 1/(1+ np.exp(-(VGPE8-thetanGPE)/sigmanGPE))
    minfGPE8 = 1/(1+ np.exp(-(VGPE8-thetamGPE)/sigmamGPE))
    hinfGPE8 = 1/(1+ np.exp(-(VGPE8-thetahGPE)/sigmahGPE))
    ainfGPE8 = 1/(1+ np.exp(-(VGPE8-thetaaGPE)/sigmaaGPE))
    rinfGPE8 = 1/(1+ np.exp(-(VGPE8-thetarGPE)/sigmarGPE))
    sinfGPE8 =1/(1+ np.exp(-(VGPE8-thetasGPE)/sigmasGPE))

    ILGPE8 = gLGPE * (VGPE8 - vLGPE)
    IKGPE8 = gKGPE * (nGPE8 ** 4) * (VGPE8 - vKGPE)
    INaGPE8 = gNaGPE * (minfGPE8 ** 3) * hGPE8 * (VGPE8 - vNaGPE)
    ITGPE8 = gTGPE * (ainfGPE8 ** 3) * rGPE8 * (VGPE8 - vCaGPE)
    ICaGPE8 = gCaGPE * (sinfGPE8 ** 2) * (VGPE8 - vCaGPE)
    IAHPGPE8 = gAHPGPE * (VGPE8 - vKGPE) * (CaGPE8/(CaGPE8 + k1GPE))

    taunGPE8 = taun0GPE + taun1GPE/(1 + np.exp(-(VGPE8 - thetaTaunGPE)/sigmaTaunGPE))
    tauhGPE8 = tauh0GPE + tauh1GPE/(1 + np.exp(-(VGPE8 - thetaTauhGPE)/sigmaTauhGPE))

    HinfGPE8 = 1/(1 + np.exp(-((VGPE8-20) + 57)/2))
   
    VGPE9 = solution[102] # GPE
    nGPE9 = solution[103]
    hGPE9 = solution[104]
    rGPE9 = solution[105]
    sGPE9 = solution[106]
    CaGPE9 = solution[107]

    ninfGPE9 = 1/(1+ np.exp(-(VGPE9-thetanGPE)/sigmanGPE))
    minfGPE9 = 1/(1+ np.exp(-(VGPE9-thetamGPE)/sigmamGPE))
    hinfGPE9 = 1/(1+ np.exp(-(VGPE9-thetahGPE)/sigmahGPE))
    ainfGPE9 = 1/(1+ np.exp(-(VGPE9-thetaaGPE)/sigmaaGPE))
    rinfGPE9 = 1/(1+ np.exp(-(VGPE9-thetarGPE)/sigmarGPE))
    sinfGPE9 =1/(1+ np.exp(-(VGPE9 -thetasGPE)/sigmasGPE))

    ILGPE9 = gLGPE * (VGPE9 - vLGPE)
    IKGPE9 = gKGPE * (nGPE9 ** 4) * (VGPE9 - vKGPE)
    INaGPE9 = gNaGPE * (minfGPE9 ** 3) * hGPE9 * (VGPE9 - vNaGPE)
    ITGPE9 = gTGPE * (ainfGPE9 ** 3) * rGPE9 * (VGPE9- vCaGPE)
    ICaGPE9 = gCaGPE * (sinfGPE9 ** 2) * (VGPE9 - vCaGPE)
    IAHPGPE9 = gAHPGPE * (VGPE9 - vKGPE) * (CaGPE9/(CaGPE9 + k1GPE))

    taunGPE9 = taun0GPE + taun1GPE/(1 + np.exp(-(VGPE9 - thetaTaunGPE)/sigmaTaunGPE))
    tauhGPE9 = tauh0GPE + tauh1GPE/(1 + np.exp(-(VGPE9 - thetaTauhGPE)/sigmaTauhGPE))

    HinfGPE9 = 1/(1 + np.exp(-((VGPE9-20) + 57)/2))

    VSNC1 = solution[108]
    nSNC1 = solution[109]
    hSNC1 = solution[110]
    rSNC1 = solution[111]
    sSNC1 = solution[112]
    CaSNC1 = solution[113]

    ninfSNC1 = 1/(1+ np.exp( -(VSNC1-thetanSTN)/sigmanSTN))
    minfSNC1 = 1/(1+ np.exp(-(VSNC1-thetamSTN)/sigmamSTN))
    hinfSNC1 = 1/(1+ np.exp(-(VSNC1-thetahSTN)/sigmahSTN))
    ainfSNC1 = 1/(1+ np.exp(-(VSNC1-thetaaSTN)/sigmaaSTN))
    rinfSNC1 = 1/(1+ np.exp(-(VSNC1-thetarSTN)/sigmarSTN))
    sinfSNC1 = 1/(1+ np.exp(-(VSNC1-thetasSTN)/sigmasSTN))
    binfSNC1 = 1/(1+ np.exp((rSNC1 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSNC1 = gLSTN * (VSNC1 - vLSTN) # PART OF IT
    IKSNC1 = gKSTN * (nSNC1 ** 4) * (VSNC1 - vKSTN) # PART OF IT
    INaSNC1 = gNaSTN * (minfSNC1 ** 3) * hSNC1 * (VSNC1 - vNaSTN) # PART OF IT
    ICaSNC1 = gCaSTN * (sinfSNC1 ** 2) * (VSNC1 - vCaSTN) # PART OF IT
    IAHPSNC1 = gAHPSTN * (VSNC1 - vKSTN) * (CaSNC1/(CaSNC1 + k1STN)) # PART OF IT

    taunSNC1 = taun0STN + taun1STN/(1 + np.exp(-(VSNC1 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSNC1 = tauh0STN + tauh1STN/(1 + np.exp(-(VSNC1- thetaTauhSTN)/sigmaTauhSTN))
    taurSNC1 = taur0STN + taur1STN/(1 + np.exp(-(VSNC1 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSNC1 = 1/(1 + np.exp(-((VSNC1-30) + 39)/8))

    VSNC2 = solution[114]
    nSNC2 = solution[115]
    hSNC2 = solution[116]
    rSNC2 = solution[117]
    sSNC2 = solution[118]
    CaSNC2 = solution[119]

    ninfSNC2 = 1/(1+ np.exp( -(VSNC2-thetanSTN)/sigmanSTN))
    minfSNC2 = 1/(1+ np.exp(-(VSNC2-thetamSTN)/sigmamSTN))
    hinfSNC2 = 1/(1+ np.exp(-(VSNC2-thetahSTN)/sigmahSTN))
    ainfSNC2 = 1/(1+ np.exp(-(VSNC2-thetaaSTN)/sigmaaSTN))
    rinfSNC2 = 1/(1+ np.exp(-(VSNC2-thetarSTN)/sigmarSTN))
    sinfSNC2 = 1/(1+ np.exp(-(VSNC2-thetasSTN)/sigmasSTN))
    binfSNC2 = 1/(1+ np.exp((rSNC2 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSNC2 = gLSTN * (VSNC2 - vLSTN) # PART OF IT
    IKSNC2 = gKSTN * (nSNC2 ** 4) * (VSNC2 - vKSTN) # PART OF IT
    INaSNC2 = gNaSTN * (minfSNC2 ** 3) * hSNC2 * (VSNC2 - vNaSTN) # PART OF IT
    ICaSNC2 = gCaSTN * (sinfSNC2 ** 2) * (VSNC2 - vCaSTN) # PART OF IT
    IAHPSNC2 = gAHPSTN * (VSNC2 - vKSTN) * (CaSNC2/(CaSNC2 + k1STN)) # PART OF IT

    taunSNC2 = taun0STN + taun1STN/(1 + np.exp(-(VSNC2 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSNC2 = tauh0STN + tauh1STN/(1 + np.exp(-(VSNC2- thetaTauhSTN)/sigmaTauhSTN))
    taurSNC2 = taur0STN + taur1STN/(1 + np.exp(-(VSNC2 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSNC2 = 1/(1 + np.exp(-((VSNC2-30) + 39)/8))

    VSNC3 = solution[120]
    nSNC3 = solution[121]
    hSNC3 = solution[122]
    rSNC3 = solution[123]
    sSNC3 = solution[124]
    CaSNC3 = solution[125]

    ninfSNC3 = 1/(1+ np.exp( -(VSNC3-thetanSTN)/sigmanSTN))
    minfSNC3 = 1/(1+ np.exp(-(VSNC3-thetamSTN)/sigmamSTN))
    hinfSNC3 = 1/(1+ np.exp(-(VSNC3-thetahSTN)/sigmahSTN))
    ainfSNC3 = 1/(1+ np.exp(-(VSNC3-thetaaSTN)/sigmaaSTN))
    rinfSNC3 = 1/(1+ np.exp(-(VSNC3-thetarSTN)/sigmarSTN))
    sinfSNC3 = 1/(1+ np.exp(-(VSNC3-thetasSTN)/sigmasSTN))
    binfSNC3 = 1/(1+ np.exp((rSNC3 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSNC3 = gLSTN * (VSNC3 - vLSTN) # PART OF IT
    IKSNC3 = gKSTN * (nSNC3 ** 4) * (VSNC3 - vKSTN) # PART OF IT
    INaSNC3 = gNaSTN * (minfSNC3 ** 3) * hSNC3 * (VSNC3 - vNaSTN) # PART OF IT
    ICaSNC3 = gCaSTN * (sinfSNC3 ** 2) * (VSNC3 - vCaSTN) # PART OF IT
    IAHPSNC3 = gAHPSTN * (VSNC3 - vKSTN) * (CaSNC3/(CaSNC3 + k1STN)) # PART OF IT

    taunSNC3 = taun0STN + taun1STN/(1 + np.exp(-(VSNC3 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSNC3 = tauh0STN + tauh1STN/(1 + np.exp(-(VSNC3- thetaTauhSTN)/sigmaTauhSTN))
    taurSNC3 = taur0STN + taur1STN/(1 + np.exp(-(VSNC3 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSNC3 = 1/(1 + np.exp(-((VSNC3-30) + 39)/8))

    VSNC4 = solution[126]
    nSNC4 = solution[127]
    hSNC4 = solution[128]
    rSNC4 = solution[129]
    sSNC4 = solution[130]
    CaSNC4 = solution[131]

    ninfSNC4 = 1/(1+ np.exp( -(VSNC1-thetanSTN)/sigmanSTN))
    minfSNC4 = 1/(1+ np.exp(-(VSNC1-thetamSTN)/sigmamSTN))
    hinfSNC4 = 1/(1+ np.exp(-(VSNC1-thetahSTN)/sigmahSTN))
    ainfSNC4 = 1/(1+ np.exp(-(VSNC1-thetaaSTN)/sigmaaSTN))
    rinfSNC4 = 1/(1+ np.exp(-(VSNC1-thetarSTN)/sigmarSTN))
    sinfSNC4 = 1/(1+ np.exp(-(VSNC1-thetasSTN)/sigmasSTN))
    binfSNC4 = 1/(1+ np.exp((rSNC1 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSNC4 = gLSTN * (VSNC4 - vLSTN) # PART OF IT
    IKSNC4 = gKSTN * (nSNC4 ** 4) * (VSNC4 - vKSTN) # PART OF IT
    INaSNC4 = gNaSTN * (minfSNC4 ** 3) * hSNC4 * (VSNC4 - vNaSTN) # PART OF IT
    ICaSNC4 = gCaSTN * (sinfSNC4 ** 2) * (VSNC4 - vCaSTN) # PART OF IT
    IAHPSNC4 = gAHPSTN * (VSNC4 - vKSTN) * (CaSNC4/(CaSNC4 + k1STN)) # PART OF IT

    taunSNC4 = taun0STN + taun1STN/(1 + np.exp(-(VSNC4 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSNC4 = tauh0STN + tauh1STN/(1 + np.exp(-(VSNC4 - thetaTauhSTN)/sigmaTauhSTN))
    taurSNC4 = taur0STN + taur1STN/(1 + np.exp(-(VSNC4 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSNC4 = 1/(1 + np.exp(-((VSNC4-30) + 39)/8))

    VSNC5 = solution[132]
    nSNC5 = solution[133]
    hSNC5 = solution[134]
    rSNC5 = solution[135]
    sSNC5 = solution[136]
    CaSNC5 = solution[137]

    ninfSNC5 = 1/(1+ np.exp( -(VSNC5-thetanSTN)/sigmanSTN))
    minfSNC5 = 1/(1+ np.exp(-(VSNC5-thetamSTN)/sigmamSTN))
    hinfSNC5 = 1/(1+ np.exp(-(VSNC5-thetahSTN)/sigmahSTN))
    ainfSNC5 = 1/(1+ np.exp(-(VSNC5-thetaaSTN)/sigmaaSTN))
    rinfSNC5 = 1/(1+ np.exp(-(VSNC5-thetarSTN)/sigmarSTN))
    sinfSNC5 = 1/(1+ np.exp(-(VSNC5-thetasSTN)/sigmasSTN))
    binfSNC5 = 1/(1+ np.exp((rSNC5 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSNC5 = gLSTN * (VSNC5 - vLSTN) # PART OF IT
    IKSNC5 = gKSTN * (nSNC5 ** 4) * (VSNC5 - vKSTN) # PART OF IT
    INaSNC5 = gNaSTN * (minfSNC5 ** 3) * hSNC5 * (VSNC5 - vNaSTN) # PART OF IT
    ICaSNC5 = gCaSTN * (sinfSNC5 ** 2) * (VSNC5 - vCaSTN) # PART OF IT
    IAHPSNC5 = gAHPSTN * (VSNC5 - vKSTN) * (CaSNC5/(CaSNC5 + k1STN)) # PART OF IT

    taunSNC5 = taun0STN + taun1STN/(1 + np.exp(-(VSNC5 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSNC5 = tauh0STN + tauh1STN/(1 + np.exp(-(VSNC5- thetaTauhSTN)/sigmaTauhSTN))
    taurSNC5 = taur0STN + taur1STN/(1 + np.exp(-(VSNC5 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSNC5 = 1/(1 + np.exp(-((VSNC1-30) + 39)/8))

    VSNC6 = solution[138]
    nSNC6 = solution[139]
    hSNC6 = solution[140]
    rSNC6 = solution[141]
    sSNC6 = solution[142]
    CaSNC6 = solution[143]

    ninfSNC6 = 1/(1+ np.exp( -(VSNC6-thetanSTN)/sigmanSTN))
    minfSNC6 = 1/(1+ np.exp(-(VSNC6-thetamSTN)/sigmamSTN))
    hinfSNC6 = 1/(1+ np.exp(-(VSNC6-thetahSTN)/sigmahSTN))
    ainfSNC6 = 1/(1+ np.exp(-(VSNC6-thetaaSTN)/sigmaaSTN))
    rinfSNC6 = 1/(1+ np.exp(-(VSNC6-thetarSTN)/sigmarSTN))
    sinfSNC6 = 1/(1+ np.exp(-(VSNC6-thetasSTN)/sigmasSTN))
    binfSNC6 = 1/(1+ np.exp((rSNC6 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSNC6 = gLSTN * (VSNC6 - vLSTN) # PART OF IT
    IKSNC6 = gKSTN * (nSNC6 ** 4) * (VSNC6 - vKSTN) # PART OF IT
    INaSNC6 = gNaSTN * (minfSNC6 ** 3) * hSNC6 * (VSNC6 - vNaSTN) # PART OF IT
    ICaSNC6 = gCaSTN * (sinfSNC6 ** 2) * (VSNC6 - vCaSTN) # PART OF IT
    IAHPSNC6 = gAHPSTN * (VSNC6 - vKSTN) * (CaSNC6/(CaSNC6 + k1STN)) # PART OF IT

    taunSNC6 = taun0STN + taun1STN/(1 + np.exp(-(VSNC6 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSNC6 = tauh0STN + tauh1STN/(1 + np.exp(-(VSNC6- thetaTauhSTN)/sigmaTauhSTN))
    taurSNC6 = taur0STN + taur1STN/(1 + np.exp(-(VSNC6 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSNC6 = 1/(1 + np.exp(-((VSNC6 -30) + 39)/8))

    VSNC7 = solution[144]
    nSNC7 = solution[145]
    hSNC7 = solution[146]
    rSNC7 = solution[147]
    sSNC7 = solution[148]
    CaSNC7 = solution[149]

    ninfSNC7 = 1/(1+ np.exp( -(VSNC7-thetanSTN)/sigmanSTN))
    minfSNC7 = 1/(1+ np.exp(-(VSNC7-thetamSTN)/sigmamSTN))
    hinfSNC7 = 1/(1+ np.exp(-(VSNC7-thetahSTN)/sigmahSTN))
    ainfSNC7 = 1/(1+ np.exp(-(VSNC7-thetaaSTN)/sigmaaSTN))
    rinfSNC7 = 1/(1+ np.exp(-(VSNC7-thetarSTN)/sigmarSTN))
    sinfSNC7 = 1/(1+ np.exp(-(VSNC7-thetasSTN)/sigmasSTN))
    binfSNC7 = 1/(1+ np.exp((rSNC7 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSNC7 = gLSTN * (VSNC7 - vLSTN) # PART OF IT
    IKSNC7 = gKSTN * (nSNC7 ** 4) * (VSNC7 - vKSTN) # PART OF IT
    INaSNC7 = gNaSTN * (minfSNC7 ** 3) * hSNC7 * (VSNC7 - vNaSTN) # PART OF IT
    ICaSNC7 = gCaSTN * (sinfSNC7 ** 2) * (VSNC7 - vCaSTN) # PART OF IT
    IAHPSNC7 = gAHPSTN * (VSNC7 - vKSTN) * (CaSNC7/(CaSNC7 + k1STN)) # PART OF IT

    taunSNC7 = taun0STN + taun1STN/(1 + np.exp(-(VSNC7 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSNC7  = tauh0STN + tauh1STN/(1 + np.exp(-(VSNC7 - thetaTauhSTN)/sigmaTauhSTN))
    taurSNC7 = taur0STN + taur1STN/(1 + np.exp(-(VSNC7 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSNC7 = 1/(1 + np.exp(-((VSNC7-30) + 39)/8))

    VSNC8 = solution[150]
    nSNC8 = solution[151]
    hSNC8 = solution[152]
    rSNC8 = solution[153]
    sSNC8 = solution[154]
    CaSNC8 = solution[155]

    ninfSNC8 = 1/(1+ np.exp( -(VSNC8-thetanSTN)/sigmanSTN))
    minfSNC8 = 1/(1+ np.exp(-(VSNC8-thetamSTN)/sigmamSTN))
    hinfSNC8 = 1/(1+ np.exp(-(VSNC8-thetahSTN)/sigmahSTN))
    ainfSNC8 = 1/(1+ np.exp(-(VSNC8-thetaaSTN)/sigmaaSTN))
    rinfSNC8 = 1/(1+ np.exp(-(VSNC8-thetarSTN)/sigmarSTN))
    sinfSNC8 = 1/(1+ np.exp(-(VSNC8-thetasSTN)/sigmasSTN))
    binfSNC8 = 1/(1+ np.exp((rSNC8 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSNC8 = gLSTN * (VSNC8 - vLSTN) # PART OF IT
    IKSNC8 = gKSTN * (nSNC8 ** 4) * (VSNC8 - vKSTN) # PART OF IT
    INaSNC8 = gNaSTN * (minfSNC8 ** 3) * hSNC8 * (VSNC8 - vNaSTN) # PART OF IT
    ICaSNC8 = gCaSTN * (sinfSNC8 ** 2) * (VSNC8 - vCaSTN) # PART OF IT
    IAHPSNC8 = gAHPSTN * (VSNC8 - vKSTN) * (CaSNC8/(CaSNC8 + k1STN)) # PART OF IT

    taunSNC8 = taun0STN + taun1STN/(1 + np.exp(-(VSNC8 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSNC8 = tauh0STN + tauh1STN/(1 + np.exp(-(VSNC8- thetaTauhSTN)/sigmaTauhSTN))
    taurSNC8 = taur0STN + taur1STN/(1 + np.exp(-(VSNC8 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSNC8 = 1/(1 + np.exp(-((VSNC8-30) + 39)/8))

    VSNC9 = solution[156]
    nSNC9 = solution[157]
    hSNC9 = solution[158]
    rSNC9 = solution[159]
    sSNC9 = solution[160]
    CaSNC9 = solution[161]

    ninfSNC9 = 1/(1+ np.exp( -(VSNC9-thetanSTN)/sigmanSTN))
    minfSNC9 = 1/(1+ np.exp(-(VSNC9-thetamSTN)/sigmamSTN))
    hinfSNC9 = 1/(1+ np.exp(-(VSNC9-thetahSTN)/sigmahSTN))
    ainfSNC9 = 1/(1+ np.exp(-(VSNC9-thetaaSTN)/sigmaaSTN))
    rinfSNC9 = 1/(1+ np.exp(-(VSNC9-thetarSTN)/sigmarSTN))
    sinfSNC9 = 1/(1+ np.exp(-(VSNC9- thetasSTN)/sigmasSTN))
    binfSNC9 = 1/(1+ np.exp((rSNC9 - thetabSTN)/sigmabSTN)) - 1/(1 + np.exp(-thetabSTN/sigmabSTN))

    ILSNC9 = gLSTN * (VSNC9 - vLSTN) # PART OF IT
    IKSNC9 = gKSTN * (nSNC9 ** 4) * (VSNC9 - vKSTN) # PART OF IT
    INaSNC9 = gNaSTN * (minfSNC9 ** 3) * hSNC9 * (VSNC9 - vNaSTN) # PART OF IT
    ICaSNC9 = gCaSTN * (sinfSNC9 ** 2) * (VSNC9 - vCaSTN) # PART OF IT
    IAHPSNC9 = gAHPSTN * (VSNC9 - vKSTN) * (CaSNC9/(CaSNC9 + k1STN)) # PART OF IT

    taunSNC9 = taun0STN + taun1STN/(1 + np.exp(-(VSNC9 - thetaTaunSTN)/sigmaTaunSTN))
    tauhSNC9 = tauh0STN + tauh1STN/(1 + np.exp(-(VSNC9 - thetaTauhSTN)/sigmaTauhSTN))
    taurSNC9 = taur0STN + taur1STN/(1 + np.exp(-(VSNC9 - thetaTaurSTN)/sigmaTaurSTN))

    HinfSNC9 = 1/(1 + np.exp(-((VSNC9-30) + 39)/8))

    #gpe to stn

    I_GPE1_to_STN3 = g_GPE_to_STN * (VSTN3 - vGPE_to_STN) * sGPE1 
    I_GPE1_to_STN8 = g_GPE_to_STN * (VSTN8 - vGPE_to_STN) * sGPE1 

    I_GPE2_to_STN4 = g_GPE_to_STN * (VSTN4 - vGPE_to_STN) * sGPE2 
    I_GPE2_to_STN9 = g_GPE_to_STN * (VSTN9 - vGPE_to_STN) * sGPE2 
    
    I_GPE3_to_STN5 = g_GPE_to_STN * (VSTN5 - vGPE_to_STN) * sGPE3 
    I_GPE3_to_STN1 = g_GPE_to_STN * (VSTN1 - vGPE_to_STN) * sGPE3 

    I_GPE4_to_STN6 = g_GPE_to_STN * (VSTN6 - vGPE_to_STN) * sGPE4 
    I_GPE4_to_STN2 = g_GPE_to_STN * (VSTN2 - vGPE_to_STN) * sGPE4 

    I_GPE5_to_STN7 = g_GPE_to_STN * (VSTN7 - vGPE_to_STN) * sGPE5 
    I_GPE5_to_STN3 = g_GPE_to_STN * (VSTN3 - vGPE_to_STN) * sGPE5 

    I_GPE6_to_STN4 = g_GPE_to_STN * (VSTN4 - vGPE_to_STN) * sGPE6  
    I_GPE6_to_STN8 = g_GPE_to_STN * (VSTN8 - vGPE_to_STN) * sGPE6 
    
    I_GPE7_to_STN5 = g_GPE_to_STN * (VSTN5 - vGPE_to_STN) * sGPE7 
    I_GPE7_to_STN9 = g_GPE_to_STN * (VSTN9 - vGPE_to_STN) * sGPE7 

    I_GPE8_to_STN6 = g_GPE_to_STN * (VSTN6 - vGPE_to_STN) * sGPE8 
    I_GPE8_to_STN1 = g_GPE_to_STN * (VSTN1 - vGPE_to_STN) * sGPE8 
    
    I_GPE9_to_STN7 = g_GPE_to_STN * (VSTN7 - vGPE_to_STN) * sGPE9 
    I_GPE9_to_STN2 = g_GPE_to_STN * (VSTN2 - vGPE_to_STN) * sGPE9 
    
    # stn to gpe

    I_STN1_to_GPE1 = g_STN_to_GPE * (VGPE1 - vSTN_to_GPE) * sSTN1 
    I_STN2_to_GPE2  = g_STN_to_GPE * (VGPE2 - vSTN_to_GPE) * sSTN2 
    I_STN3_to_GPE3 = g_STN_to_GPE * (VGPE3 - vSTN_to_GPE) * sSTN3 
    I_STN4_to_GPE4 = g_STN_to_GPE * (VGPE4 - vSTN_to_GPE) * sSTN4 
    I_STN5_to_GPE5 = g_STN_to_GPE * (VGPE5 - vSTN_to_GPE) * sSTN5 
    I_STN6_to_GPE6 = g_STN_to_GPE * (VGPE6 - vSTN_to_GPE) * sSTN6 
    I_STN7_to_GPE7 = g_STN_to_GPE * (VGPE7 - vSTN_to_GPE) * sSTN7 
    I_STN8_to_GPE8 = g_STN_to_GPE * (VGPE8 - vSTN_to_GPE) * sSTN8 
    I_STN9_to_GPE9 = g_STN_to_GPE * (VGPE9 - vSTN_to_GPE) * sSTN9 

    #gpe to gpe

    I_GPE1_to_GPE2 = g_GPE_to_GPE * (VGPE2 - vGPE_to_GPE) * sGPE1
    I_GPE1_to_GPE9 = g_GPE_to_GPE * (VGPE9 - vGPE_to_GPE) * sGPE1

    I_GPE2_to_GPE1 = g_GPE_to_GPE * (VGPE1 - vGPE_to_GPE) * sGPE2 
    I_GPE2_to_GPE3 = g_GPE_to_GPE * (VGPE3 - vGPE_to_GPE) * sGPE2

    I_GPE3_to_GPE2 = g_GPE_to_GPE * (VGPE2 - vGPE_to_GPE) * sGPE3
    I_GPE3_to_GPE4 = g_GPE_to_GPE * (VGPE4 - vGPE_to_GPE) * sGPE3

    I_GPE4_to_GPE3 = g_GPE_to_GPE * (VGPE3 - vGPE_to_GPE) * sGPE4
    I_GPE4_to_GPE5 = g_GPE_to_GPE * (VGPE5 - vGPE_to_GPE) * sGPE4

    I_GPE5_to_GPE4 = g_GPE_to_GPE * (VGPE4 - vGPE_to_GPE) * sGPE5
    I_GPE5_to_GPE6 = g_GPE_to_GPE * (VGPE6 - vGPE_to_GPE) * sGPE5

    I_GPE6_to_GPE5 = g_GPE_to_GPE * (VGPE5 - vGPE_to_GPE) * sGPE6
    I_GPE6_to_GPE7 = g_GPE_to_GPE * (VGPE7 - vGPE_to_GPE) * sGPE6
    
    I_GPE7_to_GPE6 = g_GPE_to_GPE * (VGPE6 - vGPE_to_GPE) * sGPE7
    I_GPE7_to_GPE8 = g_GPE_to_GPE * (VGPE8 - vGPE_to_GPE) * sGPE7

    I_GPE8_to_GPE7 = g_GPE_to_GPE * (VGPE7 - vGPE_to_GPE) * sGPE8
    I_GPE8_to_GPE9 = g_GPE_to_GPE * (VGPE9 - vGPE_to_GPE) * sGPE8

    I_GPE9_to_GPE1 = g_GPE_to_GPE * (VGPE1 - vGPE_to_GPE) * sGPE9 
    I_GPE9_to_GPE8 = g_GPE_to_GPE * (VGPE8 - vGPE_to_GPE) * sGPE9

    #snc to gpe

    I_SNC1_to_GPE1 = g_STN_to_GPE * (VGPE1 - vSTN_to_GPE) * sSNC1 
    I_SNC2_to_GPE2  = g_STN_to_GPE * (VGPE2 - vSTN_to_GPE) * sSNC2 
    I_SNC3_to_GPE3 = g_STN_to_GPE * (VGPE3 - vSTN_to_GPE) * sSNC3 
    I_SNC4_to_GPE4 = g_STN_to_GPE * (VGPE4 - vSTN_to_GPE) * sSNC4 
    I_SNC5_to_GPE5 = g_STN_to_GPE * (VGPE5 - vSTN_to_GPE) * sSNC5 
    I_SNC6_to_GPE6 = g_STN_to_GPE * (VGPE6 - vSTN_to_GPE) * sSNC6 
    I_SNC7_to_GPE7 = g_STN_to_GPE * (VGPE7 - vSTN_to_GPE) * sSNC7 
    I_SNC8_to_GPE8 = g_STN_to_GPE * (VGPE8 - vSTN_to_GPE) * sSNC8 
    I_SNC9_to_GPE9 = g_STN_to_GPE * (VGPE9 - vSTN_to_GPE) * sSNC9 
    
    I_App = I_app

    output = [0] * 162

    output[0] = (-ILSTN1 - IKSTN1 - INaSTN1 - ITSTN1 - ICaSTN1 - IAHPSTN1 - I_GPE3_to_STN1 - I_GPE8_to_STN1)/C_m
    output[1] = phinSTN * (ninfSTN1 - nSTN1)/taunSTN1
    output[2] = phihSTN * (hinfSTN1 - hSTN1)/tauhSTN1
    output[3] = phirSTN * (rinfSTN1 - rSTN1)/taurSTN1
    output[4] = 5 * HinfSTN1 * (1-sSTN1) - 1 * sSTN1
    output[5] = epsilonSTN * (-ICaSTN1 - ITSTN1 - kCaSTN * CaSTN1) # Calcium

    output[6] = (-ILSTN2 - IKSTN2 - INaSTN2 - ITSTN2 - ICaSTN2 - IAHPSTN2 - I_GPE9_to_STN2 - I_GPE4_to_STN2)/C_m
    output[7] = phinSTN * (ninfSTN2 - nSTN2)/taunSTN2
    output[8] = phihSTN * (hinfSTN2 - hSTN2)/tauhSTN2
    output[9] = phirSTN * (rinfSTN2 - rSTN2)/taurSTN2
    output[10] = 5 * HinfSTN2 * (1-sSTN2) - 1 * sSTN2
    output[11] = epsilonSTN * (-ICaSTN2 - ITSTN2 - kCaSTN * CaSTN2) # Calcium

    output[12] = (-ILSTN3 - IKSTN3 - INaSTN3 - ITSTN3 - ICaSTN3 - IAHPSTN3 - I_GPE1_to_STN3 - I_GPE5_to_STN3)/C_m
    output[13] = phinSTN * (ninfSTN3 - nSTN3)/taunSTN3
    output[14] = phihSTN * (hinfSTN3 - hSTN3)/tauhSTN3
    output[15] = phirSTN * (rinfSTN3 - rSTN3)/taurSTN3
    output[16] = 5 * HinfSTN3 * (1-sSTN3) - 1 * sSTN3
    output[17] = epsilonSTN * (-ICaSTN3 - ITSTN3 - kCaSTN * CaSTN3) # Calcium

    output[18] = (-ILSTN4 - IKSTN4 - INaSTN4 - ITSTN4 - ICaSTN4 - IAHPSTN4 - I_GPE2_to_STN4 - I_GPE6_to_STN4)/C_m
    output[19] = phinSTN * (ninfSTN4 - nSTN4)/taunSTN4
    output[20] = phihSTN * (hinfSTN4 - hSTN4)/tauhSTN4
    output[21] = phirSTN * (rinfSTN4 - rSTN4)/taurSTN4
    output[22] = 5 * HinfSTN4 * (1-sSTN4) - 1 * sSTN4
    output[23] = epsilonSTN * (-ICaSTN4 - ITSTN4 - kCaSTN * CaSTN4) # Calcium

    output[24] = (-ILSTN5 - IKSTN5 - INaSTN5 - ITSTN5 - ICaSTN5 - IAHPSTN5 - I_GPE3_to_STN5 - I_GPE7_to_STN5)/C_m
    output[25] = phinSTN * (ninfSTN5 - nSTN5)/taunSTN5
    output[26] = phihSTN * (hinfSTN5 - hSTN5)/tauhSTN5
    output[27] = phirSTN * (rinfSTN5 - rSTN5)/taurSTN5
    output[28] = 5 * HinfSTN5 * (1-sSTN5) - 1 * sSTN5
    output[29] = epsilonSTN * (-ICaSTN5 - ITSTN5 - kCaSTN * CaSTN5) # Calcium

    output[30] = (-ILSTN6 - IKSTN6- INaSTN6 - ITSTN6 - ICaSTN6 - IAHPSTN6 - I_GPE8_to_STN6 - I_GPE4_to_STN6)/C_m
    output[31] = phinSTN * (ninfSTN6 - nSTN6)/taunSTN6
    output[32] = phihSTN * (hinfSTN6 - hSTN6)/tauhSTN6
    output[33] = phirSTN * (rinfSTN6 - rSTN6)/taurSTN6
    output[34] = 5 * HinfSTN6 * (1-sSTN6) - 1 * sSTN6
    output[35] = epsilonSTN * (-ICaSTN6 - ITSTN6 - kCaSTN * CaSTN6) # Calcium

    output[36] = (-ILSTN7 - IKSTN7 - INaSTN7 - ITSTN7 - ICaSTN7 - IAHPSTN7 - I_GPE9_to_STN7 - I_GPE5_to_STN7)/C_m
    output[37] = phinSTN * (ninfSTN7 - nSTN7)/taunSTN7
    output[38] = phihSTN * (hinfSTN7 - hSTN7)/tauhSTN7
    output[39] = phirSTN * (rinfSTN7 - rSTN7)/taurSTN7
    output[40] = 5 * HinfSTN7 * (1-sSTN7) - 1 * sSTN7
    output[41] = epsilonSTN * (-ICaSTN7 - ITSTN7 - kCaSTN * CaSTN7) # Calcium

    output[42] = (-ILSTN8 - IKSTN8 - INaSTN8 - ITSTN8 - ICaSTN8 - IAHPSTN8 - I_GPE1_to_STN8 - I_GPE6_to_STN8)/C_m
    output[43] = phinSTN * (ninfSTN8 - nSTN8)/taunSTN8
    output[44] = phihSTN * (hinfSTN8 - hSTN8)/tauhSTN8
    output[45] = phirSTN * (rinfSTN8 - rSTN8)/taurSTN8
    output[46] = 5 * HinfSTN8 * (1-sSTN8) - 1 * sSTN8
    output[47] = epsilonSTN * (-ICaSTN8 - ITSTN8 - kCaSTN * CaSTN8) # Calcium

    output[48] = (-ILSTN9 - IKSTN9 - INaSTN9 - ITSTN9 - ICaSTN9 - IAHPSTN9 - I_GPE2_to_STN9 - I_GPE7_to_STN9)/C_m
    output[49] = phinSTN * (ninfSTN9 - nSTN9)/taunSTN9
    output[50] = phihSTN * (hinfSTN9 - hSTN9)/tauhSTN9
    output[51] = phirSTN * (rinfSTN9 - rSTN9)/taurSTN9
    output[52] = 5 * HinfSTN9 * (1-sSTN9) - 1 * sSTN9
    output[53] = epsilonSTN * (-ICaSTN9 - ITSTN9 - kCaSTN * CaSTN9) # Calcium
   
    output[54] = (-ILGPE1 - IKGPE1 - INaGPE1 - ITGPE1 - ICaGPE1 - IAHPGPE1 - I_STN1_to_GPE1 - I_GPE2_to_GPE1 - I_GPE9_to_GPE1 - I_SNC1_to_GPE1)/C_m
    output[55] = phinGPE * (ninfGPE1 - nGPE1)/taunGPE1 # n
    output[56] = phihGPE * (hinfGPE1 - hGPE1)/tauhGPE1 # h
    output[57] = phirGPE * (rinfGPE1 - rGPE1)/taurGPE # r
    output[58] = 2 * HinfGPE1 * (1-sGPE1) - 0.08 * sGPE1
    output[59] = epsilonGPE * (-ICaGPE1 - ITGPE1 - kCaGPE * CaGPE1) 

    output[60] = (-ILGPE2 - IKGPE2 - INaGPE2 - ITGPE2 - ICaGPE2 - IAHPGPE2 - I_STN2_to_GPE2 - I_GPE1_to_GPE2 - I_GPE3_to_GPE2 - I_SNC2_to_GPE2)/C_m
    output[61] = phinGPE * (ninfGPE2 - nGPE2)/taunGPE2 # n
    output[62] = phihGPE * (hinfGPE2 - hGPE2)/tauhGPE2 # h
    output[63] = phirGPE * (rinfGPE2 - rGPE2)/taurGPE # r
    output[64] = 2 * HinfGPE2 * (1-sGPE2) - 0.08 * sGPE2
    output[65] = epsilonGPE * (-ICaGPE2 - ITGPE2 - kCaGPE * CaGPE2) 

    output[66] = (-ILGPE3 - IKGPE3 - INaGPE3 - ITGPE3 - ICaGPE3 - IAHPGPE3- I_STN3_to_GPE3 - I_GPE2_to_GPE3 - I_GPE4_to_GPE3 - I_SNC3_to_GPE3)/C_m
    output[67] = phinGPE * (ninfGPE3 - nGPE3)/taunGPE3 # n
    output[68] = phihGPE * (hinfGPE3 - hGPE3)/tauhGPE3 # h
    output[69] = phirGPE * (rinfGPE3 - rGPE3)/taurGPE # r
    output[70] = 2 * HinfGPE3 * (1-sGPE3) - 0.08 * sGPE3
    output[71] = epsilonGPE * (-ICaGPE3 - ITGPE3 - kCaGPE * CaGPE3) 

    output[72] = (-ILGPE4 - IKGPE4 - INaGPE4 - ITGPE4 - ICaGPE4 - IAHPGPE4 - I_STN4_to_GPE4 - I_GPE3_to_GPE4 - I_GPE5_to_GPE4 - I_SNC4_to_GPE4)/C_m
    output[73] = phinGPE * (ninfGPE4 - nGPE4)/taunGPE4 # n
    output[74] = phihGPE * (hinfGPE4 - hGPE4)/tauhGPE4 # h
    output[75] = phirGPE * (rinfGPE4 - rGPE4)/taurGPE # r
    output[76] = 2 * HinfGPE4 * (1-sGPE4) - 0.08 * sGPE4
    output[77] = epsilonGPE * (-ICaGPE4 - ITGPE4 - kCaGPE * CaGPE4) 

    output[78] = (-ILGPE5 - IKGPE5 - INaGPE5 - ITGPE5 - ICaGPE5 - IAHPGPE5 - I_STN5_to_GPE5 - I_GPE4_to_GPE5 - I_GPE6_to_GPE5 - I_SNC5_to_GPE5)/C_m
    output[79] = phinGPE * (ninfGPE5 - nGPE5)/taunGPE5 # n
    output[80] = phihGPE * (hinfGPE5 - hGPE5)/tauhGPE5 # h
    output[81] = phirGPE * (rinfGPE5 - rGPE5)/taurGPE # r
    output[82] = 2 * HinfGPE5 * (1-sGPE5) - 0.08 * sGPE5
    output[83] = epsilonGPE * (-ICaGPE5 - ITGPE5 - kCaGPE * CaGPE5) 

    output[84] = (-ILGPE6 - IKGPE6 - INaGPE6 - ITGPE6 - ICaGPE6 - IAHPGPE6 - I_STN6_to_GPE6 - I_GPE5_to_GPE6 - I_GPE7_to_GPE6 - I_SNC6_to_GPE6)/C_m
    output[85] = phinGPE * (ninfGPE6 - nGPE6)/taunGPE6 # n
    output[86] = phihGPE * (hinfGPE6 - hGPE6)/tauhGPE6 # h
    output[87] = phirGPE * (rinfGPE6 - rGPE6)/taurGPE # r
    output[88] = 2 * HinfGPE6 * (1-sGPE6) - 0.08 * sGPE6
    output[89] = epsilonGPE * (-ICaGPE6 - ITGPE6 - kCaGPE * CaGPE6) 

    output[90] = (-ILGPE7 - IKGPE7 - INaGPE7 - ITGPE7 - ICaGPE7 - IAHPGPE7 - I_STN7_to_GPE7 - I_GPE6_to_GPE7 - I_GPE8_to_GPE7 - I_SNC7_to_GPE7)/C_m
    output[91] = phinGPE * (ninfGPE7 - nGPE7)/taunGPE7 # n
    output[92] = phihGPE * (hinfGPE7 - hGPE7)/tauhGPE7 # h
    output[93] = phirGPE * (rinfGPE7- rGPE7)/taurGPE # r
    output[94] = 2 * HinfGPE7 * (1-sGPE7) - 0.08 * sGPE7
    output[95] = epsilonGPE * (-ICaGPE7 - ITGPE7 - kCaGPE * CaGPE7) 

    output[96] = (-ILGPE8 - IKGPE8 - INaGPE8 - ITGPE8 - ICaGPE8 - IAHPGPE8 - I_STN8_to_GPE8 - I_GPE7_to_GPE8 - I_GPE9_to_GPE8 - I_SNC8_to_GPE8)/C_m
    output[97] = phinGPE * (ninfGPE8 - nGPE8)/taunGPE8# n
    output[98] = phihGPE * (hinfGPE8 - hGPE8)/tauhGPE8 # h
    output[99] = phirGPE * (rinfGPE8 - rGPE8)/taurGPE # r
    output[100] = 2 * HinfGPE8 * (1-sGPE8) - 0.08 * sGPE8
    output[101] = epsilonGPE * (-ICaGPE8 - ITGPE8 - kCaGPE * CaGPE8) 

    output[102] = (-ILGPE9 - IKGPE9 - INaGPE9 - ITGPE9 - ICaGPE9 - IAHPGPE9 - I_STN9_to_GPE9  - I_GPE1_to_GPE9 - I_GPE8_to_GPE9 - I_SNC9_to_GPE9)/C_m
    output[103] = phinGPE * (ninfGPE9 - nGPE9)/taunGPE9 # n
    output[104] = phihGPE * (hinfGPE9 - hGPE9)/tauhGPE9 # h
    output[105] = phirGPE * (rinfGPE9 - rGPE9)/taurGPE # r
    output[106] = 2 * HinfGPE9 * (1-sGPE9) - 0.08 * sGPE9
    output[107] = epsilonGPE * (-ICaGPE9 - ITGPE9 - kCaGPE * CaGPE9) 

    output[108] = (-ILSNC1 - IKSNC1 - INaSNC1 - ICaSNC1 - IAHPSNC1)/C_m
    output[109] = phinSTN * (ninfSNC1 - nSNC1)/taunSNC1
    output[110] = phihSTN * (hinfSNC1 - hSNC1)/tauhSNC1
    output[111] = phirSTN * (rinfSNC1 - rSNC1)/taurSNC1
    output[112] = 5 * HinfSNC1 * (1-sSNC1) - 1 * sSNC1
    output[113] = epsilonSTN * (-ICaSNC1 - kCaSTN * CaSNC1) # Calcium

    output[114] = (-ILSNC2 - IKSNC2 - INaSNC2 - ICaSNC2 - IAHPSNC2 + I_App)/C_m
    output[115] = phinSTN * (ninfSNC2 - nSNC2)/taunSNC2
    output[116] = phihSTN * (hinfSNC2 - hSNC2)/tauhSNC2
    output[117] = phirSTN * (rinfSNC2 - rSNC2)/taurSNC2
    output[118] = 5 * HinfSNC2 * (1-sSNC2) - 1 * sSNC2
    output[119] = epsilonSTN * (-ICaSNC2 - kCaSTN * CaSNC2) # Calcium

    output[120] = (-ILSNC3 - IKSNC3 - INaSNC3 - ICaSNC3 - IAHPSNC3 + I_App)/C_m
    output[121] = phinSTN * (ninfSNC3 - nSNC3)/taunSNC3
    output[122] = phihSTN * (hinfSNC3 - hSNC3)/tauhSNC3
    output[123] = phirSTN * (rinfSNC3 - rSNC3)/taurSNC3
    output[124] = 5 * HinfSNC3 * (1-sSNC3) - 1 * sSNC3
    output[125] = epsilonSTN * (-ICaSNC3 - kCaSTN * CaSNC3) # Calcium

    output[126] = (-ILSNC4 - IKSNC4 - INaSNC4 - ICaSNC4 - IAHPSNC4 + I_App)/C_m
    output[127] = phinSTN * (ninfSNC4 - nSNC4)/taunSNC4
    output[128] = phihSTN * (hinfSNC4 - hSNC4)/tauhSNC4
    output[129] = phirSTN * (rinfSNC4 - rSNC4)/taurSNC4
    output[130] = 5 * HinfSNC4 * (1-sSNC4) - 1 * sSNC4
    output[131] = epsilonSTN * (-ICaSNC4 - kCaSTN * CaSNC4) # Calcium

    output[132] = (-ILSNC5 - IKSNC5 - INaSNC5 - ICaSNC5 - IAHPSNC5)/C_m
    output[133] = phinSTN * (ninfSNC5 - nSNC5)/taunSNC5
    output[134] = phihSTN * (hinfSNC5 - hSNC5)/tauhSNC5
    output[135] = phirSTN * (rinfSNC5 - rSNC5)/taurSNC5
    output[136] = 5 * HinfSNC5 * (1-sSNC5) - 1 * sSNC5
    output[137] = epsilonSTN * (-ICaSNC5 - kCaSTN * CaSNC5) # Calcium

    output[138] = (-ILSNC6 - IKSNC6 - INaSNC6 - ICaSNC6 - IAHPSNC6 + I_App)/C_m
    output[139] = phinSTN * (ninfSNC6 - nSNC6)/taunSNC6
    output[140] = phihSTN * (hinfSNC6 - hSNC6)/tauhSNC6
    output[141] = phirSTN * (rinfSNC6 - rSNC6)/taurSNC6
    output[142] = 5 * HinfSNC6 * (1-sSNC6) - 1 * sSNC6
    output[143] = epsilonSTN * (-ICaSNC6 - kCaSTN * CaSNC6) # Calcium

    output[144] = (-ILSNC7 - IKSNC7 - INaSNC7 - ICaSNC7 - IAHPSNC7 + I_App)/C_m
    output[145] = phinSTN * (ninfSNC7 - nSNC7)/taunSNC7
    output[146] = phihSTN * (hinfSNC7 - hSNC7)/tauhSNC7
    output[147] = phirSTN * (rinfSNC7 - rSNC7)/taurSNC7
    output[148] = 5 * HinfSNC7 * (1-sSNC7) - 1 * sSNC7
    output[149] = epsilonSTN * (-ICaSNC7 - kCaSTN * CaSNC7) # Calcium

    output[150] = (-ILSNC8 - IKSNC8 - INaSNC8 - ICaSNC8 - IAHPSNC8)/C_m
    output[151] = phinSTN * (ninfSNC8 - nSNC8)/taunSNC8
    output[152] = phihSTN * (hinfSNC8 - hSNC8)/tauhSNC8
    output[153] = phirSTN * (rinfSNC8 - rSNC8)/taurSNC8
    output[154] = 5 * HinfSNC8 * (1-sSNC8) - 1 * sSNC8
    output[155] = epsilonSTN * (-ICaSNC8 - kCaSTN * CaSNC8) # Calcium

    output[156] = (-ILSNC9 - IKSNC9 - INaSNC9 - ICaSNC9 - IAHPSNC9 + I_App)/C_m
    output[157] = phinSTN * (ninfSNC9 - nSNC9)/taunSNC9
    output[158] = phihSTN * (hinfSNC9 - hSNC9)/tauhSNC9
    output[159] = phirSTN * (rinfSNC9 - rSNC9)/taurSNC9
    output[160] = 5 * HinfSNC9 * (1-sSNC9) - 1 * sSNC9
    output[161] = epsilonSTN * (-ICaSNC9 - kCaSTN * CaSNC9) # Calcium

    return output

timeSpan = (0, 4000)

VSTN =-60
nSTN = 0.01
hSTN= 0.01
rSTN = 0.01
sSTN= 0.01
CaSTN = 0.1

VGPE = -60 # GPE
nGPE  = 0.01
hGPE= 0.01
rGPE = 0.01
sGPE = 0.01
CaGPE = 0.1

initial_condition = []
for i in range (1, 28):
    initial_condition.append(VSTN)
    initial_condition.append(nSTN)
    initial_condition.append(hSTN)
    initial_condition.append(rSTN)
    initial_condition.append(sSTN)
    initial_condition.append(CaSTN)
    
     # i don't know what these are
# VINT, nINT, hINT, rINT, sX, CaINT
outputVal = solve_ivp(INTtoX_DE2, timeSpan, initial_condition)

modelTraceSTN1 = outputVal.y[0, :]
modelTraceSTN2 = outputVal.y[6, :]
modelTraceSTN3 = outputVal.y[12, :]
modelTraceSTN4 = outputVal.y[18, :]
modelTraceSTN5 = outputVal.y[24, :]
modelTraceSTN6 = outputVal.y[30, :]
modelTraceSTN7 = outputVal.y[36, :]
modelTraceSTN8 = outputVal.y[42, :]
modelTraceSTN9 = outputVal.y[48, :]

modelTraceGPE1 = outputVal.y[54, :]
modelTraceGPE2 = outputVal.y[60, :]
modelTraceGPE3 = outputVal.y[66, :]
modelTraceGPE4 = outputVal.y[72, :]
modelTraceGPE5 = outputVal.y[78, :]
modelTraceGPE6 = outputVal.y[84, :]
modelTraceGPE7 = outputVal.y[90, :]
modelTraceGPE8 = outputVal.y[96, :]
modelTraceGPE9 = outputVal.y[102, :]

modelTraceSNC1 = outputVal.y[108, :]
modelTraceSNC2 = outputVal.y[114, :]
modelTraceSNC3 = outputVal.y[120, :]
modelTraceSNC4 = outputVal.y[126, :]
modelTraceSNC5 = outputVal.y[132, :]
modelTraceSNC6 = outputVal.y[138, :]
modelTraceSNC7 = outputVal.y[144, :]
modelTraceSNC8 = outputVal.y[150, :]
modelTraceSNC9 = outputVal.y[156, :]


plt.figure(1)
plt.suptitle('STN Neurons')

plt.subplot(9, 1, 1)
plt.plot(outputVal.t, modelTraceSTN1, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 2)
plt.plot(outputVal.t, modelTraceSTN2, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 3)
plt.plot(outputVal.t, modelTraceSTN3, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 4)
plt.plot(outputVal.t, modelTraceSTN4, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 5)
plt.plot(outputVal.t, modelTraceSTN5, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 6)
plt.plot(outputVal.t, modelTraceSTN6, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 7)
plt.plot(outputVal.t, modelTraceSTN7, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 8)
plt.plot(outputVal.t, modelTraceSTN8, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 9)
plt.plot(outputVal.t, modelTraceSTN9, color=(0.91, 0.41, 0.17), linewidth=1)

plt.figure(2)
plt.suptitle('GPe Neurons')

plt.subplot(9, 1, 1)
plt.plot(outputVal.t, modelTraceGPE1, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 2)
plt.plot(outputVal.t, modelTraceGPE2, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 3)
plt.plot(outputVal.t, modelTraceGPE3, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 4)
plt.plot(outputVal.t, modelTraceGPE4, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 5)
plt.plot(outputVal.t, modelTraceGPE5, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 6)
plt.plot(outputVal.t, modelTraceGPE6, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 7)
plt.plot(outputVal.t, modelTraceGPE7, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 8)
plt.plot(outputVal.t, modelTraceGPE8, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 9)
plt.plot(outputVal.t, modelTraceGPE9, color=(0.91, 0.41, 0.17), linewidth=1)

plt.figure(3)
plt.suptitle('SNC Neurons')

plt.subplot(9, 1, 1)
plt.plot(outputVal.t, modelTraceSNC1, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 2)
plt.plot(outputVal.t, modelTraceSNC2, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 3)
plt.plot(outputVal.t, modelTraceSNC3, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 4)
plt.plot(outputVal.t, modelTraceSNC4, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 5)
plt.plot(outputVal.t, modelTraceSNC5, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 6)
plt.plot(outputVal.t, modelTraceSNC6, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 7)
plt.plot(outputVal.t, modelTraceSNC7, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 8)
plt.plot(outputVal.t, modelTraceSNC8, color=(0.91, 0.41, 0.17), linewidth=1)
plt.subplot(9, 1, 9)
plt.plot(outputVal.t, modelTraceSNC9, color=(0.91, 0.41, 0.17), linewidth=1)

plt.show()
