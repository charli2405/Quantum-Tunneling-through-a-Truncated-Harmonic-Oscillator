import numpy as np

# Physical & numerical constants
en0 = 1.0*10**-5
de = 0.001
ec = 1.60218*10**-19       
em = 9.10938*10**-31       
plc = 6.62607*10**-34      
pi = np.pi
nd = 250
hcrs = 0.5 * plc / pi         
omg = 1.52*10**14
hcog = hcrs * omg / ec
fck = em * omg * omg / ec
vz = 2.5

# SHO + barrier geometry
shtw = 2.0 * vz / fck
htw = np.sqrt(shtw)
tw = 2.0 * htw
tb = 5.6*10**-9            
dl = 2.0 * tb + tw
dz1 = htw / nd
c1 = 2.0 * em / hcrs * ec / hcrs
cpim = 1j
hdl = tb + htw
z0 = -hdl
v0 = 0.0

# Grid allocation
nd2 = nd + 2
nd3 = nd + 3
tnd2 = 2 * nd + 2
tnd3 = tnd2 + 1
tnd4 = tnd2 + 2

# index size safety
size = tnd4 + 5
z = np.zeros(size)
v = np.zeros(size)

# Left side
z[1] = -(tb + htw)
v[1] = 0.0
z[2] = -htw
v[2] = vz
u1 = 0.5 * fck * dz1 * dz1
z[nd2] = 0.0
v[nd2] = 0.5 * u1
v[nd3] = v[nd2]

# Harmonic potential (FEM core)
for i in range(1, nd):
    iz = i + 1
    ij = i + nd3
    ik = nd2 - i
    ij1 = nd2 + i

    zia = iz * dz1
    uint = 0.5 * fck * zia * zia

    v[ij] = 0.5 * (u1 + uint)
    u1 = uint
    v[ik] = v[ij]

    z[ij1] = i * dz1
    z[ik] = -z[ij1]

# Right side
z[tnd2] = htw
v[tnd3] = vz
z[tnd3] = htw + tb
v[tnd4] = 0.0

# Energy loop
transmission = []
energy = []

for ie in range(1, 2001):
    enz = en0 + de * ie
    mf = np.identity(2, dtype=complex)

    for j1 in range(1, tnd4):
        jp1 = j1 + 1

        kl1s = c1 * (enz - v[j1])
        kr1s = c1 * (enz - v[jp1])
        kl1 = np.sqrt(kl1s + 0j)
        kr1 = np.sqrt(kr1s + 0j)
        lam1 = kl1 / kr1

        eim1 = cpim * (kl1 - kr1) * z[j1]
        eip1 = cpim * (kl1 + kr1) * z[j1]
        emm1 = np.exp(eim1)
        epp1 = np.exp(eip1)

        b11 = 0.5 * (1 + lam1) * emm1
        b12 = 0.5 * (1 - lam1) / epp1
        b21 = 0.5 * (1 - lam1) * epp1
        b22 = 0.5 * (1 + lam1) / emm1

        msj1 = np.array([[b11, b12],
                         [b21, b22]], dtype=complex)

        mj1 = msj1 / np.linalg.det(msj1)
        mf = mj1 @ mf

    metc = 1.0 / mf[1, 1]
    tc1 = np.abs(metc)**2
    rca = mf[1, 0] / mf[1, 1]
    rc1 = np.abs(rca)**2
    ncon = rc1 + tc1
    tcn = tc1 / ncon

    energy.append(enz)
    transmission.append(tc1)

    print(tc1)

# Convert to arrays for plotting
energy = np.array(energy)
transmission = np.array(transmission)
# Plotting
import matplotlib.pyplot as plt
plt.plot(energy, transmission)
plt.yscale("log")
plt.xlim(0.0, 2.2)
plt.ylim(1*10**-120, 1*10**-30)
plt.xlabel("Energy")
plt.ylabel("Transmission")
plt.show()