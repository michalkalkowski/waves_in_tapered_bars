"""
Piecewise uniform model for exponentially tapered circular rods
Author: Michal K. Kalkowski
M.Kalkowski@soton.ac.uk

Copyright (c) 2016 by Michal K. Kalkowski
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

Anyone who uses/copies/reuses/modifies/merges/publishes this code
is asked to give attribution to the author by citing the following publication:

Michał K. Kalkowski, Jen M. Muggleton, Emiliano Rustighi, An experimental approach for the determination of axial and flexural wavenumbers in circular exponentially tapered bars, Journal of Sound and Vibration, Volume 390, 3 March 2017, Pages 67-85, ISSN 0022-460X, http://dx.doi.org/10.1016/j.jsv.2016.10.018.
(http://www.sciencedirect.com/science/article/pii/S0022460X1630551X)
"""
from __future__ import print_function
import numpy as np

def piecewise_uniform_beam(f, E, nu, rho, R0, L, beta, n_el, x_Ls, G=0,
                           kappa='a', nat=False, nat_only=False):
    """
    Calculates the response of an exponentially tapered circular beam
    according to the piecewise uniform model
    abd the Timoshenko beam theory.

    Parameters:
    -----------
    f : frequency vector (1D array)
    E : Young's modulus
    nu : Poisson's ratio
    rho : density
    R0 : starting radius of the tapered beam
    L : length of the beam
    beta : flare constant
    n_el : number of elements in the piecewise formulation
    x_Ls : locations at which the response is requested (1D array)
    G : shear modulus (if not related to E via Poisson's ratio)
    kappa: Timoshenko shear correction factor
            (if different from the standard one)
    nat : boolean, specifies whether the natural frequencies
            should also be computed
    nat_only : boolean, if True, only natural frequencies are computed (no FRF)

    Returns:
    --------
    resp : receptances at the desired locations
    nat_frq : natural frequencies, if requested

    or
    nat_frq, if nat_only is True
    """

    def matmul_tup(matrices):
        """
        Helper function for numpy.matmul. Allows for performing a matrix
        multiuplication on a series
        of matrices. As in matmul, the multiplication is performed on
        the last two dimensions.

        Parameters:
        -----------
        matrices : list of matrices to multiply
        """
        for i in range(len(matrices) - 1):
            if i == 0:
                product = np.matmul(matrices[0], matrices[1])
            else:
                product = np.matmul(product, matrices[i + 1])
        return product

    if G == 0:
        G = E/(2*(1 + nu))
    if kappa == 'a':
        kappa = 6*(1 + nu)/(7 + 6*nu)

    # configure piecewise elements
    x0_discrete = np.linspace(0, L, n_el, False)
    element_length = x0_discrete[1] - x0_discrete[0]
    mid_x_discrete = np.linspace(element_length/2, L-element_length/2,
                                 n_el, True)
    r_discrete = R0*np.exp(-beta*mid_x_discrete)

    A = np.pi*r_discrete**2
    I = A*r_discrete**2/4

    Cs = (G*kappa/rho)**0.5
    Cb = (E*I/(rho*A.astype('complex')))**0.5
    Cr = (I/A.astype('complex'))**0.5

    omega = 2*np.pi*f.reshape(-1, 1)

    # calculating wavenumbers
    k1 = np.sqrt(omega**2/2*(1/Cs**2 + Cr**2/Cb**2) +
                 np.sqrt(omega**4/4*(1/Cs**2 - Cr**2/Cb**2)**2 +
                         omega**2/Cb**2))
    k2 = np.sqrt(omega**2/2*(1/Cs**2 + Cr**2/Cb**2) -
                 np.sqrt(omega**4/4*(1/Cs**2 - Cr**2/Cb**2)**2 +
                         omega**2/Cb**2))
    inds = np.where(abs(np.exp(-1j*k2*0.1)) > 1)
    k2[inds[0], inds[1]] *= -1

    # calculate propagation matrices
    tau = np.zeros([len(f), n_el, 4, 4], 'complex')
    tau[:, :, 0, 0] = np.exp(-1j*k1*element_length)
    tau[:, :, 1, 1] = np.exp(-1j*k2*element_length)
    tau[:, :, 2, 2] = np.exp(1j*k1*element_length)
    tau[:, :, 3, 3] = np.exp(1j*k2*element_length)

    # calculate P coefficients (see paper)
    P1 = k1*(1 - omega**2/(k1**2*Cs**2))
    P2 = k2*(1 - omega**2/(k2**2*Cs**2))

    # define wave mode shapes
    phi_q_pos = np.ones([len(f), n_el, 2, 2], 'complex')
    phi_q_neg = np.ones([len(f), n_el, 2, 2], 'complex')
    phi_f_pos = np.zeros([len(f), n_el, 2, 2], 'complex')
    phi_f_neg = np.zeros([len(f), n_el, 2, 2], 'complex')
    phi_q_pos[:, :, 1, 0] = -1j*P1
    phi_q_pos[:, :, 1, 1] = -1j*P2
    phi_q_neg[:, :, 1, 0] = 1j*P1
    phi_q_neg[:, :, 1, 1] = 1j*P2
    phi_f_pos[:, :, 0, 0] = G*A*kappa*1j*(P1 - k1)
    phi_f_pos[:, :, 0, 1] = G*A*kappa*1j*(P2 - k2)
    phi_f_pos[:, :, 1, 0] = -k1*P1*E*I
    phi_f_pos[:, :, 1, 1] = -k2*P2*E*I
    phi_f_neg[:, :, 0, 0] = -G*A*kappa*1j*(P1 - k1)
    phi_f_neg[:, :, 0, 1] = -G*A*kappa*1j*(P2 - k2)
    phi_f_neg[:, :, 1, 0] = -k1*P1*E*I
    phi_f_neg[:, :, 1, 1] = -k2*P2*E*I

    # calculate reflection matrices
    R_L = np.zeros([len(omega), 2, 2], 'complex')
    R_R = np.zeros([len(omega), 2, 2], 'complex')
    scalar_L = -1/(P2[:, 0]*k2[:, 0]*(P1[:, 0] - k1[:, 0]) + \
                P1[:, 0]*k1[:, 0]*(k2[:, 0] - P2[:, 0]))
    R_L[:, 0, 0] = scalar_L*(-(P1[:, 0]*k1[:, 0]*(P2[:, 0] - k2[:, 0]) + \
                        P2[:, 0]*k2[:, 0]*(P1[:, 0] - k1[:, 0])))
    R_L[:, 0, 1] = scalar_L*(-2*k2[:, 0]*P2[:, 0]*(P2[:, 0] - k2[:, 0]))
    R_L[:, 1, 0] = scalar_L*(2*P1[:, 0]*k1[:, 0]*(P1[:, 0] - k1[:, 0]))
    R_L[:, 1, 1] = scalar_L*(P1[:, 0]*k1[:, 0]*(P2[:, 0] - k2[:, 0]) + \
                        P2[:, 0]*k2[:, 0]*(P1[:, 0] - k1[:, 0]))
    scalar_R = -1/(P2[:, -1]*k2[:, -1]*(P1[:, -1] - k1[:, -1]) + \
                P1[:, -1]*k1[:, -1]*(k2[:, -1] - P2[:, -1]))
    R_R[:, 0, 0] = scalar_R*(-(P1[:, -1]*k1[:, -1]*(P2[:, -1] - k2[:, -1]) + \
                        P2[:, -1]*k2[:, -1]*(P1[:, -1] - k1[:, -1])))
    R_R[:, 0, 1] = scalar_R*(-2*k2[:, -1]*P2[:, -1]*(P2[:, -1] - k2[:, -1]))
    R_R[:, 1, 0] = scalar_R*(2*P1[:, -1]*k1[:, -1]*(P1[:, -1] - k1[:, -1]))
    R_R[:, 1, 1] = scalar_R*(P1[:, -1]*k1[:, -1]*(P2[:, -1] - k2[:, -1]) + \
                        P2[:, -1]*k2[:, -1]*(P1[:, -1] - k1[:, -1]))

    # calculate scattering matrices at the junctions as T = inv(Ta).dot(Tb)
    phi_q_pos_1 = phi_q_pos[:, :-1, :, :]
    phi_q_neg_1 = phi_q_neg[:, :-1, :, :]
    phi_f_pos_1 = phi_f_pos[:, :-1, :, :]
    phi_f_neg_1 = phi_f_neg[:, :-1, :, :]

    phi_q_pos_2 = phi_q_pos[:, 1:, :, :]
    phi_q_neg_2 = phi_q_neg[:, 1:, :, :]
    phi_f_pos_2 = phi_f_pos[:, 1:, :, :]
    phi_f_neg_2 = phi_f_neg[:, 1:, :, :]

    Ta = np.zeros([len(f), n_el-1, 4, 4], 'complex')
    Ta[:, :, :2, :2] = phi_q_pos_1
    Ta[:, :, :2, 2:] = -phi_q_neg_2
    Ta[:, :, 2:, :2] = phi_f_pos_1
    Ta[:, :, 2:, 2:] = -phi_f_neg_2

    Tb = np.zeros([len(f), n_el-1, 4, 4], 'complex')
    Tb[:, :, :2, :2] = -phi_q_neg_1
    Tb[:, :, :2, 2:] = phi_q_pos_2
    Tb[:, :, 2:, :2] = -phi_f_neg_1
    Tb[:, :, 2:, 2:] = phi_f_pos_2

    Tainv = np.linalg.inv(Ta)
    T = np.matmul(Tainv, Tb)

    # calculate excited wave amplitudes (Q=1, M=0)
    scalar = 1/(k2*P2*(P1 - k1) + k1*P1*(k2 - P2))[:, 0]
    e_pos_1 = scalar*-k2[:, 0]*P2[:, 0]*(-1j*1/G/A[0]/kappa)
    e_pos_2 = scalar*k1[:, 0]*P1[:, 0]*(-1j*1/G/A[0]/kappa)
    e_pos = np.c_[e_pos_1, e_pos_2]

    # set up the piecewise unform model (see Appendix B of the paper)
    Tau = tau[:, :, :2, :2]
    ident1 = np.dstack(n_el*[np.eye(2)]).T
    ident2 = np.dstack(len(f)*[np.eye(2)]).T
    ident = np.tile(ident1, (len(f), 1, 1, 1))

    Tau_prev = np.concatenate((np.zeros([len(f), 1, 2, 2], 'complex'),
                               tau[:, :-1, :2, :2]), axis=1)
    Tau_next = np.concatenate((tau[:, 1:, :2, :2],
                               np.zeros([len(f), 1, 2, 2], 'complex')), axis=1)

    RLs = np.zeros([len(f), n_el, 2, 2], 'complex')
    RRs = np.zeros([len(f), n_el, 2, 2], 'complex')
    TLs = np.zeros([len(f), n_el, 2, 2], 'complex')
    TRs = np.zeros([len(f), n_el, 2, 2], 'complex')

    RLs[:, 0] = R_L
    RLs[:, 1:] = T[:, :, 2:, 2:]
    RRs[:, -1] = R_R
    RRs[:, :-1] = T[:, :, :2, :2]
    TLs[:, 1:] = T[:, :, 2:, :2]
    TRs[:, :-1] = T[:, :, :2, 2:]

    A_pos = np.linalg.inv(ident - matmul_tup((RLs, Tau, RRs, Tau)))
    C_pos = matmul_tup((A_pos, TLs, Tau_prev))
    D_pos = matmul_tup((A_pos, RLs, Tau, TRs, Tau_next))

    B_neg = np.linalg.inv(ident - matmul_tup((RRs, Tau, RLs, Tau)))
    A_neg = matmul_tup((B_neg, RRs, Tau))
    C_neg = matmul_tup((B_neg, RRs, Tau, TLs, Tau_prev))
    D_neg = matmul_tup((B_neg, TRs, Tau_next))

    H_neg = np.zeros([len(f), n_el, 2, 2], 'complex')
    H_neg[:, -1] = matmul_tup((RRs[:, -1], Tau[:, -1], C_pos[:, -1]))

    H_pos = np.zeros([len(f), n_el, 2, 2], 'complex')
    H_pos[:, -1] = C_pos[:, -1]

    # perform the piecewise uniform reduction
    for i in range(n_el-2, 0, -1):
        H_pos[:, i] = np.matmul(np.linalg.inv(ident2 - np.matmul(D_pos[:, i],
                                                H_neg[:, i + 1])), C_pos[:, i])
        H_neg[:, i] = C_neg[:, i] + matmul_tup((D_neg[:, i], H_neg[:, i + 1],
                                                                H_pos[:, i]))
    # if only natural frequencies are requested, calculate and return them
    if nat_only:
        nat_frq = np.linalg.det(np.matmul(D_pos[:, 0], H_neg[:, 1]) - ident2)
        return nat_frq
    # otherwise, start calculating travelling waves in the elements (element-by-element)
    else:
        a_pos = np.zeros([len(f), n_el, 2, 1], 'complex')
        a_neg = np.zeros([len(f), n_el, 2, 1], 'complex')

        a_pos[:, 0] = matmul_tup((np.linalg.inv(ident2 - np.matmul(D_pos[:, 0],
                     H_neg[:, 1])), A_pos[:, 0], e_pos.reshape(len(f), 2, 1)))
        a_neg[:, 0] = np.matmul(A_neg[:, 0], e_pos.reshape(len(f), 2, 1)) + \
                            matmul_tup((D_neg[:, 0], H_neg[:, 1], a_pos[:, 0]))
        for i in range(1, n_el):
            if i == n_el - 1:
                a_pos[:, i] = np.matmul(H_pos[:, i], a_pos[:, i - 1])
                a_neg[:, i] = matmul_tup((R_R, tau[:, -1, :2, :2], a_pos[:, i]))
            else:
                a_pos[:, i] = np.matmul(H_pos[:, i], a_pos[:, i - 1])
                a_neg[:, i] = np.matmul(H_neg[:, i], a_pos[:, i - 1])
            section_inds = []
        section_loc = []
        # calculate the receptance at desired locations in (x_Ls)
        for x_L in x_Ls:
            # identify in which piecewise uniform are locations in x_Ls
            where_is = np.argmin(abs(x0_discrete - x_L))
            if x_L < x0_discrete[where_is] or \
               x0_discrete[where_is] == element_length:
                section_inds.append(where_is - 1)
                section_loc.append(x_L - x0_discrete[where_is - 1])
            else:
                section_inds.append(where_is)
                section_loc.append(x_L - x0_discrete[where_is])
        # calculate the response according to relevant propagation matrices
        resp = np.zeros([len(f), len(section_inds)], 'complex')
        for i, j in enumerate(section_inds):
            tau_pos = np.zeros([len(f), 2, 2], 'complex')
            tau_neg = np.zeros([len(f), 2, 2], 'complex')
            tau_pos[:, 0, 0] = np.exp(-1j*k1[:, j]*section_loc[i])
            tau_pos[:, 1, 1] = np.exp(-1j*k2[:, j]*section_loc[i])
            tau_neg[:, 0, 0] = np.exp(-1j*k1[:, j]*(element_length -
                                                    section_loc[i]))
            tau_neg[:, 1, 1] = np.exp(-1j*k2[:, j]*(element_length -
                                                    section_loc[i]))

            resp[:, i] = (matmul_tup((phi_q_pos[:, j, :, :],
                                      tau_pos, a_pos[:, j])) + \
                        matmul_tup((phi_q_neg[:, j, :, :],
                                    tau_neg, a_neg[:, j])))[:, 0].squeeze()
	# if natural frequencies are also of interest
        if nat:
            nat_frq = np.linalg.det(np.matmul(D_pos[:, 0],
                                              H_neg[:, 1]) - ident2)
            return resp, nat_frq
        else:
            return resp
