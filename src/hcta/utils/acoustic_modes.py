import numpy as np
from scipy.optimize import root_scalar


def dispersion(omega, rho_up, rho_down, x_f, delta_L_up=0, delta_L_down=0):
    """
    Dispersion relationship
    Used to determine the acoustic angular frequencies in the piece-wise
    formulation of the Galerkin modes
    See Magri, 2014 for the formulation

    Args:
    omega: acoustic angular frequency
    rho_up, rho_down: mean flow density in the up- and downstream regions
    x_f: heat source location

    Returns:
    f: dispersion relationship,
       equal to 0 for the acoustic angular frequencies of the system
    """
    gamma = omega * np.sqrt(rho_up) * (x_f + delta_L_up)
    beta = omega * np.sqrt(rho_down) * (1 + delta_L_down - x_f)
    f = np.sin(beta) * np.cos(gamma) + np.cos(beta) * np.sin(gamma) * np.sqrt(
        rho_up / rho_down
    )
    return f


def dispersion_d(omega, rho_up, rho_down, x_f):
    """
    Differential of the dispersion relationship
    """
    gamma = omega * np.sqrt(rho_up * x_f)
    beta = omega * np.sqrt(rho_down) * (1 - x_f)
    df_domega1 = (
        np.cos(beta)
        * np.cos(gamma)
        * (
            np.sqrt(rho_down) * (1 - x_f)
            + np.sqrt(rho_up / rho_down) * np.sqrt(rho_up) * x_f
        )
    )
    df_domega2 = (
        -np.sin(beta)
        * np.sin(gamma)
        * (
            np.sqrt(rho_up) * x_f
            + np.sqrt(rho_up / rho_down) * np.sqrt(rho_down) * (1 - x_f)
        )
    )
    df_domega = df_domega1 + df_domega2
    return df_domega


def solve_dispersion(N_g, rho_up, rho_down, x_f, delta_L_up=0, delta_L_down=0):
    """
    Solves the dispersion relationship and returns acoustic frequencies

    Args:
    N_g: number of Galerkin modes
    omega: acoustic angular frequency
    rho_up, rho_down: mean flow density in the up- and downstream regions
    x_f: heat source location

    Returns:
    omega: acoustic angular frequencies
    k_up, k_down: acoustic wavenumbers in the up- and downstream regions
    """
    omega = np.zeros(N_g, dtype=np.float32)
    omega0 = np.pi  # initialise the first angular frequency
    for j in range(N_g):
        # solve the dispersion relationship as a root-finding problem
        # the brackets restrict the domain of the solution
        sol = root_scalar(
            dispersion,
            x0=omega0,
            bracket=[omega0 - 2, omega0 + 2],
            args=(rho_up, rho_down, x_f, delta_L_up, delta_L_down),
            # fprime=dispersion_d,
        )
        omega[j] = sol.root
        omega0 = omega[j] + np.pi  # initialise the next solution

    # pre-calculate wavenumbers etc that are used in the modes
    k_up = omega * np.sqrt(rho_up)
    k_down = omega * np.sqrt(rho_down)
    gamma = k_up * (x_f + delta_L_up)
    beta = k_down * (1 + delta_L_down - x_f)
    # @todo: checks for sin(beta) = 0
    upsilon = np.sin(gamma) / np.sin(beta)
    return omega, k_up, k_down, upsilon, gamma, beta


def acoustic_modes(N_g, rho_up, rho_down, x_f, x, delta_L_up=0, delta_L_down=0):
    """
    Construct the piece-wise Galerkin modes
    See Magri, 2014 for the formulation

    Args:
    N_g: number of Galerkin modes
    omega: acoustic angular frequency
    rho_up, rho_down: mean flow density in the up- and downstream regions
    x_f: heat source location
    x: spatial coordinates, 1 x N_x

    Returns:
    Psi: pressure modes, N_g x N_x
    Phi: velocity modes, N_g+1 x N_x
    """
    # Determine the acoustic frequencies
    _, k_up, k_down, upsilon, _, _ = solve_dispersion(
        N_g, rho_up, rho_down, x_f, delta_L_up, delta_L_down
    )

    # Find the indices for up- and downstream regions
    up_idx = np.where(x <= x_f)[0]
    down_idx = np.where(x > x_f)[0]

    # Pressure modes
    Psi_up = -np.sin(np.outer(k_up, x[up_idx] + delta_L_up))
    Psi_down = -upsilon[:, None] * np.sin(
        np.outer(k_down, (1 + delta_L_down - x[down_idx]))
    )
    Psi = np.hstack((Psi_up, Psi_down))

    # Velocity modes
    Phi_up = 1 / np.sqrt(rho_up) * np.cos(np.outer(k_up, x[up_idx] + delta_L_up))
    Phi_down = (
        -1
        / np.sqrt(rho_down)
        * upsilon[:, None]
        * np.cos(np.outer(k_down, (1 + delta_L_down - x[down_idx])))
    )

    # Add the zeroth mode, omega_0 = 0
    Phi_0_up = 1 / np.sqrt(rho_up) * np.ones(len(x[up_idx]))
    upsilon_0 = -np.sqrt(rho_down) / np.sqrt(rho_up)
    Phi_0_down = -1 / np.sqrt(rho_down) * upsilon_0 * np.ones(len(x[down_idx]))
    Phi_up = np.vstack((Phi_0_up, Phi_up))
    Phi_down = np.vstack((Phi_0_down, Phi_down))

    Phi = np.hstack((Phi_up, Phi_down))

    return Psi_up, Psi_down, Psi, Phi_up, Phi_down, Phi


def acoustic_modes_ddx(N_g, rho_up, rho_down, x_f, x):
    """
    Construct the spatial derivative of piece-wise Galerkin modes

    Args:
    N_g: number of Galerkin modes
    omega: acoustic angular frequency
    rho_up, rho_down: mean flow density in the up- and downstream regions
    x_f: heat source location
    x: spatial coordinates, 1 x N_x

    Returns:
    Psi_ddx: pressure modes, N_g x N_x
    Phi_ddx: velocity modes, N_g+1 x N_x
    """
    # Determine the acoustic frequencies
    _, k_up, k_down, upsilon, _, _ = solve_dispersion(N_g, rho_up, rho_down, x_f)

    # Find the indices for up- and downstream regions
    up_idx = np.where(x <= x_f)[0]
    down_idx = np.where(x > x_f)[0]

    # Pressure modes
    Psi_up_ddx = -k_up[:, None] * np.cos(np.outer(k_up, x[up_idx]))
    Psi_down_ddx = (
        upsilon[:, None] * k_down[:, None] * np.cos(np.outer(k_down, (1 - x[down_idx])))
    )
    Psi_ddx = np.hstack((Psi_up_ddx, Psi_down_ddx))

    # Velocity modes
    Phi_up_ddx = (
        -1 / np.sqrt(rho_up) * k_up[:, None] * np.sin(np.outer(k_up, x[up_idx]))
    )
    Phi_down_ddx = (
        -1
        / np.sqrt(rho_down)
        * upsilon[:, None]
        * k_down[:, None]
        * np.sin(np.outer(k_down, (1 - x[down_idx])))
    )

    # Add the zeroth mode, omega_0 = 0
    Phi_0_up_ddx = np.zeros(len(x[up_idx]))
    Phi_0_down_ddx = np.zeros(len(x[down_idx]))
    Phi_up_ddx = np.vstack((Phi_0_up_ddx, Phi_up_ddx))
    Phi_down_ddx = np.vstack((Phi_0_down_ddx, Phi_down_ddx))

    Phi_ddx = np.hstack((Phi_up_ddx, Phi_down_ddx))

    return Psi_up_ddx, Psi_down_ddx, Psi_ddx, Phi_up_ddx, Phi_down_ddx, Phi_ddx


def project(N_g, rho_up, rho_down, x_f, x, t, P, U):
    """
    Project data onto the Galerkin modes and solve for the Galerkin coefficients
    Then reconstruct the data

    Args:
    N_g: number of Galerkin modes
    omega: acoustic angular frequency
    rho_up, rho_down: mean flow density in the up- and downstream regions
    x_f: heat source location
    x: spatial coordinates, 1 x N_x
    t: time coordinates (not used), 1 x N_t
    P: pressure data, N_t x N_x
    U: velocity data, N_t x N_x

    Returns:
    mu: pressure mode coefficients, N_t x N_g
    eta: pressure mode coefficients, N_t x N_g+1
    P_recon: pressure reconstruction, N_t x N_x
    U_recon: velocity reconstruction, N_t x N_x
    """
    omega, k_up, k_down, upsilon, gamma, beta = solve_dispersion(
        N_g, rho_up, rho_down, x_f
    )
    # Find the indices for up- and downstream regions
    up_idx = np.where(x <= x_f)[0]
    down_idx = np.where(x >= x_f)[0]

    # Find the Nyquist frequency and fix N_g to that if bigger
    dx = x[1] - x[0]
    omega_nyquist = np.pi / dx
    print("Omega nyquist = ", omega_nyquist)
    print(
        "Upstream wavenumber with the given number of Galerkin modes = ", k_up[N_g - 1]
    )
    print(
        "Downstream wavenumber with the given number of Galerkin modes = ",
        k_down[N_g - 1],
    )
    N_g_max_up = np.where(k_up <= omega_nyquist)[0][-1] + 1
    N_g_max_down = np.where(k_down <= omega_nyquist)[0][-1] + 1
    N_g = np.minimum(N_g, np.minimum(N_g_max_up, N_g_max_down))
    print("Final N_g = ", N_g)
    print(
        "Upstream wavenumber with the final number of Galerkin modes = ", k_up[N_g - 1]
    )
    print(
        "Downstream wavenumber with the final number of Galerkin modes = ",
        k_down[N_g - 1],
    )

    # recompute with the new N_g
    omega, k_up, k_down, upsilon, gamma, beta = solve_dispersion(
        N_g, rho_up, rho_down, x_f
    )
    _, _, Psi, _, _, Phi = acoustic_modes(N_g, rho_up, rho_down, x_f, x)
    # in order for the numerical integration to work properly,
    # both up- and downstream modes should include x_f
    # here the definition of down_idx differs from acoustic_modes method (>= instead of >)
    Psi_up = Psi[:, up_idx]
    Psi_down = Psi[:, down_idx]
    Phi_up = Phi[:, up_idx]
    Phi_down = Phi[:, down_idx]

    P_up = P[:, up_idx]
    P_down = P[:, down_idx]
    U_up = U[:, up_idx]
    U_down = U[:, down_idx]

    int_A_total = np.zeros((len(t), N_g))
    for j in range(N_g):
        # multiply pressure with the upstream pressure Galerkin mode
        A_up = Psi_up[j, :] * P_up
        # integrate from 0 to x_f
        int_A_up = np.trapz(A_up, x[up_idx], axis=1)

        # multiply pressure with the downstream pressure Galerkin mode
        A_down = Psi_down[j, :] * P_down
        # integrate from x_f to 1
        int_A_down = np.trapz(A_down, x[down_idx], axis=1)

        # total integral
        int_A_total[:, j] = int_A_up + int_A_down

    int_B_total = np.zeros((len(t), N_g + 1))
    for j in range(N_g + 1):
        # multiply velocity with the upstream velocity Galerkin mode
        B_up = Phi_up[j, :] * U_up
        # integrate from 0 to x_f
        int_B_up = np.trapz(B_up, x[up_idx], axis=1)

        # multiply velocity with the downstream velocity Galerkin mode
        B_down = Phi_down[j, :] * U_down
        # integrate from x_f to 1
        int_B_down = np.trapz(B_down, x[down_idx], axis=1)

        # total integral
        int_B_total[:, j] = int_B_up + int_B_down

    # integral of the pressure galerkin mode multiplied with itself
    # upstream from 0 to x_f
    int_Psi_up = 1 / (4 * k_up) * (2 * k_up * x_f - np.sin(2 * k_up * x_f))
    # downstream from x_f to 1
    int_Psi_down = (
        upsilon**2
        * 1
        / (4 * k_down)
        * (2 * k_down * (1 - x_f) - np.sin(2 * k_down * (1 - x_f)))
    )
    # total (multiplies mu_j)
    int_Psi_total = int_Psi_up + int_Psi_down

    # integral of the velocity galerkin mode multiplied with itself
    # upstream from 0 to x_f
    int_Phi_up_diag = (
        (1 / rho_up) * 1 / (4 * k_up) * (2 * k_up * x_f + np.sin(2 * k_up * x_f))
    )
    # downstream from x_f to 1
    int_Phi_down_diag = (
        (1 / rho_down)
        * upsilon**2
        * 1
        / (4 * k_down)
        * (2 * k_down * (1 - x_f) + np.sin(2 * k_down * (1 - x_f)))
    )
    # total
    int_Phi_total_diag = int_Phi_up_diag + int_Phi_down_diag

    int_Phi_total = np.zeros((N_g, N_g))
    for j in range(N_g):
        for k in range(N_g):
            if j == k:
                int_Phi_total[j, k] = int_Phi_total_diag[j]
            else:
                # since velocity modes are not orthogonal, need to compute the integrals in the off-diagonals
                D = (omega[k] - omega[j]) * (
                    omega[k] + omega[j]
                )  # denominator with omega terms
                D_rho = (
                    rho_up * np.sqrt(rho_up) * rho_down * np.sqrt(rho_down)
                )  # denominator with rho terms

                # compute the total analytical integral in two parts
                int_Phi_off_diag_1 = (
                    (omega[k] / D)
                    * (np.sin(gamma[k]) / np.sin(beta[j]))
                    * (1 / D_rho)
                    * (
                        rho_down
                        * np.sqrt(rho_down)
                        * np.sin(beta[j])
                        * np.cos(gamma[j])
                        + rho_up * np.sqrt(rho_up) * np.sin(gamma[j]) * np.cos(beta[j])
                    )
                )

                int_Phi_off_diag_2 = (
                    -(omega[j] / D)
                    * (np.sin(gamma[j]) / np.sin(beta[k]))
                    * (1 / D_rho)
                    * (
                        rho_down
                        * np.sqrt(rho_down)
                        * np.sin(beta[k])
                        * np.cos(gamma[k])
                        + rho_up * np.sqrt(rho_up) * np.sin(gamma[k]) * np.cos(beta[k])
                    )
                )

                int_Phi_total_off_diag = int_Phi_off_diag_1 + int_Phi_off_diag_2
                int_Phi_total[j, k] = int_Phi_total_off_diag

    # add the zeroth mode
    int_Phi_diag_0 = np.array([1 / np.sqrt(rho_up)])
    int_Phi_off_diag_0_up = (
        (1 / rho_up) * (1 / (omega * np.sqrt(rho_up))) * np.sin(gamma)
    )
    int_Phi_off_diag_0_down = (
        -(1 / (np.sqrt(rho_up) * np.sqrt(rho_down)))
        * upsilon
        * (1 / (omega * np.sqrt(rho_down)))
        * np.sin(beta)
    )
    int_Phi_total_0_row = int_Phi_off_diag_0_up + int_Phi_off_diag_0_down
    int_Phi_total = np.vstack((int_Phi_total_0_row, int_Phi_total))
    int_Phi_total_0_col = np.vstack(
        (int_Phi_diag_0[:, None], int_Phi_total_0_row[:, None])
    )
    int_Phi_total = np.hstack((int_Phi_total_0_col, int_Phi_total))

    # Solve for the pressure Galerkin coefficients
    mu = int_A_total / int_Psi_total

    # Solve for the velocity Galerkin coefficients
    # as a linear system of equations
    eta = np.linalg.solve(int_Phi_total, int_B_total.T).T

    # Reconstruct pressure
    P_recon = np.matmul(mu, Psi)

    # Reconstruct velocity
    U_recon = np.matmul(eta, Phi)
    return mu, eta, P_recon, U_recon
