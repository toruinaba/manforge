! =============================================================================
! manforge -- YUKinematic3D UMAT (Yoshida-Uemori 2-surface + stagnation surface)
!
! Fortran port of YUKinematic3D (src/manforge/models/yu_kinematic.py).
! Algorithm: fully-implicit Newton-Raphson on [stress(6), dlambda(1),
!            theta(6), beta(6)] = 19 unknowns.
!
! NOTE on naming:
!   - Fortran is case-insensitive.  The Python model has two parameters:
!       B    (bound surface size, self.B)
!       b    (backstress parameter, self.b)
!     These would collide in Fortran argument lists.  To resolve this,
!     "B" (bound surface size) is named "B_bnd" throughout this file.
!   - The output array argument "J" in Python is named "jmat" (6x6), "jvec"
!     (6-vector), or "jval" (scalar) to avoid collision with the Fortran loop
!     variable j.
!
! Material properties (PROPS, 12 parameters, matches model.param_names):
!   PROPS(1)  = E        Young's modulus
!   PROPS(2)  = nu       Poisson's ratio
!   PROPS(3)  = Y        initial yield stress
!   PROPS(4)  = B_bnd    bound surface size  (Python: self.B)
!   PROPS(5)  = C_1      kinematic hardening rate (theta, pre-reversal)
!   PROPS(6)  = C_2      kinematic hardening rate (theta, post-reversal)
!   PROPS(7)  = Rsat     saturation value of R
!   PROPS(8)  = k        boundary surface hardening rate
!   PROPS(9)  = b        backstress (beta) hardening rate  (Python: self.b)
!   PROPS(10) = h        stagnation surface parameter
!   PROPS(11) = Ea       asymptotic modulus (nonlinear elasticity)
!   PROPS(12) = xi_param nonlinear elasticity decay parameter (Python: self.xi)
!
! State variables (STATEV, 22 slots, model.state_names without "stress"):
!   STATEV(1..6)   = theta(1..6)     relative backstress of yield surface
!   STATEV(7..12)  = beta(1..6)      relative backstress of boundary surface
!   STATEV(13)     = R               radius increment of boundary surface
!   STATEV(14..19) = q(1..6)         center of stagnation surface
!   STATEV(20)     = r               radius of stagnation surface
!   STATEV(21)     = eps_eq          equivalent plastic strain
!   STATEV(22)     = theta_max       max ||theta|| in history
!
! Voigt convention: [s11, s22, s33, s12, s13, s23]
!   Stress components: physical shear (sigma_12 = tensor shear)
!   Strain components: engineering shear (gamma_12 = 2 * tensor shear)
!
! NR unknown vector layout (19 components):
!   x = [stress(6), dlambda(1), theta(6), beta(6)]
! Residual vector layout (matching x):
!   r = [R_stress(6), R_yield(1), R_theta(6), R_beta(6)]
!
! Build (from fortran/ directory):
!   uv run python -m numpy.f2py -c abaqus_stubs.f90 yu_kinematic_3d.f90 -m yu_kinematic_3d
! =============================================================================


! =============================================================================
! yu_kinematic_3d_elastic_stiffness
!
! Returns the secant elastic stiffness tensor C_e = f(eps_eq) * C_iso
! where f = 1 - (1 - Ea/E) * (1 - exp(-xi_param * eps_eq)).
!
! Parameters
! ----------
! E         [in]  : Young's modulus
! nu        [in]  : Poisson's ratio
! eps_eq    [in]  : equivalent plastic strain (from state)
! Ea        [in]  : asymptotic modulus
! xi_param  [in]  : decay parameter (self.xi in Python)
! C         [out] : 6x6 Voigt stiffness (Fortran column-major order)
! =============================================================================
subroutine yu_kinematic_3d_elastic_stiffness(E, nu, eps_eq, Ea, xi_param, C)
    implicit none
    double precision, intent(in)  :: E, nu, eps_eq, Ea, xi_param
    double precision, intent(out) :: C(6,6)

    double precision :: lam, mu, factor
    integer :: ii, jj

    mu     = E / (2.0d0 * (1.0d0 + nu))
    lam    = E * nu / ((1.0d0 + nu) * (1.0d0 - 2.0d0 * nu))
    factor = 1.0d0 - (1.0d0 - Ea / E) * (1.0d0 - exp(-xi_param * eps_eq))

    do jj = 1, 6
        do ii = 1, 6
            C(ii,jj) = 0.0d0
        end do
    end do

    do ii = 1, 3
        do jj = 1, 3
            C(ii,jj) = lam
        end do
        C(ii,ii) = lam + 2.0d0 * mu
    end do
    do ii = 4, 6
        C(ii,ii) = mu
    end do

    do jj = 1, 6
        do ii = 1, 6
            C(ii,jj) = factor * C(ii,jj)
        end do
    end do

end subroutine yu_kinematic_3d_elastic_stiffness


! =============================================================================
! yu_vonmises_norm  (internal helper, not f2py-callable)
!
! Von Mises equivalent norm for a physical-shear deviatoric vector:
!   ||xi||_vm = sqrt(1.5 * (xi(1)^2 + xi(2)^2 + xi(3)^2
!                          + 2*(xi(4)^2 + xi(5)^2 + xi(6)^2)))
! =============================================================================
subroutine yu_vonmises_norm(xi, xi_norm)
    implicit none
    double precision, intent(in)  :: xi(6)
    double precision, intent(out) :: xi_norm

    double precision :: sss
    integer :: ii

    sss = 0.0d0
    do ii = 1, 3
        sss = sss + xi(ii)**2
    end do
    do ii = 4, 6
        sss = sss + 2.0d0 * xi(ii)**2
    end do
    xi_norm = sqrt(1.5d0 * sss)

end subroutine yu_vonmises_norm


! =============================================================================
! yu_deviatoric  (internal helper, not f2py-callable)
!
! Deviatoric part of a stress vector:
!   dev(s)_i = s_i - (s_11+s_22+s_33)/3  for i=1..3
!   dev(s)_i = s_i                         for i=4..6
! =============================================================================
subroutine yu_deviatoric(s, s_dev)
    implicit none
    double precision, intent(in)  :: s(6)
    double precision, intent(out) :: s_dev(6)

    double precision :: p_mean
    integer :: ii

    p_mean = (s(1) + s(2) + s(3)) / 3.0d0
    do ii = 1, 6
        s_dev(ii) = s(ii)
    end do
    do ii = 1, 3
        s_dev(ii) = s_dev(ii) - p_mean
    end do

end subroutine yu_deviatoric


! =============================================================================
! yu_calc_norm_n_flow
!
! Computes the von Mises norm of xi and the plastic flow direction.
!   xi_norm = ||xi||_vm
!   flow_i  = 1.5 * T_i * xi_i / xi_norm
!           = 1.5 * xi_i / xi_norm   for i=1..3 (normal)
!           = 3.0 * xi_i / xi_norm   for i=4..6 (shear, T_i=2)
!
! Parameters
! ----------
! xi      [in]  : 6-component deviatoric driving stress
! xi_norm [out] : von Mises norm of xi
! flow    [out] : plastic flow direction (6,)
! =============================================================================
subroutine yu_calc_norm_n_flow(xi, xi_norm, flow)
    implicit none
    double precision, intent(in)  :: xi(6)
    double precision, intent(out) :: xi_norm, flow(6)

    integer :: ii

    call yu_vonmises_norm(xi, xi_norm)

    do ii = 1, 3
        flow(ii) = 1.5d0 * xi(ii) / xi_norm
    end do
    do ii = 4, 6
        flow(ii) = 3.0d0 * xi(ii) / xi_norm
    end do

end subroutine yu_calc_norm_n_flow


! =============================================================================
! yu_prepare_rstress
!
! Computes the partial derivative of the plastic flow direction n with respect
! to stress: dn/dsig = 1.5/xi_norm * (T @ I_dev - outer(n_hat, n_hat))
! where n_hat = flow / sqrt(1.5).
!
! Parameters
! ----------
! xi       [in]  : 6-component deviatoric driving stress
! dn_dsig  [out] : (6,6) partial derivative matrix
! =============================================================================
subroutine yu_prepare_rstress(xi, dn_dsig)
    implicit none
    double precision, intent(in)  :: xi(6)
    double precision, intent(out) :: dn_dsig(6,6)

    double precision :: xi_norm, flow(6)
    double precision :: n_hat(6), T_I_dev(6,6), P_ij
    double precision, parameter :: SQRT15 = 1.2247448713915890d0  ! sqrt(1.5)
    integer :: ii, jj

    call yu_calc_norm_n_flow(xi, xi_norm, flow)

    ! n_hat = flow / sqrt(1.5)
    do ii = 1, 6
        n_hat(ii) = flow(ii) / SQRT15
    end do

    ! T_I_dev = T @ I_dev
    ! T = diag(1,1,1,2,2,2)
    ! I_dev_ij = delta_ij - 1/3 * [i<=3 and j<=3]
    do ii = 1, 6
        do jj = 1, 6
            P_ij = 0.0d0
            if (ii == jj) P_ij = 1.0d0
            if (ii <= 3 .and. jj <= 3) P_ij = P_ij - 1.0d0/3.0d0
            if (ii > 3) then
                T_I_dev(ii,jj) = 2.0d0 * P_ij
            else
                T_I_dev(ii,jj) = P_ij
            end if
        end do
    end do

    do ii = 1, 6
        do jj = 1, 6
            dn_dsig(ii,jj) = 1.5d0 / xi_norm * (T_I_dev(ii,jj) - n_hat(ii) * n_hat(jj))
        end do
    end do

end subroutine yu_prepare_rstress


! =============================================================================
! yu_prepare_rtheta
!
! Computes auxiliary scalars used by the theta-residual Jacobian blocks.
! Contains two hard branches identical to the Python reference:
!   (1) C_k is C_1 if B_bnd-Y > theta_max, else C_2
!   (2) a_prime = 0 when R == R_n (floating-point equality)
!
! Parameters (input)
! ------------------
! B_bnd, Y, k, Rsat, C_1, C_2  : material parameters
! theta(6)                      : current theta iterate
! theta_max                     : max ||theta|| at step start
! R                             : current R iterate
! R_n                           : R at step start
! dlambda                       : current delta-lambda iterate
!
! Parameters (output)
! -------------------
! theta_bar   : von Mises norm of theta
! theta_flow  : flow direction for theta (6,)
! C_k         : selected kinematic hardening coefficient
! s           : 1 / (1 + k * dlambda)
! a           : B_bnd + R - Y
! a_prime     : dR/d(dlambda) if R != R_n, else 0
! =============================================================================
subroutine yu_prepare_rtheta(B_bnd, Y, k, Rsat, C_1, C_2, &
                              theta, theta_max, R, R_n, dlambda, &
                              theta_bar, theta_flow, C_k, s, a, a_prime)
    implicit none
    double precision, intent(in)  :: B_bnd, Y, k, Rsat, C_1, C_2
    double precision, intent(in)  :: theta(6), theta_max, R, R_n, dlambda
    double precision, intent(out) :: theta_bar, theta_flow(6), C_k, s, a, a_prime

    integer :: ii

    call yu_vonmises_norm(theta, theta_bar)

    do ii = 1, 3
        theta_flow(ii) = 1.5d0 * theta(ii) / theta_bar
    end do
    do ii = 4, 6
        theta_flow(ii) = 3.0d0 * theta(ii) / theta_bar
    end do

    ! Hard step (Jacobian side): matches Python _prepare_Rtheta line 224
    if (B_bnd - Y > theta_max) then
        C_k = C_1
    else
        C_k = C_2
    end if

    s = 1.0d0 / (1.0d0 + k * dlambda)
    a = B_bnd + R - Y

    ! Floating-point equality: matches Python _prepare_Rtheta line 227
    if (R /= R_n) then
        a_prime = -k * s * s * (R_n + k * Rsat * dlambda) + s * k * Rsat
    else
        a_prime = 0.0d0
    end if

end subroutine yu_prepare_rtheta


! =============================================================================
! yu_dRs_dstress
!
! dR_stress / d_stress = I + dlambda * C @ dn_dsig
!
! Parameters
! ----------
! C(6,6)  [in]  : elastic stiffness
! xi(6)   [in]  : deviatoric driving stress
! dlambda [in]  : consistency parameter
! jmat    [out] : result (6,6)
! =============================================================================
subroutine yu_dRs_dstress(C, xi, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: C(6,6), xi(6), dlambda
    double precision, intent(out) :: jmat(6,6)

    double precision :: dn_dsig(6,6)
    integer :: ii, jj, kk

    call yu_prepare_rstress(xi, dn_dsig)

    do ii = 1, 6
        do jj = 1, 6
            jmat(ii,jj) = 0.0d0
            if (ii == jj) jmat(ii,jj) = 1.0d0
            do kk = 1, 6
                jmat(ii,jj) = jmat(ii,jj) + dlambda * C(ii,kk) * dn_dsig(kk,jj)
            end do
        end do
    end do

end subroutine yu_dRs_dstress


! =============================================================================
! yu_dRs_dbeta
!
! dR_stress / d_beta = -dlambda * C @ dn_dsig
! =============================================================================
subroutine yu_dRs_dbeta(C, xi, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: C(6,6), xi(6), dlambda
    double precision, intent(out) :: jmat(6,6)

    double precision :: dn_dsig(6,6)
    integer :: ii, jj, kk

    call yu_prepare_rstress(xi, dn_dsig)

    do ii = 1, 6
        do jj = 1, 6
            jmat(ii,jj) = 0.0d0
            do kk = 1, 6
                jmat(ii,jj) = jmat(ii,jj) - dlambda * C(ii,kk) * dn_dsig(kk,jj)
            end do
        end do
    end do

end subroutine yu_dRs_dbeta


! =============================================================================
! yu_dRs_dtheta
!
! dR_stress / d_theta = -dlambda * C @ dn_dsig  (identical to dRs_dbeta)
! =============================================================================
subroutine yu_dRs_dtheta(C, xi, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: C(6,6), xi(6), dlambda
    double precision, intent(out) :: jmat(6,6)

    call yu_dRs_dbeta(C, xi, dlambda, jmat)

end subroutine yu_dRs_dtheta


! =============================================================================
! yu_dRs_dlambda
!
! dR_stress / d_lambda = C @ flow
!     - xi_param * (1 - Ea/E) * exp(-xi_param*eps_eq) / factor * dlambda * (C @ flow)
! where factor = Ea/E + (1 - Ea/E) * exp(-xi_param * eps_eq).
!
! Parameters
! ----------
! E, Ea, xi_param  : material parameters
! C(6,6)           : elastic stiffness
! xi(6)            : deviatoric driving stress
! eps_eq           : equivalent plastic strain
! dlambda          : consistency parameter
! jvec(6)          : result vector
! =============================================================================
subroutine yu_dRs_dlambda(E, Ea, xi_param, C, xi, eps_eq, dlambda, jvec)
    implicit none
    double precision, intent(in)  :: E, Ea, xi_param
    double precision, intent(in)  :: C(6,6), xi(6), eps_eq, dlambda
    double precision, intent(out) :: jvec(6)

    double precision :: xi_norm, flow(6), Cn(6), factor, exp_term
    integer :: ii, kk

    call yu_calc_norm_n_flow(xi, xi_norm, flow)

    do ii = 1, 6
        Cn(ii) = 0.0d0
        do kk = 1, 6
            Cn(ii) = Cn(ii) + C(ii,kk) * flow(kk)
        end do
    end do

    exp_term = exp(-xi_param * eps_eq)
    factor = Ea / E + (1.0d0 - Ea / E) * exp_term

    do ii = 1, 6
        jvec(ii) = Cn(ii) - xi_param * (1.0d0 - Ea / E) * exp_term / factor * dlambda * Cn(ii)
    end do

end subroutine yu_dRs_dlambda


! =============================================================================
! yu_dRb_dstress
!
! dR_beta / d_stress = -k*b_kin*dlambda/Y * I_dev
!
! Parameters
! ----------
! k, b_kin, Y  : material parameters (b_kin is Python self.b)
! dlambda      : consistency parameter
! jmat(6,6)    : result
! =============================================================================
subroutine yu_dRb_dstress(k, b_kin, Y, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: k, b_kin, Y, dlambda
    double precision, intent(out) :: jmat(6,6)

    double precision :: coeff, P_ij
    integer :: ii, jj

    coeff = -k * b_kin * dlambda / Y

    do ii = 1, 6
        do jj = 1, 6
            P_ij = 0.0d0
            if (ii == jj) P_ij = 1.0d0
            if (ii <= 3 .and. jj <= 3) P_ij = P_ij - 1.0d0/3.0d0
            jmat(ii,jj) = coeff * P_ij
        end do
    end do

end subroutine yu_dRb_dstress


! =============================================================================
! yu_dRb_dbeta
!
! dR_beta / d_beta = (1 + k*b_kin*dlambda/Y + k*dlambda) * I
! =============================================================================
subroutine yu_dRb_dbeta(k, b_kin, Y, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: k, b_kin, Y, dlambda
    double precision, intent(out) :: jmat(6,6)

    double precision :: coeff
    integer :: ii, jj

    coeff = 1.0d0 + k * b_kin / Y * dlambda + k * dlambda

    do ii = 1, 6
        do jj = 1, 6
            jmat(ii,jj) = 0.0d0
        end do
        jmat(ii,ii) = coeff
    end do

end subroutine yu_dRb_dbeta


! =============================================================================
! yu_dRb_dtheta
!
! dR_beta / d_theta = k*b_kin*dlambda/Y * I
! =============================================================================
subroutine yu_dRb_dtheta(k, b_kin, Y, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: k, b_kin, Y, dlambda
    double precision, intent(out) :: jmat(6,6)

    double precision :: coeff
    integer :: ii, jj

    coeff = k * b_kin / Y * dlambda

    do ii = 1, 6
        do jj = 1, 6
            jmat(ii,jj) = 0.0d0
        end do
        jmat(ii,ii) = coeff
    end do

end subroutine yu_dRb_dtheta


! =============================================================================
! yu_dRb_dlambda
!
! dR_beta / d_lambda = -k*b_kin/Y * xi + k * beta
! =============================================================================
subroutine yu_dRb_dlambda(k, b_kin, Y, xi, beta, dlambda, jvec)
    implicit none
    double precision, intent(in)  :: k, b_kin, Y
    double precision, intent(in)  :: xi(6), beta(6), dlambda
    double precision, intent(out) :: jvec(6)

    integer :: ii

    do ii = 1, 6
        jvec(ii) = -k * b_kin / Y * xi(ii) + k * beta(ii)
    end do

end subroutine yu_dRb_dlambda


! =============================================================================
! yu_dRt_dstress
!
! dR_theta / d_stress = -a*C_k*dlambda/Y * I_dev
! =============================================================================
subroutine yu_dRt_dstress(B_bnd, Y, k, Rsat, C_1, C_2, &
                           theta, theta_max, R, R_n, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: B_bnd, Y, k, Rsat, C_1, C_2
    double precision, intent(in)  :: theta(6), theta_max, R, R_n, dlambda
    double precision, intent(out) :: jmat(6,6)

    double precision :: theta_bar, theta_flow(6), C_k, s, a, a_prime
    double precision :: coeff, P_ij
    integer :: ii, jj

    call yu_prepare_rtheta(B_bnd, Y, k, Rsat, C_1, C_2, &
                           theta, theta_max, R, R_n, dlambda, &
                           theta_bar, theta_flow, C_k, s, a, a_prime)

    coeff = -a * C_k * dlambda / Y

    do ii = 1, 6
        do jj = 1, 6
            P_ij = 0.0d0
            if (ii == jj) P_ij = 1.0d0
            if (ii <= 3 .and. jj <= 3) P_ij = P_ij - 1.0d0/3.0d0
            jmat(ii,jj) = coeff * P_ij
        end do
    end do

end subroutine yu_dRt_dstress


! =============================================================================
! yu_dRt_dbeta
!
! dR_theta / d_beta = a*C_k*dlambda/Y * I
! =============================================================================
subroutine yu_dRt_dbeta(B_bnd, Y, k, Rsat, C_1, C_2, &
                         theta, theta_max, R, R_n, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: B_bnd, Y, k, Rsat, C_1, C_2
    double precision, intent(in)  :: theta(6), theta_max, R, R_n, dlambda
    double precision, intent(out) :: jmat(6,6)

    double precision :: theta_bar, theta_flow(6), C_k, s, a, a_prime
    double precision :: coeff
    integer :: ii, jj

    call yu_prepare_rtheta(B_bnd, Y, k, Rsat, C_1, C_2, &
                           theta, theta_max, R, R_n, dlambda, &
                           theta_bar, theta_flow, C_k, s, a, a_prime)

    coeff = a * C_k * dlambda / Y

    do ii = 1, 6
        do jj = 1, 6
            jmat(ii,jj) = 0.0d0
        end do
        jmat(ii,ii) = coeff
    end do

end subroutine yu_dRt_dbeta


! =============================================================================
! yu_dRt_dtheta
!
! dR_theta / d_theta.
! Has a theta_bar < 1e-14 guard to avoid division by zero (Python line 275).
! =============================================================================
subroutine yu_dRt_dtheta(B_bnd, Y, k, Rsat, C_1, C_2, &
                          theta, theta_max, R, R_n, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: B_bnd, Y, k, Rsat, C_1, C_2
    double precision, intent(in)  :: theta(6), theta_max, R, R_n, dlambda
    double precision, intent(out) :: jmat(6,6)

    double precision :: theta_bar, theta_flow(6), C_k, s, a, a_prime
    double precision :: diag_coeff, outer_coeff
    double precision, parameter :: SQRT15 = 1.2247448713915890d0
    integer :: ii, jj

    call yu_prepare_rtheta(B_bnd, Y, k, Rsat, C_1, C_2, &
                           theta, theta_max, R, R_n, dlambda, &
                           theta_bar, theta_flow, C_k, s, a, a_prime)

    do ii = 1, 6
        do jj = 1, 6
            jmat(ii,jj) = 0.0d0
        end do
    end do

    if (theta_bar < 1.0d-14) then
        ! Degenerate case: only diagonal term (Python line 276)
        diag_coeff = 1.0d0 + a * C_k * dlambda / Y
        do ii = 1, 6
            jmat(ii,ii) = diag_coeff
        end do
    else
        diag_coeff  = 1.0d0 + a * C_k * dlambda / Y &
                      + C_k * dlambda * sqrt(a / theta_bar)
        outer_coeff = SQRT15 * C_k * dlambda * sqrt(a / theta_bar) / (2.0d0 * theta_bar)
        do ii = 1, 6
            do jj = 1, 6
                ! outer product: (theta_flow/sqrt(1.5))_i * theta_j
                jmat(ii,jj) = -outer_coeff * (theta_flow(ii) / SQRT15) * theta(jj)
            end do
            jmat(ii,ii) = jmat(ii,ii) + diag_coeff
        end do
    end if

end subroutine yu_dRt_dtheta


! =============================================================================
! yu_dRt_dlambda
!
! dR_theta / d_lambda.
! Has a theta_bar < 1e-14 guard (Python line 283).
! =============================================================================
subroutine yu_dRt_dlambda(B_bnd, Y, k, Rsat, C_1, C_2, &
                           xi, theta, theta_max, R, R_n, dlambda, jvec)
    implicit none
    double precision, intent(in)  :: B_bnd, Y, k, Rsat, C_1, C_2
    double precision, intent(in)  :: xi(6), theta(6), theta_max, R, R_n, dlambda
    double precision, intent(out) :: jvec(6)

    double precision :: theta_bar, theta_flow(6), C_k, s, a, a_prime
    integer :: ii

    call yu_prepare_rtheta(B_bnd, Y, k, Rsat, C_1, C_2, &
                           theta, theta_max, R, R_n, dlambda, &
                           theta_bar, theta_flow, C_k, s, a, a_prime)

    if (theta_bar < 1.0d-14) then
        ! Degenerate case (Python line 284)
        do ii = 1, 6
            jvec(ii) = (-a * C_k / Y - C_k * dlambda / Y * a_prime) * xi(ii)
        end do
    else
        do ii = 1, 6
            jvec(ii) = - a * C_k / Y * xi(ii) &
                       - C_k * dlambda / Y * a_prime * xi(ii) &
                       + C_k * sqrt(a / theta_bar) * theta(ii) &
                       + C_k * dlambda * sqrt(1.0d0 / (theta_bar * a)) * a_prime / 2.0d0 * theta(ii)
        end do
    end if

end subroutine yu_dRt_dlambda


! =============================================================================
! yu_dRl_dstress
!
! dR_yield / d_stress = flow
! =============================================================================
subroutine yu_dRl_dstress(xi, jvec)
    implicit none
    double precision, intent(in)  :: xi(6)
    double precision, intent(out) :: jvec(6)

    double precision :: xi_norm, flow(6)

    call yu_calc_norm_n_flow(xi, xi_norm, flow)

    jvec = flow

end subroutine yu_dRl_dstress


! =============================================================================
! yu_dRl_dbeta
!
! dR_yield / d_beta = -flow
! =============================================================================
subroutine yu_dRl_dbeta(xi, jvec)
    implicit none
    double precision, intent(in)  :: xi(6)
    double precision, intent(out) :: jvec(6)

    double precision :: xi_norm, flow(6)
    integer :: ii

    call yu_calc_norm_n_flow(xi, xi_norm, flow)

    do ii = 1, 6
        jvec(ii) = -flow(ii)
    end do

end subroutine yu_dRl_dbeta


! =============================================================================
! yu_dRl_dtheta
!
! dR_yield / d_theta = -flow
! =============================================================================
subroutine yu_dRl_dtheta(xi, jvec)
    implicit none
    double precision, intent(in)  :: xi(6)
    double precision, intent(out) :: jvec(6)

    call yu_dRl_dbeta(xi, jvec)

end subroutine yu_dRl_dtheta


! =============================================================================
! yu_dRl_dlambda
!
! dR_yield / d_lambda = 0
! =============================================================================
subroutine yu_dRl_dlambda(jval)
    implicit none
    double precision, intent(out) :: jval

    jval = 0.0d0

end subroutine yu_dRl_dlambda


! =============================================================================
! yu_smooth_heaviside (internal helper for residual)
!
! Smooth Heaviside: 0.5 * (1 + tanh(beta * x / 2))
! with beta=50.0, matching Python smooth_heaviside default.
! =============================================================================
subroutine yu_smooth_heaviside(x, hv)
    implicit none
    double precision, intent(in)  :: x
    double precision, intent(out) :: hv

    double precision, parameter :: BETA_H = 50.0d0

    hv = 0.5d0 * (1.0d0 + tanh(0.5d0 * BETA_H * x))

end subroutine yu_smooth_heaviside


! =============================================================================
! yu_smooth_sqrt (internal helper for residual)
!
! Smooth square root: sqrt(x + eps^2) with eps = 1e-30 (Python default).
! =============================================================================
subroutine yu_smooth_sqrt(x, result)
    implicit none
    double precision, intent(in)  :: x
    double precision, intent(out) :: result

    double precision, parameter :: EPS_SQRT = 1.0d-30

    result = sqrt(x + EPS_SQRT**2)

end subroutine yu_smooth_sqrt


! =============================================================================
! yu_calc_residual
!
! Computes the full 19-element residual vector for the Newton-Raphson system.
! r = [R_stress(6), R_yield(1), R_theta(6), R_beta(6)]
!
! Note on C_k: uses smooth_heaviside (residual side), unlike the Jacobian
! which uses a hard step via _prepare_Rtheta. Matches Python calc_residual.
!
! Parameters
! ----------
! E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
!             : 12 material parameters (order = model.param_names)
! stress_new(6), theta_new(6), beta_new(6)  : NR iterates (Implicit states)
! R_new       : current R NR iterate
! eps_eq_new  : equivalent plastic strain at current iterate
! theta_n(6), beta_n(6)  : states at step start
! theta_max_n : theta_max at step start
! stress_trial(6) : elastic predictor stress
! dlambda     : current delta-lambda iterate
! r_vec(19)   : output residual vector
! =============================================================================
subroutine yu_calc_residual(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                             stress_new, theta_new, beta_new, R_new, eps_eq_new, &
                             theta_n, beta_n, theta_max_n, &
                             stress_trial, dlambda, &
                             r_vec)
    implicit none
    double precision, intent(in)  :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in)  :: stress_new(6), theta_new(6), beta_new(6)
    double precision, intent(in)  :: R_new, eps_eq_new
    double precision, intent(in)  :: theta_n(6), beta_n(6), theta_max_n
    double precision, intent(in)  :: stress_trial(6), dlambda
    double precision, intent(out) :: r_vec(19)

    double precision :: C(6,6), dev_stress(6), xi(6)
    double precision :: xi_norm, flow(6), theta_norm
    double precision :: C_k, hv, a, sm_sqrt_result
    double precision :: R_stress(6), R_theta(6), R_beta(6), R_yield
    integer :: ii, kk

    ! Elastic stiffness
    call yu_kinematic_3d_elastic_stiffness(E, nu, eps_eq_new, Ea, xi_param, C)

    ! C_k via smooth_heaviside (residual side)
    call yu_smooth_heaviside(theta_max_n - (B_bnd - Y), hv)
    C_k = C_1 - (C_1 - C_2) * hv

    a = B_bnd + R_new - Y

    ! xi = dev(stress_new) - theta_new - beta_new
    call yu_deviatoric(stress_new, dev_stress)
    do ii = 1, 6
        xi(ii) = dev_stress(ii) - theta_new(ii) - beta_new(ii)
    end do

    call yu_calc_norm_n_flow(xi, xi_norm, flow)
    call yu_vonmises_norm(theta_new, theta_norm)

    ! R_stress = stress_new - stress_trial + dlambda * C @ flow
    do ii = 1, 6
        R_stress(ii) = stress_new(ii) - stress_trial(ii)
        do kk = 1, 6
            R_stress(ii) = R_stress(ii) + dlambda * C(ii,kk) * flow(kk)
        end do
    end do

    ! R_theta = theta_new - theta_n
    !         - (C_k*a/Y*xi - C_k*smooth_sqrt(a/theta_norm)*theta_new)*dlambda
    call yu_smooth_sqrt(a / theta_norm, sm_sqrt_result)
    do ii = 1, 6
        R_theta(ii) = theta_new(ii) - theta_n(ii) &
                    - (C_k * a / Y * xi(ii) - C_k * sm_sqrt_result * theta_new(ii)) * dlambda
    end do

    ! R_beta = beta_new - beta_n - (k*b_kin/Y*xi - k*beta_new)*dlambda
    do ii = 1, 6
        R_beta(ii) = beta_new(ii) - beta_n(ii) &
                   - (k * b_kin / Y * xi(ii) - k * beta_new(ii)) * dlambda
    end do

    ! R_yield = vonmises_norm(xi) - Y
    R_yield = xi_norm - Y

    ! r_vec = [R_stress(6), R_yield(1), R_theta(6), R_beta(6)]
    do ii = 1, 6
        r_vec(ii)    = R_stress(ii)
        r_vec(7+ii)  = R_theta(ii)
        r_vec(13+ii) = R_beta(ii)
    end do
    r_vec(7) = R_yield

end subroutine yu_calc_residual


! =============================================================================
! yu_calc_jacobian
!
! Computes the full 19x19 Jacobian matrix of the NR system.
! Column order: stress(1..6), dlambda(7), theta(8..13), beta(14..19)
! Row order:    R_stress(1..6), R_yield(7), R_theta(8..13), R_beta(14..19)
!
! Parameters
! ----------
! E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
!             : 12 material parameters
! stress_new(6), theta_new(6), beta_new(6)  : NR iterates
! R_new       : current R iterate
! eps_eq_new  : equivalent plastic strain at current iterate
! theta_max_new : theta_max (= state_n["theta_max"] used for Jacobian)
! R_n         : R at step start
! dlambda     : current delta-lambda iterate
! jac(19,19)  : output Jacobian
! =============================================================================
subroutine yu_calc_jacobian(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                             stress_new, theta_new, beta_new, R_new, eps_eq_new, &
                             theta_max_new, R_n, dlambda, &
                             jac)
    implicit none
    double precision, intent(in)  :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in)  :: stress_new(6), theta_new(6), beta_new(6)
    double precision, intent(in)  :: R_new, eps_eq_new, theta_max_new, R_n, dlambda
    double precision, intent(out) :: jac(19,19)

    double precision :: C(6,6), dev_stress(6), xi(6)
    double precision :: Rs_s(6,6), Rs_b(6,6), Rs_t(6,6), Rs_l(6)
    double precision :: Rb_s(6,6), Rb_b(6,6), Rb_t(6,6), Rb_l(6)
    double precision :: Rt_s(6,6), Rt_b(6,6), Rt_t(6,6), Rt_l(6)
    double precision :: Rl_s(6), Rl_b(6), Rl_t(6), Rl_l
    integer :: ii, jj

    call yu_kinematic_3d_elastic_stiffness(E, nu, eps_eq_new, Ea, xi_param, C)

    call yu_deviatoric(stress_new, dev_stress)
    do ii = 1, 6
        xi(ii) = dev_stress(ii) - theta_new(ii) - beta_new(ii)
    end do

    ! Rstress blocks
    call yu_dRs_dstress(C, xi, dlambda, Rs_s)
    call yu_dRs_dbeta(C, xi, dlambda, Rs_b)
    call yu_dRs_dtheta(C, xi, dlambda, Rs_t)
    call yu_dRs_dlambda(E, Ea, xi_param, C, xi, eps_eq_new, dlambda, Rs_l)

    ! Rbeta blocks
    call yu_dRb_dstress(k, b_kin, Y, dlambda, Rb_s)
    call yu_dRb_dbeta(k, b_kin, Y, dlambda, Rb_b)
    call yu_dRb_dtheta(k, b_kin, Y, dlambda, Rb_t)
    call yu_dRb_dlambda(k, b_kin, Y, xi, beta_new, dlambda, Rb_l)

    ! Rtheta blocks
    call yu_dRt_dstress(B_bnd, Y, k, Rsat, C_1, C_2, &
                        theta_new, theta_max_new, R_new, R_n, dlambda, Rt_s)
    call yu_dRt_dbeta(B_bnd, Y, k, Rsat, C_1, C_2, &
                      theta_new, theta_max_new, R_new, R_n, dlambda, Rt_b)
    call yu_dRt_dtheta(B_bnd, Y, k, Rsat, C_1, C_2, &
                       theta_new, theta_max_new, R_new, R_n, dlambda, Rt_t)
    call yu_dRt_dlambda(B_bnd, Y, k, Rsat, C_1, C_2, &
                        xi, theta_new, theta_max_new, R_new, R_n, dlambda, Rt_l)

    ! Ryield blocks
    call yu_dRl_dstress(xi, Rl_s)
    call yu_dRl_dbeta(xi, Rl_b)
    call yu_dRl_dtheta(xi, Rl_t)
    call yu_dRl_dlambda(Rl_l)

    ! Assemble 19x19
    do jj = 1, 19
        do ii = 1, 19
            jac(ii,jj) = 0.0d0
        end do
    end do

    ! Rows 1..6: R_stress; cols: stress(1..6), dlambda(7), theta(8..13), beta(14..19)
    do ii = 1, 6
        do jj = 1, 6
            jac(ii, jj)    = Rs_s(ii,jj)
            jac(ii, 7+jj)  = Rs_t(ii,jj)
            jac(ii, 13+jj) = Rs_b(ii,jj)
        end do
        jac(ii, 7) = Rs_l(ii)
    end do

    ! Row 7: R_yield
    do jj = 1, 6
        jac(7, jj)    = Rl_s(jj)
        jac(7, 7+jj)  = Rl_t(jj)
        jac(7, 13+jj) = Rl_b(jj)
    end do
    jac(7,7) = Rl_l

    ! Rows 8..13: R_theta
    do ii = 1, 6
        do jj = 1, 6
            jac(7+ii, jj)    = Rt_s(ii,jj)
            jac(7+ii, 7+jj)  = Rt_t(ii,jj)
            jac(7+ii, 13+jj) = Rt_b(ii,jj)
        end do
        jac(7+ii, 7) = Rt_l(ii)
    end do

    ! Rows 14..19: R_beta
    do ii = 1, 6
        do jj = 1, 6
            jac(13+ii, jj)    = Rb_s(ii,jj)
            jac(13+ii, 7+jj)  = Rb_t(ii,jj)
            jac(13+ii, 13+jj) = Rb_b(ii,jj)
        end do
        jac(13+ii, 7) = Rb_l(ii)
    end do

end subroutine yu_calc_jacobian
