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
    double precision, parameter :: EPS_SQRT = 1.0d-30
    integer :: ii

    sss = 0.0d0
    do ii = 1, 3
        sss = sss + xi(ii)**2
    end do
    do ii = 4, 6
        sss = sss + 2.0d0 * xi(ii)**2
    end do
    xi_norm = sqrt(1.5d0 * sss + EPS_SQRT**2)

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

    double precision :: hv
    integer :: ii

    call yu_vonmises_norm(theta, theta_bar)

    do ii = 1, 3
        theta_flow(ii) = 1.5d0 * theta(ii) / theta_bar
    end do
    do ii = 4, 6
        theta_flow(ii) = 3.0d0 * theta(ii) / theta_bar
    end do

    ! smooth_heaviside (consistent with residual): matches Python _prepare_Rtheta
    call yu_smooth_heaviside(theta_max - (B_bnd - Y), hv)
    C_k = C_1 - (C_1 - C_2) * hv

    s = 1.0d0 / (1.0d0 + k * dlambda)
    a = B_bnd + R - Y

    ! Floating-point equality: matches Python _prepare_Rtheta line 251
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
! yu_dflow_dxi
!
! d(flow)/d(xi)  (without I_dev projection; use for dRs/dtheta and dRs/dbeta).
!
! flow = 1.5*T@xi / |xi|_VM,  d(flow)/d(xi) = 1.5/|xi| * (T - outer(n_hat, n_hat))
! where n_hat = flow/sqrt(1.5)  (unit-norm direction).
!
! Unlike yu_prepare_rstress (which computes d(flow)/d(xi) @ I_dev for dRs/dstress),
! this returns the un-projected derivative, needed because
!   d(xi)/d(theta) = d(xi)/d(beta) = -I  (no deviatoric projection).
!
! Parameters
! ----------
! xi(6)    [in]  : deviatoric driving stress
! dflow(6,6) [out] : result matrix
! =============================================================================
subroutine yu_dflow_dxi(xi, dflow)
    implicit none
    double precision, intent(in)  :: xi(6)
    double precision, intent(out) :: dflow(6,6)

    double precision :: xi_norm, flow(6), n_hat(6)
    double precision, parameter :: SQRT15 = 1.2247448713915890d0
    integer :: ii, jj

    call yu_calc_norm_n_flow(xi, xi_norm, flow)

    do ii = 1, 6
        n_hat(ii) = flow(ii) / SQRT15
    end do

    ! off-diagonal: -1.5/|xi| * n_hat(i) * n_hat(j)
    ! diagonal adds T(i,i)/|xi| * 1.5: 1.5/|xi| for i<=3, 3.0/|xi| for i>3
    do ii = 1, 6
        do jj = 1, 6
            dflow(ii,jj) = 1.5d0 / xi_norm * (-n_hat(ii) * n_hat(jj))
        end do
        if (ii <= 3) then
            dflow(ii,ii) = dflow(ii,ii) + 1.5d0 / xi_norm
        else
            dflow(ii,ii) = dflow(ii,ii) + 3.0d0 / xi_norm
        end if
    end do

end subroutine yu_dflow_dxi


! =============================================================================
! yu_dRs_dbeta
!
! dR_stress / d_beta = -dlambda * C @ dflow_dxi
!
! d(xi)/d(beta) = -I (no deviatoric projection), so use yu_dflow_dxi without I_dev.
! =============================================================================
subroutine yu_dRs_dbeta(C, xi, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: C(6,6), xi(6), dlambda
    double precision, intent(out) :: jmat(6,6)

    double precision :: dflow(6,6)
    integer :: ii, jj, kk

    call yu_dflow_dxi(xi, dflow)

    do ii = 1, 6
        do jj = 1, 6
            jmat(ii,jj) = 0.0d0
            do kk = 1, 6
                jmat(ii,jj) = jmat(ii,jj) - dlambda * C(ii,kk) * dflow(kk,jj)
            end do
        end do
    end do

end subroutine yu_dRs_dbeta


! =============================================================================
! yu_dRs_dtheta
!
! dR_stress / d_theta = -dlambda * C @ dflow_dxi  (identical to dRs_dbeta)
!
! d(xi)/d(theta) = -I (no deviatoric projection), so use yu_dflow_dxi without I_dev.
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


! =============================================================================
! solve19 -- in-place LU solver for a 19x19 system with NRHS right-hand sides
!
! Solves A*X = B using Gaussian elimination with partial pivoting.
! B is overwritten with the solution X.
! Returns info=0 on success, info=k if pivot k is essentially zero.
!
! Parameters
! ----------
! A(19,19)    [inout] : coefficient matrix (overwritten with LU)
! B(19,NRHS) [inout] : right-hand sides (overwritten with solution)
! NRHS        [in]   : number of right-hand sides
! info        [out]  : 0 = success, k = singular at pivot k
! =============================================================================
subroutine solve19(A, B, NRHS, info)
    implicit none
    integer,          intent(in)    :: NRHS
    double precision, intent(inout) :: A(19,19), B(19,NRHS)
    integer,          intent(out)   :: info

    integer          :: i, j, k, piv
    double precision :: maxval_loc, tmp_d, factor
    double precision, parameter :: EPS = 1.0d-14

    info = 0

    do k = 1, 19
        ! -- find pivot row
        maxval_loc = abs(A(k,k))
        piv = k
        do i = k+1, 19
            if (abs(A(i,k)) > maxval_loc) then
                maxval_loc = abs(A(i,k))
                piv = i
            end if
        end do

        ! -- check for singularity
        if (maxval_loc < EPS) then
            info = k
            return
        end if

        ! -- swap rows k and piv in A
        if (piv /= k) then
            do j = 1, 19
                tmp_d    = A(k,j)
                A(k,j)   = A(piv,j)
                A(piv,j) = tmp_d
            end do
            ! -- swap rows k and piv in B
            do j = 1, NRHS
                tmp_d    = B(k,j)
                B(k,j)   = B(piv,j)
                B(piv,j) = tmp_d
            end do
        end if

        ! -- eliminate column k below diagonal
        do i = k+1, 19
            factor = A(i,k) / A(k,k)
            do j = k, 19
                A(i,j) = A(i,j) - factor * A(k,j)
            end do
            do j = 1, NRHS
                B(i,j) = B(i,j) - factor * B(k,j)
            end do
        end do
    end do

    ! -- back substitution
    do k = 19, 1, -1
        do j = 1, NRHS
            do i = k+1, 19
                B(k,j) = B(k,j) - A(k,i) * B(i,j)
            end do
            B(k,j) = B(k,j) / A(k,k)
        end do
    end do

end subroutine solve19


! =============================================================================
! yu_smooth_max (internal helper)
!
! Smooth maximum: b + 0.5 * (d + sqrt(d^2 + eps^2)), d = a - b.
! Matches Python smooth_max(a, b, eps=1e-30).
! =============================================================================
subroutine yu_smooth_max(a, b, result)
    implicit none
    double precision, intent(in)  :: a, b
    double precision, intent(out) :: result

    double precision :: d
    double precision, parameter :: EPS_SQRT = 1.0d-30

    d = a - b
    result = b + 0.5d0 * (d + sqrt(d*d + EPS_SQRT**2))

end subroutine yu_smooth_max


! =============================================================================
! yu_inner_mu_newton
!
! Inner Newton loop to solve for mu (stagnation surface update).
! Matches Python YUKinematic3D.user_defined_return_mapping lines 161-170.
!
! Newton iteration:
!   H_mu = sqrt(r_n^2 + 6*h*Fn / (1+mu))
!   F_mu = 3*Gn - r_n*(r_n + H_mu)*(1+mu)^2 - 3*h*Fn*(1+mu)
!   stop when F_mu < 1e-16
!   F_mu' = 3*h*Fn/H_mu*(r_n - H_mu) - 2*r_n*(1+mu)*(r_n + H_mu)
!   mu = mu - F_mu / F_mu'
!
! Parameters
! ----------
! h       [in]  : stagnation surface parameter
! r_n     [in]  : stagnation radius at step start
! Gn      [in]  : deviatoric_inner_product(g_xi, g_xi)
! Fn      [in]  : deviatoric_inner_product(g_xi, d_beta)
! mu_out  [out] : converged mu
! info    [out] : 0 = success, 1 = not converged (10 iter exceeded)
! =============================================================================
subroutine yu_inner_mu_newton(h, r_n, Gn, Fn, mu_out, info)
    implicit none
    double precision, intent(in)  :: h, r_n, Gn, Fn
    double precision, intent(out) :: mu_out
    integer,          intent(out) :: info

    integer :: i
    double precision :: mu, H_mu, F_mu, F_mu_prime
    double precision, parameter :: EPS_SQRT = 1.0d-30

    mu = 0.0d0
    info = 0  ! default: converged (mu=0 is returned if r_n is degenerate)

    ! Guard: when r_n=0 the mu equation has no physical solution (mu>=0).
    ! H_mu arg = 6*h*Fn which is negative when Fn<0, causing sqrt(negative).
    ! Correct answer is mu=0 (no stagnation update).
    if (r_n < 1.0d-14) then
        mu_out = 0.0d0
        return
    end if

    info = 1

    do i = 1, 10
        H_mu = sqrt(r_n*r_n + 6.0d0*h*Fn / (1.0d0 + mu) + EPS_SQRT**2)
        F_mu = 3.0d0*Gn - r_n*(r_n + H_mu)*(1.0d0+mu)**2 &
             - 3.0d0*h*Fn*(1.0d0 + mu)
        if (F_mu < 1.0d-16) then
            info = 0
            exit
        end if
        F_mu_prime = 3.0d0*h*Fn/H_mu*(r_n - H_mu) &
                   - 2.0d0*r_n*(1.0d0+mu)*(r_n + H_mu)
        mu = mu - F_mu / F_mu_prime
    end do

    mu_out = mu

end subroutine yu_inner_mu_newton


! =============================================================================
! yu_calc_ddsdde
!
! Computes the consistent algorithmic tangent (6x6) from the converged state.
! Matches Python YUKinematic3D.calc_ddsdde (:421-447).
!
! Algorithm:
!   1. Build the full 19x19 Jacobian via yu_calc_jacobian
!   2. Reconstruct C and compute C_inv (via 6x6 LU solve with I_6 RHS)
!   3. Pre-multiply stress-block rows (1..6) by C_inv
!   4. Invert the modified 19x19 Jacobian (solve with I_19 RHS)
!   5. Extract upper-left 6x6 block as ddsdde
!
! Parameters
! ----------
! (same 12 props + state at convergence as yu_calc_jacobian)
! ddsdde(6,6) [out] : consistent tangent
! =============================================================================
subroutine yu_calc_ddsdde(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                           stress_new, theta_new, beta_new, R_new, eps_eq_new, &
                           theta_max_new, R_n, dlambda, &
                           ddsdde, eps_eq_n)
    implicit none
    double precision, intent(in)  :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in)  :: stress_new(6), theta_new(6), beta_new(6)
    double precision, intent(in)  :: R_new, eps_eq_new, theta_max_new, R_n, dlambda
    double precision, intent(out) :: ddsdde(6,6)
    double precision, intent(in)  :: eps_eq_n  ! step-start eps_eq for C_n (rhs scaling)

    double precision :: jac(19,19), C(6,6), C_n(6,6), C_inv(6,6), C_work(6,6)
    double precision :: rhs19(19,19), jac_stress_block(6,19)
    integer :: ii, jj, kk, info_lu
    double precision, parameter :: ZERO = 0.0d0, ONE = 1.0d0

    ! Step 1: Build full 19x19 Jacobian (uses eps_eq_new for C in dRs/dlambda)
    call yu_calc_jacobian(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                          stress_new, theta_new, beta_new, R_new, eps_eq_new, &
                          theta_max_new, R_n, dlambda, &
                          jac)

    ! Step 2: Reconstruct C for Jacobian (eps_eq_new) and C_n for rhs scaling (eps_eq_n).
    ! C_n (step-start stiffness) is the correct rhs scaling: J*dx/de = [C_n; 0; ...]
    ! -> ddsdde = J^{-1}[0:6,0:6]*C_n. Using C(state_new) was a bug when E varies.
    call yu_kinematic_3d_elastic_stiffness(E, nu, eps_eq_new, Ea, xi_param, C)
    call yu_kinematic_3d_elastic_stiffness(E, nu, eps_eq_n,   Ea, xi_param, C_n)
    do ii = 1, 6
        do jj = 1, 6
            C_work(ii,jj) = C_n(ii,jj)
            C_inv(ii,jj) = ZERO
        end do
        C_inv(ii,ii) = ONE
    end do
    call solve6_inplace(C_work, C_inv, info_lu)
    if (info_lu /= 0) then
        do ii = 1, 6
            do jj = 1, 6
                ddsdde(ii,jj) = C_n(ii,jj)
            end do
        end do
        return
    end if

    ! Step 3: Pre-multiply stress-block rows (rows 1..6) by C_inv.
    ! Copy rows first to avoid overwrite aliasing.
    do ii = 1, 6
        do jj = 1, 19
            jac_stress_block(ii,jj) = jac(ii,jj)
        end do
    end do
    do ii = 1, 6
        do jj = 1, 19
            jac(ii,jj) = ZERO
            do kk = 1, 6
                jac(ii,jj) = jac(ii,jj) + C_inv(ii,kk) * jac_stress_block(kk,jj)
            end do
        end do
    end do

    ! Step 4: Invert modified Jacobian by solving jac * X = I_19
    do ii = 1, 19
        do jj = 1, 19
            rhs19(ii,jj) = ZERO
        end do
        rhs19(ii,ii) = ONE
    end do
    call solve19(jac, rhs19, 19, info_lu)
    if (info_lu /= 0) then
        do ii = 1, 6
            do jj = 1, 6
                ddsdde(ii,jj) = C(ii,jj)
            end do
        end do
        return
    end if

    ! Step 5: Extract upper-left 6x6 block
    do ii = 1, 6
        do jj = 1, 6
            ddsdde(ii,jj) = rhs19(ii,jj)
        end do
    end do

end subroutine yu_calc_ddsdde


! =============================================================================
! solve6_inplace -- in-place LU solver for a 6x6 system with NRHS=6
!
! Internal helper for yu_calc_ddsdde (C_inv computation).
! Solves A*X = B with Gaussian elimination + partial pivoting.
! =============================================================================
subroutine solve6_inplace(A, B, info)
    implicit none
    double precision, intent(inout) :: A(6,6), B(6,6)
    integer,          intent(out)   :: info

    integer          :: i, j, k, piv
    double precision :: maxval_loc, tmp_d, factor
    double precision, parameter :: EPS = 1.0d-14

    info = 0

    do k = 1, 6
        maxval_loc = abs(A(k,k))
        piv = k
        do i = k+1, 6
            if (abs(A(i,k)) > maxval_loc) then
                maxval_loc = abs(A(i,k))
                piv = i
            end if
        end do
        if (maxval_loc < EPS) then
            info = k
            return
        end if
        if (piv /= k) then
            do j = 1, 6
                tmp_d    = A(k,j)
                A(k,j)   = A(piv,j)
                A(piv,j) = tmp_d
                tmp_d    = B(k,j)
                B(k,j)   = B(piv,j)
                B(piv,j) = tmp_d
            end do
        end if
        do i = k+1, 6
            factor = A(i,k) / A(k,k)
            do j = k, 6
                A(i,j) = A(i,j) - factor * A(k,j)
            end do
            do j = 1, 6
                B(i,j) = B(i,j) - factor * B(k,j)
            end do
        end do
    end do

    do k = 6, 1, -1
        do j = 1, 6
            do i = k+1, 6
                B(k,j) = B(k,j) - A(k,i) * B(i,j)
            end do
            B(k,j) = B(k,j) / A(k,k)
        end do
    end do

end subroutine solve6_inplace


! =============================================================================
! yu_kinematic_3d
!
! f2py-callable core subroutine for YUKinematic3D.
! Implements the full return mapping (elastic predictor + NR plastic corrector)
! and the consistent algorithmic tangent (DDSDDE).
!
! Matches Python YUKinematic3D.user_defined_return_mapping (:129-190)
! and YUKinematic3D.user_defined_tangent / calc_ddsdde (:192-447).
!
! Argument order follows the FortranIntegrator.from_model convention:
!   (*param_fn(), stress_n, *state_tup_in_declaration_order, strain_inc)
!   -> (stress_out, *state_tup_out, ddsdde, n_iter, converged)
!
! State order (YUKinematic declaration, "stress" excluded):
!   theta(6), beta(6), R(scalar), q(6), r(scalar), eps_eq(scalar), theta_max(scalar)
!
! Parameters
! ----------
! E..xi_param   [in]  : 12 material parameters (model.param_names order)
! stress_n(6)   [in]  : stress at step start
! theta_n(6)    [in]  : relative backstress at step start
! beta_n(6)     [in]  : boundary backstress at step start
! R_n           [in]  : boundary radius increment at step start
! q_n(6)        [in]  : stagnation center at step start
! r_n           [in]  : stagnation radius at step start
! eps_eq_n      [in]  : equivalent plastic strain at step start
! theta_max_n   [in]  : max ||theta|| history at step start
! dstran(6)     [in]  : strain increment (engineering shear)
! stress_out(6)                  [out] : updated stress
! theta_out(6)                   [out] : updated theta
! beta_out(6)                    [out] : updated beta
! R_out                          [out] : updated R
! q_out(6)                       [out] : updated q
! r_out                          [out] : updated r
! eps_eq_out                     [out] : updated eps_eq
! theta_max_out                  [out] : updated theta_max
! ddsdde(6,6)                    [out] : consistent algorithmic tangent
! n_iter                         [out] : number of outer NR iterations performed
! converged                      [out] : 1 = converged, 0 = not converged
! =============================================================================
subroutine yu_kinematic_3d( &
        E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
        stress_n, &
        theta_n, beta_n, Rbnd_n, q_n, rstag_n, eps_eq_n, theta_max_n, &
        dstran, &
        stress_out, &
        theta_out, beta_out, Rbnd_out, q_out, rstag_out, eps_eq_out, theta_max_out, &
        ddsdde, &
        n_iter, converged)
    implicit none
    ! -- inputs: 12 props
    double precision, intent(in) :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    ! -- inputs: stress + state at step start
    double precision, intent(in) :: stress_n(6)
    double precision, intent(in) :: theta_n(6), beta_n(6), Rbnd_n, q_n(6), rstag_n, eps_eq_n, theta_max_n
    ! -- input: strain increment
    double precision, intent(in) :: dstran(6)
    ! -- outputs: updated stress + state
    double precision, intent(out) :: stress_out(6)
    double precision, intent(out) :: theta_out(6), beta_out(6), Rbnd_out, q_out(6), rstag_out, eps_eq_out, theta_max_out
    ! -- outputs: tangent + diagnostics
    double precision, intent(out) :: ddsdde(6,6)
    integer,          intent(out) :: n_iter
    integer,          intent(out) :: converged

    ! -- local variables
    double precision :: C(6,6), stress_trial(6)
    double precision :: stress_new(6), theta_new(6), beta_new(6)
    double precision :: Rbnd_new, rstag_new, eps_eq_new
    double precision :: q_new(6)
    double precision :: dlambda
    double precision :: r_vec(19), jac(19,19), dx(19,1)
    double precision :: r_norm, xi_trial(6), dev_s(6), xi_trial_norm
    double precision :: g_xi(6), d_beta(6), stag_norm, g_stag, g_flag
    double precision :: Gn, Fn, mu, delta_q(6), delta_rstag, delta_Rbnd, s_fac
    double precision :: H_mu_fin, theta_new_norm, theta_max_cand
    integer :: iter, ii, jj, info_lu, info_mu
    double precision, parameter :: TOL_NR   = 1.0d-10
    double precision, parameter :: EPS_SQRT = 1.0d-30

    ! ==========================================================================
    ! Elastic predictor: stress_trial = stress_n + C(eps_eq_n) @ dstran
    ! ==========================================================================
    call yu_kinematic_3d_elastic_stiffness(E, nu, eps_eq_n, Ea, xi_param, C)
    do ii = 1, 6
        stress_trial(ii) = stress_n(ii)
        do jj = 1, 6
            stress_trial(ii) = stress_trial(ii) + C(ii,jj) * dstran(jj)
        end do
    end do

    ! ==========================================================================
    ! Yield check at trial state
    ! ==========================================================================
    call yu_deviatoric(stress_trial, dev_s)
    do ii = 1, 6
        xi_trial(ii) = dev_s(ii) - theta_n(ii) - beta_n(ii)
    end do
    call yu_vonmises_norm(xi_trial, xi_trial_norm)

    if (xi_trial_norm <= Y) then
        ! Elastic step: accept trial stress
        do ii = 1, 6
            stress_out(ii) = stress_trial(ii)
            theta_out(ii)  = theta_n(ii)
            beta_out(ii)   = beta_n(ii)
            q_out(ii)      = q_n(ii)
        end do
        Rbnd_out      = Rbnd_n
        rstag_out     = rstag_n
        eps_eq_out    = eps_eq_n
        theta_max_out = theta_max_n
        do ii = 1, 6
            do jj = 1, 6
                ddsdde(ii,jj) = C(ii,jj)
            end do
        end do
        n_iter    = 0
        converged = 1
        return
    end if

    ! ==========================================================================
    ! Plastic step: outer NR loop (max 50 iter)
    ! ==========================================================================
    ! Initialize NR state
    do ii = 1, 6
        stress_new(ii) = stress_trial(ii)
        theta_new(ii)  = theta_n(ii)
        beta_new(ii)   = beta_n(ii)
        q_new(ii)      = q_n(ii)
    end do
    Rbnd_new   = Rbnd_n
    rstag_new  = rstag_n
    eps_eq_new = eps_eq_n
    dlambda    = 0.0d0
    n_iter     = 0
    converged  = 0

    do iter = 1, 50
        ! Residual (theta_max passed as state_n value -- not updated during NR)
        call yu_calc_residual(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                              stress_new, theta_new, beta_new, Rbnd_new, eps_eq_new, &
                              theta_n, beta_n, theta_max_n, &
                              stress_trial, dlambda, &
                              r_vec)

        ! Convergence check: L2 norm (matches np.linalg.norm)
        r_norm = 0.0d0
        do ii = 1, 19
            r_norm = r_norm + r_vec(ii)**2
        end do
        r_norm = sqrt(r_norm)

        if (r_norm < TOL_NR) then
            converged = 1
            n_iter    = iter - 1
            exit
        end if

        ! Jacobian and linear solve
        call yu_calc_jacobian(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                              stress_new, theta_new, beta_new, Rbnd_new, eps_eq_new, &
                              theta_max_n, Rbnd_n, dlambda, &
                              jac)

        do ii = 1, 19
            dx(ii,1) = r_vec(ii)
        end do
        call solve19(jac, dx, 1, info_lu)
        if (info_lu /= 0) then
            converged = 0
            n_iter    = iter
            exit
        end if

        ! NR update: state_new -= dx
        do ii = 1, 6
            stress_new(ii) = stress_new(ii) - dx(ii, 1)
            theta_new(ii)  = theta_new(ii)  - dx(7+ii, 1)
            beta_new(ii)   = beta_new(ii)   - dx(13+ii, 1)
        end do
        dlambda = dlambda - dx(7, 1)

        ! ----------------------------------------------------------------
        ! Explicit state updates (stagnation surface)
        ! g_flag: hard branch (matches user_defined_return_mapping:158)
        ! ----------------------------------------------------------------
        do ii = 1, 6
            d_beta(ii) = beta_new(ii) - beta_n(ii)
            g_xi(ii)   = beta_new(ii) - q_n(ii)
        end do
        call yu_vonmises_norm(g_xi, stag_norm)
        g_stag = stag_norm - rstag_n
        if (g_stag > 0.0d0) then
            g_flag = 1.0d0
        else
            g_flag = 0.0d0
        end if

        ! deviatoric_inner_product for SOLID_3D:
        !   Gn = sum(g_xi(1:3)^2) + 2*sum(g_xi(4:6)^2)   (Mandel)
        !   Fn = sum(g_xi(1:3)*d_beta(1:3)) + 2*sum(g_xi(4:6)*d_beta(4:6))
        Gn = 0.0d0
        Fn = 0.0d0
        do ii = 1, 3
            Gn = Gn + g_xi(ii)**2
            Fn = Fn + g_xi(ii) * d_beta(ii)
        end do
        do ii = 4, 6
            Gn = Gn + 2.0d0 * g_xi(ii)**2
            Fn = Fn + 2.0d0 * g_xi(ii) * d_beta(ii)
        end do

        ! Inner mu Newton
        call yu_inner_mu_newton(h, rstag_n, Gn, Fn, mu, info_mu)
        if (info_mu /= 0) then
            converged = 0
            n_iter    = iter
            exit
        end if

        ! delta_q, delta_rstag, delta_Rbnd
        do ii = 1, 6
            delta_q(ii) = mu * g_xi(ii) / (1.0d0 + mu)
        end do
        H_mu_fin   = sqrt(rstag_n*rstag_n + 6.0d0*h*Fn / (1.0d0 + mu) + EPS_SQRT**2)
        delta_rstag = 0.5d0 * (rstag_n + H_mu_fin) - rstag_n
        s_fac       = 1.0d0 / (1.0d0 + k * dlambda)
        delta_Rbnd  = s_fac * (Rbnd_n + k * Rsat * dlambda) - Rbnd_n

        Rbnd_new = Rbnd_n + g_flag * delta_Rbnd
        do ii = 1, 6
            q_new(ii) = q_n(ii) + g_flag * delta_q(ii)
        end do
        rstag_new  = rstag_n + g_flag * delta_rstag
        eps_eq_new = eps_eq_n + dlambda

    end do

    ! If we exhausted the loop without converging, n_iter not set yet
    if (converged == 0 .and. n_iter == 0) n_iter = 50

    ! ==========================================================================
    ! theta_max update (after outer NR, matches :181-182)
    ! ==========================================================================
    call yu_vonmises_norm(theta_new, theta_new_norm)
    call yu_smooth_max(theta_max_n, theta_new_norm, theta_max_cand)
    theta_max_out = theta_max_cand

    ! ==========================================================================
    ! Consistent tangent (DDSDDE)
    ! Use theta_max_n (step-start value, consistent with residual C_k).
    ! Pass eps_eq_n so yu_calc_ddsdde uses C(state_n) for rhs scaling.
    ! ==========================================================================
    call yu_calc_ddsdde(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                        stress_new, theta_new, beta_new, Rbnd_new, eps_eq_new, &
                        theta_max_n, Rbnd_n, dlambda, &
                        ddsdde, eps_eq_n)

    ! ==========================================================================
    ! Copy converged state to outputs
    ! ==========================================================================
    do ii = 1, 6
        stress_out(ii) = stress_new(ii)
        theta_out(ii)  = theta_new(ii)
        beta_out(ii)   = beta_new(ii)
        q_out(ii)      = q_new(ii)
    end do
    Rbnd_out   = Rbnd_new
    rstag_out  = rstag_new
    eps_eq_out = eps_eq_new

end subroutine yu_kinematic_3d


! =============================================================================
! umat -- ABAQUS UMAT interface for YUKinematic3D
!
! Thin shim that unpacks PROPS(12) and STATEV(22) into named arguments
! and calls yu_kinematic_3d.  Non-convergence is signalled to ABAQUS
! via PNEWDT = 0.5 (request to halve the time increment).
!
! PROPS layout: see file header (PROPS(1)=E .. PROPS(12)=xi_param)
! STATEV layout: see file header (STATEV(1..6)=theta .. STATEV(22)=theta_max)
! =============================================================================
subroutine umat(STRESS, STATEV, DDSDDE, SSE, SPD, SCD, &
                RPL, DDSDDT, DRPLDE, DRPLDT, &
                STRAN, DSTRAN, TIME, DTIME, TEMP, DTEMP, &
                PREDEF, DPRED, CMNAME, NDI, NSHR, NTENS, &
                NSTATV, PROPS, NPROPS, COORDS, DROT, PNEWDT, &
                CELENT, DFGRD0, DFGRD1, NOEL, NPT, LAYER, &
                KSPT, KSTEP, KINC)
    implicit none
    character(len=80),    intent(in)    :: CMNAME
    integer,              intent(in)    :: NDI, NSHR, NTENS, NSTATV, NPROPS
    integer,              intent(in)    :: NOEL, NPT, LAYER, KSPT, KSTEP, KINC
    double precision,     intent(inout) :: STRESS(NTENS)
    double precision,     intent(inout) :: STATEV(NSTATV)
    double precision,     intent(out)   :: DDSDDE(NTENS, NTENS)
    double precision,     intent(out)   :: SSE, SPD, SCD, RPL, DRPLDT
    double precision,     intent(out)   :: DDSDDT(NTENS), DRPLDE(NTENS)
    double precision,     intent(in)    :: STRAN(NTENS), DSTRAN(NTENS)
    double precision,     intent(in)    :: TIME(2), DTIME, TEMP, DTEMP
    double precision,     intent(in)    :: PREDEF(1), DPRED(1)
    double precision,     intent(in)    :: PROPS(NPROPS), COORDS(3)
    double precision,     intent(in)    :: DROT(3,3), DFGRD0(3,3), DFGRD1(3,3)
    double precision,     intent(inout) :: PNEWDT
    double precision,     intent(in)    :: CELENT

    double precision :: theta_n(6), beta_n(6), Rbnd_n, q_n(6), rstag_n
    double precision :: eps_eq_n, theta_max_n
    double precision :: stress_out(6), theta_out(6), beta_out(6), Rbnd_out
    double precision :: q_out(6), rstag_out, eps_eq_out, theta_max_out
    double precision :: ddsdde_local(6,6)
    double precision :: theta_rot(6), beta_rot(6), q_rot(6)
    double precision :: theta_out_rot(6), beta_out_rot(6), q_out_rot(6)
    integer :: i, j, n_iter, converged
    ! Diagnostic: per-increment failure counters (retained across calls via save)
    integer,          save :: yu_kstep = -1
    integer,          save :: yu_kinc  = -1
    integer,          save :: yu_nfail = 0
    integer,          save :: yu_nnr   = 0
    integer,          save :: yu_nmu   = 0
    double precision, save :: yu_time  = 0.0d0
    double precision, save :: yu_dtime = 0.0d0

    ! Guard: this UMAT is for 3-D solid elements only (NTENS=6, NDI=3, NSHR=3)
    if (NTENS /= 6 .or. NDI /= 3 .or. NSHR /= 3 .or. &
        NSTATV < 22 .or. NPROPS < 12) then
        write(7,'(A)') 'YUKinematic3D UMAT: incompatible element/material definition.'
        write(7,'(A,I0,A,I0,A,I0)') '  Expected NTENS=6 NDI=3 NSHR=3, got NTENS=', &
            NTENS, ' NDI=', NDI, ' NSHR=', NSHR
        write(7,'(A,I0,A,I0)') '  Expected NSTATV>=22 NPROPS>=12, got NSTATV=', &
            NSTATV, ' NPROPS=', NPROPS
        PNEWDT = 0.0d0
        return
    end if

    ! STATEV unpack + co-rotate tensor state variables
    ! STRESS is already co-rotated by ABAQUS before UMAT entry (no ROTSIG needed).
    ! theta, beta, q are deviatoric stress-like tensors stored in STATEV and must
    ! be co-rotated here so the return-mapping operates in the rotated frame.
    do i = 1, 6
        theta_n(i) = STATEV(i)
        beta_n(i)  = STATEV(6 + i)
        q_n(i)     = STATEV(13 + i)
    end do
    Rbnd_n      = STATEV(13)
    rstag_n     = STATEV(20)
    eps_eq_n    = STATEV(21)
    theta_max_n = STATEV(22)

    call ROTSIG(theta_n, DROT, theta_rot, 1, NDI, NSHR)
    call ROTSIG(beta_n,  DROT, beta_rot,  1, NDI, NSHR)
    call ROTSIG(q_n,     DROT, q_rot,     1, NDI, NSHR)

    call yu_kinematic_3d( &
        PROPS(1), PROPS(2), PROPS(3), PROPS(4), PROPS(5), PROPS(6), &
        PROPS(7), PROPS(8), PROPS(9), PROPS(10), PROPS(11), PROPS(12), &
        STRESS, &
        theta_rot, beta_rot, Rbnd_n, q_rot, rstag_n, eps_eq_n, theta_max_n, &
        DSTRAN, &
        stress_out, &
        theta_out, beta_out, Rbnd_out, q_out, rstag_out, eps_eq_out, theta_max_out, &
        ddsdde_local, n_iter, converged)

    ! Write-back stress and tangent
    do i = 1, NTENS
        STRESS(i) = stress_out(i)
        do j = 1, NTENS
            DDSDDE(i, j) = ddsdde_local(i, j)
        end do
    end do

    ! STATEV repack
    ! theta_out, beta_out, q_out are already in the rotated frame (return mapping
    ! was performed there); no further rotation needed before storing.
    do i = 1, 6
        STATEV(i)      = theta_out(i)
        STATEV(6 + i)  = beta_out(i)
        STATEV(13 + i) = q_out(i)
    end do
    STATEV(13) = Rbnd_out
    STATEV(20) = rstag_out
    STATEV(21) = eps_eq_out
    STATEV(22) = theta_max_out

    ! Non-convergence: request ABAQUS to reduce the time increment.
    ! Use min() to avoid accidentally relaxing a smaller cutback already in PNEWDT.
    !
    ! Diagnostic: accumulate per-increment failure counts and flush one summary
    ! line to the msg file when the increment changes.  Format (grep "YU-NC"):
    !   YU-NC  kstep  kinc     time      dtime   n_fail  n_nr  n_mu
    ! n_nr: return-mapping NR non-convergence (n_iter==50, all 50 iters consumed)
    ! n_mu: internal failure (mu Newton or solve19, n_iter<50)
    if (converged == 0) then
        ! Flush previous increment summary when increment changes
        if (KSTEP /= yu_kstep .or. KINC /= yu_kinc) then
            if (yu_kinc /= -1) then
                write(7,'(A,2I6,2ES11.3,3I8)') 'YU-NC ', &
                    yu_kstep, yu_kinc, yu_time, yu_dtime, &
                    yu_nfail, yu_nnr, yu_nmu
            end if
            yu_kstep = KSTEP
            yu_kinc  = KINC
            yu_time  = TIME(1)
            yu_dtime = DTIME
            yu_nfail = 0
            yu_nnr   = 0
            yu_nmu   = 0
            ! Detail line for the first failure of this increment:
            !   YU-DT kstep kinc elem pt dtime
            !   YU-DS dstran(1..6)  -- strain increment magnitude
            !   YU-SS stress(1..6)  -- stress at step start
            write(7,'(A,4I6,ES11.3)') 'YU-DT ', &
                KSTEP, KINC, NOEL, NPT, DTIME
            write(7,'(A,6ES10.3)') 'YU-DS ', (DSTRAN(i), i=1,6)
            write(7,'(A,6ES10.3)') 'YU-SS ', (STRESS(i), i=1,6)
            ! State variables at step start (before return mapping)
            ! YU-TH: theta(1..6), YU-BT: beta(1..6)
            ! YU-RQ: Rbnd  rstag  eps_eq
            write(7,'(A,6ES10.3)') 'YU-TH ', (STATEV(i), i=1,6)
            write(7,'(A,6ES10.3)') 'YU-BT ', (STATEV(6+i), i=1,6)
            write(7,'(A,3ES11.3)') 'YU-RQ ', STATEV(13), STATEV(20), STATEV(21)
        end if

        yu_nfail = yu_nfail + 1
        if (n_iter >= 50) then
            yu_nnr = yu_nnr + 1   ! all 50 NR iters consumed -> return mapping
        else
            yu_nmu = yu_nmu + 1   ! early exit: mu Newton or solve19 failure
        end if

        PNEWDT = min(PNEWDT, 0.5d0)
    end if

    ! Zero unused output fields
    SSE = 0.0d0; SPD = 0.0d0; SCD = 0.0d0; RPL = 0.0d0; DRPLDT = 0.0d0
    do i = 1, NTENS
        DDSDDT(i) = 0.0d0
        DRPLDE(i) = 0.0d0
    end do

end subroutine umat
