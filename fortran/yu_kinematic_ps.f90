! =============================================================================
! manforge -- YUKinematicPS UMAT (Yoshida-Uemori 2-surface + stagnation surface)
!
! Fortran port of YUKinematicPS (src/manforge/models/yu_kinematic.py).
! Plane stress under the P-metric convention (PLANE_STRESS_P), ntens = 3.
! Algorithm: fully-implicit Newton-Raphson on [stress(3), dlambda(1),
!            theta(3), beta(3)] = 10 unknowns.
!
! NOTE this is NOT a dimension-reduction of yu_kinematic_3d.f90.  The yield
! function there is the norm form ||xi|| - Y; here it is the quadratic form
!   f = 0.5 * xi^T P xi - Y^2/3
! so dlambda is (2/3)Y times smaller and every residual carries 2/3 and Y
! factors that have no counterpart in the 3-D file.  The two ports are
! deliberately separate.
!
! P metric (P = T * Pi_dev with T = diag(1,1,2)):
!         [  2/3  -1/3   0  ]
!   P  =  [ -1/3   2/3   0  ]
!         [   0     0    2  ]
! It absorbs both the deviatoric projection and the Mandel shear weighting, so
!   dev(s)                        is the IDENTITY  (P already projects)
!   dev(s):dev(t)                 = s^T P t
!   ||s||_vm                      = sqrt(1.5 * s^T P s)
!   P @ xi                        comes out in engineering shear
! Projecting before applying P would double-project.
!
! NOTE on naming:
!   - Fortran is case-insensitive, so the Python parameters B (bound surface
!     size) and b (backstress rate) would collide.  B is named "B_bnd" and
!     b is named "b_kin" throughout, matching yu_kinematic_3d.f90.
!   - Output matrix/vector arguments are named jmat / jvec / jval to avoid
!     collision with the loop variable j.
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
!   PROPS(9)  = b_kin    backstress (beta) hardening rate  (Python: self.b)
!   PROPS(10) = h        stagnation surface parameter
!   PROPS(11) = Ea       asymptotic modulus (nonlinear elasticity)
!   PROPS(12) = xi_param nonlinear elasticity decay parameter (Python: self.xi)
!
! State variables (STATEV, 13 slots, model.state_names without "stress"):
!   STATEV(1..3)   theta      relative backstress of yield surface
!   STATEV(4..6)   beta       relative backstress of boundary surface
!   STATEV(7)      R          radius increment of boundary surface
!   STATEV(8..10)  q          center of stagnation surface
!   STATEV(11)     r          radius of stagnation surface
!   STATEV(12)     eps_eq     equivalent plastic strain
!   STATEV(13)     theta_max  max ||theta|| in history
!
! Target elements: S4, S3, S4R, S3R (and CPS4-family plane stress), all of
! which enter with NDI=2, NSHR=1, NTENS=3.  Transverse shear is expected to be
! supplied via *TRANSVERSE SHEAR STIFFNESS on the section, not by this UMAT;
! the NTENS=5 shell path is rejected by the guard in umat.
!
! One `umat` per source file: ABAQUS links a single umat symbol per job, so
! this file and yu_kinematic_3d.f90 are alternatives, not companions.  A job
! mixing 3-D solids and shells with this material is not supported.
!
! Voigt convention (ntens=3): [s11, s22, s12]
!   Stress components: physical shear   (sigma_12 = tensor shear)
!   Strain components: engineering shear (gamma_12 = 2 * tensor shear)
!   Stress-like quantities store the raw in-plane tensor with the 33 component
!   identically zero -- NOT a 3-D deviator with s33 = -(s11+s22).
!
! NR unknown vector layout (10 components):
!   x = [stress(1..3), dlambda(4), theta(5..7), beta(8..10)]
! Residual vector layout (matching x):
!   r = [R_stress(1..3), R_yield(4), R_theta(5..7), R_beta(8..10)]
!
! Build (from fortran/ directory):
!   uv run python -m numpy.f2py -c abaqus_stubs.f90 yu_kinematic_ps.f90 -m yu_kinematic_ps
! =============================================================================


! =============================================================================
! yu_ps_pmat  (internal helper, not f2py-callable)
!
! Returns the plane-stress deviatoric metric P.
! =============================================================================
subroutine yu_ps_pmat(P)
    implicit none
    double precision, intent(out) :: P(3,3)

    P(1,1) =  2.0d0 / 3.0d0
    P(1,2) = -1.0d0 / 3.0d0
    P(1,3) =  0.0d0
    P(2,1) = -1.0d0 / 3.0d0
    P(2,2) =  2.0d0 / 3.0d0
    P(2,3) =  0.0d0
    P(3,1) =  0.0d0
    P(3,2) =  0.0d0
    P(3,3) =  2.0d0

end subroutine yu_ps_pmat


! =============================================================================
! yu_ps_pmul  (internal helper, not f2py-callable)
!
! Computes y = P @ x for the plane-stress metric.
! =============================================================================
subroutine yu_ps_pmul(x, y)
    implicit none
    double precision, intent(in)  :: x(3)
    double precision, intent(out) :: y(3)

    y(1) = (2.0d0 * x(1) - x(2)) / 3.0d0
    y(2) = (2.0d0 * x(2) - x(1)) / 3.0d0
    y(3) = 2.0d0 * x(3)

end subroutine yu_ps_pmul


! =============================================================================
! yu_ps_dev_inner  (internal helper, not f2py-callable)
!
! Deviatoric double contraction via the P metric: dev(s):dev(t) = s^T P t.
! =============================================================================
subroutine yu_ps_dev_inner(s, t, val)
    implicit none
    double precision, intent(in)  :: s(3), t(3)
    double precision, intent(out) :: val

    double precision :: Pt(3)

    call yu_ps_pmul(t, Pt)
    val = s(1) * Pt(1) + s(2) * Pt(2) + s(3) * Pt(3)

end subroutine yu_ps_dev_inner


! =============================================================================
! yu_ps_vonmises_norm  (internal helper, not f2py-callable)
!
! Von Mises equivalent norm under the P metric:
!   ||s||_vm = sqrt(1.5 * s^T P s)
! =============================================================================
subroutine yu_ps_vonmises_norm(s, s_norm)
    implicit none
    double precision, intent(in)  :: s(3)
    double precision, intent(out) :: s_norm

    double precision :: sPs
    double precision, parameter :: EPS_SQRT = 1.0d-12

    call yu_ps_dev_inner(s, s, sPs)
    s_norm = sqrt(1.5d0 * sPs + EPS_SQRT**2)

end subroutine yu_ps_vonmises_norm


! =============================================================================
! yu_ps_smooth_heaviside  (internal helper, not f2py-callable)
!
! Smooth Heaviside step 0.5*(1 + tanh(beta*x/2)) with beta = 500, matching
! manforge.utils.smooth.smooth_heaviside.
! =============================================================================
subroutine yu_ps_smooth_heaviside(x, hv)
    implicit none
    double precision, intent(in)  :: x
    double precision, intent(out) :: hv

    double precision, parameter :: BETA_HV = 500.0d0
    double precision :: arg

    arg = BETA_HV * x / 2.0d0
    if (arg > 350.0d0) then
        hv = 1.0d0
    else if (arg < -350.0d0) then
        hv = 0.0d0
    else
        hv = 0.5d0 * (1.0d0 + tanh(arg))
    end if

end subroutine yu_ps_smooth_heaviside


! =============================================================================
! yu_ps_smooth_sqrt  (internal helper, not f2py-callable)
!
! sqrt(x + eps^2), matching manforge.utils.smooth.smooth_sqrt.
! =============================================================================
subroutine yu_ps_smooth_sqrt(x, res)
    implicit none
    double precision, intent(in)  :: x
    double precision, intent(out) :: res

    double precision, parameter :: EPS_SQRT = 1.0d-12

    res = sqrt(x + EPS_SQRT**2)

end subroutine yu_ps_smooth_sqrt


! =============================================================================
! yu_ps_elastic_stiffness
!
! Plane-stress secant elastic stiffness C_e = f(eps_eq) * C_iso, where
!   f = 1 - (1 - Ea/E) * (1 - exp(-xi_param * eps_eq))
!
! C_iso is the Schur condensation of the 3-D isotropic stiffness onto the
! in-plane components (sigma_33 = 0), matching
! _PlaneStressPDimension.isotropic_C:
!   C_rr - outer(C_rc, C_rc) / C_cc   with retain = (11, 22, 12), condensed = 33
!
! Parameters
! ----------
! E         [in]  : Young's modulus
! nu        [in]  : Poisson's ratio
! eps_eq    [in]  : equivalent plastic strain (from state)
! Ea        [in]  : asymptotic modulus
! xi_param  [in]  : decay parameter (self.xi in Python)
! C         [out] : 3x3 Voigt stiffness
! =============================================================================
subroutine yu_ps_elastic_stiffness(E, nu, eps_eq, Ea, xi_param, C)
    implicit none
    double precision, intent(in)  :: E, nu, eps_eq, Ea, xi_param
    double precision, intent(out) :: C(3,3)

    double precision :: mu, lam, factor, denom
    integer :: ii, jj

    mu  = E / (2.0d0 * (1.0d0 + nu))
    lam = E * nu / ((1.0d0 + nu) * (1.0d0 - 2.0d0 * nu))

    do jj = 1, 3
        do ii = 1, 3
            C(ii,jj) = 0.0d0
        end do
    end do

    ! Condense out the 33 component: C_cc = lam + 2*mu, C_rc = (lam, lam, 0)
    denom = lam + 2.0d0 * mu
    C(1,1) = lam + 2.0d0 * mu - lam * lam / denom
    C(2,2) = C(1,1)
    C(1,2) = lam - lam * lam / denom
    C(2,1) = C(1,2)
    C(3,3) = mu

    factor = 1.0d0 - (1.0d0 - Ea / E) * (1.0d0 - exp(-xi_param * eps_eq))
    do jj = 1, 3
        do ii = 1, 3
            C(ii,jj) = factor * C(ii,jj)
        end do
    end do

end subroutine yu_ps_elastic_stiffness


! =============================================================================
! yu_ps_ck  (internal helper, not f2py-callable)
!
! Kinematic hardening rate with the post-reversal branch:
!   C_k = C_1 - (C_1 - C_2) * H(theta_max - (B_bnd - Y))
! =============================================================================
subroutine yu_ps_ck(C_1, C_2, B_bnd, Y, theta_max, C_k)
    implicit none
    double precision, intent(in)  :: C_1, C_2, B_bnd, Y, theta_max
    double precision, intent(out) :: C_k

    double precision :: hv

    call yu_ps_smooth_heaviside(theta_max - (B_bnd - Y), hv)
    C_k = C_1 - (C_1 - C_2) * hv

end subroutine yu_ps_ck


! =============================================================================
! yu_ps_calc_residual
!
! Computes the full 10-element residual vector for the Newton-Raphson system.
!   r = [R_stress(3), R_yield(1), R_theta(3), R_beta(3)]
!
! R_stress = sigma - sigma_trial + dlambda * C @ (P @ xi)
! R_yield  = 0.5 * xi^T P xi - Y^2/3
! R_theta  = theta - theta_n - 2/3*(C_k*a*xi - C_k*Y*sqrt(a/theta_bar)*theta)*dl
! R_beta   = beta  - beta_n  - 2/3*(k*b_kin*xi - k*Y*beta)*dl
! with xi = sigma - theta - beta  (dev is the identity under P) and
! a = B_bnd + R - Y.
!
! Parameters
! ----------
! E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
!             : 12 material parameters (order = model.param_names)
! stress_new(3), theta_new(3), beta_new(3)  : NR iterates
! R_new       : current R iterate
! eps_eq_new  : equivalent plastic strain at the current iterate
! theta_n(3), beta_n(3)  : states at step start
! theta_max_n : theta_max at step start (NOT updated during NR)
! stress_trial(3) : elastic predictor stress
! dlambda     : current dlambda iterate
! r_vec(10)   : output residual vector
! =============================================================================
subroutine yu_ps_calc_residual(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                                stress_new, theta_new, beta_new, R_new, eps_eq_new, &
                                theta_n, beta_n, theta_max_n, &
                                stress_trial, dlambda, &
                                r_vec)
    implicit none
    double precision, intent(in)  :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in)  :: stress_new(3), theta_new(3), beta_new(3)
    double precision, intent(in)  :: R_new, eps_eq_new
    double precision, intent(in)  :: theta_n(3), beta_n(3), theta_max_n
    double precision, intent(in)  :: stress_trial(3), dlambda
    double precision, intent(out) :: r_vec(10)

    double precision :: C(3,3), xi(3), Pxi(3), Cn_flow(3)
    double precision :: a, theta_bar, C_k, xiPxi, sq_a_tb
    integer :: ii, jj

    call yu_ps_elastic_stiffness(E, nu, eps_eq_new, Ea, xi_param, C)

    do ii = 1, 3
        xi(ii) = stress_new(ii) - theta_new(ii) - beta_new(ii)
    end do
    call yu_ps_pmul(xi, Pxi)

    a = B_bnd + R_new - Y
    call yu_ps_vonmises_norm(theta_new, theta_bar)
    call yu_ps_ck(C_1, C_2, B_bnd, Y, theta_max_n, C_k)
    call yu_ps_smooth_sqrt(a / theta_bar, sq_a_tb)

    ! R_stress = sigma - sigma_trial + dlambda * C @ (P @ xi)
    do ii = 1, 3
        Cn_flow(ii) = 0.0d0
        do jj = 1, 3
            Cn_flow(ii) = Cn_flow(ii) + C(ii,jj) * Pxi(jj)
        end do
    end do
    do ii = 1, 3
        r_vec(ii) = stress_new(ii) - stress_trial(ii) + dlambda * Cn_flow(ii)
    end do

    ! R_yield (quadratic form)
    call yu_ps_dev_inner(xi, xi, xiPxi)
    r_vec(4) = 0.5d0 * xiPxi - Y * Y / 3.0d0

    ! R_theta / R_beta
    do ii = 1, 3
        r_vec(4 + ii) = theta_new(ii) - theta_n(ii) &
            - 2.0d0 / 3.0d0 * (C_k * a * xi(ii) &
                               - C_k * Y * sq_a_tb * theta_new(ii)) * dlambda
        r_vec(7 + ii) = beta_new(ii) - beta_n(ii) &
            - 2.0d0 / 3.0d0 * (k * b_kin * xi(ii) - k * Y * beta_new(ii)) * dlambda
    end do

end subroutine yu_ps_calc_residual


! =============================================================================
! Jacobian blocks
!
! Row/column naming mirrors the Python methods so the benchmark tests map 1:1:
!   yu_ps_fy_*  <-> calc_fy_*  (R_yield  row)
!   yu_ps_fe_*  <-> calc_fe_*  (R_stress row)
!   yu_ps_ft_*  <-> calc_ft_*  (R_theta  row)
!   yu_ps_fb_*  <-> calc_fb_*  (R_beta   row)
! =============================================================================


! =============================================================================
! yu_ps_fy_fs / yu_ps_fy_ft / yu_ps_fy_fb / yu_ps_fy_fl
!
! d(R_yield)/d(sigma) = P @ eta;  d/d(theta) = d/d(beta) = -P @ eta;  d/dl = 0.
! =============================================================================
subroutine yu_ps_fy_fs(stress, theta, beta, jvec)
    implicit none
    double precision, intent(in)  :: stress(3), theta(3), beta(3)
    double precision, intent(out) :: jvec(3)

    double precision :: eta(3)
    integer :: ii

    do ii = 1, 3
        eta(ii) = stress(ii) - theta(ii) - beta(ii)
    end do
    call yu_ps_pmul(eta, jvec)

end subroutine yu_ps_fy_fs


subroutine yu_ps_fy_ft(stress, theta, beta, jvec)
    implicit none
    double precision, intent(in)  :: stress(3), theta(3), beta(3)
    double precision, intent(out) :: jvec(3)

    integer :: ii

    call yu_ps_fy_fs(stress, theta, beta, jvec)
    do ii = 1, 3
        jvec(ii) = -jvec(ii)
    end do

end subroutine yu_ps_fy_ft


subroutine yu_ps_fy_fb(stress, theta, beta, jvec)
    implicit none
    double precision, intent(in)  :: stress(3), theta(3), beta(3)
    double precision, intent(out) :: jvec(3)

    call yu_ps_fy_ft(stress, theta, beta, jvec)

end subroutine yu_ps_fy_fb


subroutine yu_ps_fy_fl(jval)
    implicit none
    double precision, intent(out) :: jval

    jval = 0.0d0

end subroutine yu_ps_fy_fl


! =============================================================================
! yu_ps_dc_deq  (internal helper, not f2py-callable)
!
! The dC/dsigma contribution shared by the R_stress blocks.  C depends on
! eps_eq through the E-degradation factor, and eps_eq depends on sigma through
! dlambda*sqrt(2/3*g), so d(R_sigma)/d(sigma) is NOT just I + dl*C@P.
!
!   fb      = -xi_param*(1 - Ea/E)*exp(-xi_param*eps_eq)
!   f       = E-degradation factor
!   deq_ds  = 2/3*dl*P@eta / sqrt(2/3 * eta^T P eta)
!   dC_deq  = fb/f * C @ deq_ds
! =============================================================================
subroutine yu_ps_dc_deq(E, Ea, xi_param, eps_eq, C, eta, dlambda, dC_deq, Peta)
    implicit none
    double precision, intent(in)  :: E, Ea, xi_param, eps_eq, C(3,3), eta(3), dlambda
    double precision, intent(out) :: dC_deq(3), Peta(3)

    double precision :: fb, f, etaPeta, sq, deq_ds(3)
    integer :: ii, jj

    fb = -xi_param * (1.0d0 - Ea / E) * exp(-xi_param * eps_eq)
    f  = 1.0d0 - (1.0d0 - Ea / E) * (1.0d0 - exp(-xi_param * eps_eq))

    call yu_ps_pmul(eta, Peta)
    call yu_ps_dev_inner(eta, eta, etaPeta)
    call yu_ps_smooth_sqrt(2.0d0 / 3.0d0 * etaPeta, sq)

    do ii = 1, 3
        deq_ds(ii) = 2.0d0 / 3.0d0 * dlambda * Peta(ii) / sq
    end do
    do ii = 1, 3
        dC_deq(ii) = 0.0d0
        do jj = 1, 3
            dC_deq(ii) = dC_deq(ii) + C(ii,jj) * deq_ds(jj)
        end do
        dC_deq(ii) = fb / f * dC_deq(ii)
    end do

end subroutine yu_ps_dc_deq


! =============================================================================
! yu_ps_fe_fs
!
! d(R_stress)/d(sigma) = I + dl*C@P + dl*outer(dC_deq, P@eta)
! =============================================================================
subroutine yu_ps_fe_fs(E, nu, Ea, xi_param, eps_eq, stress, theta, beta, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: E, nu, Ea, xi_param, eps_eq
    double precision, intent(in)  :: stress(3), theta(3), beta(3), dlambda
    double precision, intent(out) :: jmat(3,3)

    double precision :: C(3,3), P(3,3), CP(3,3), eta(3), dC_deq(3), Peta(3)
    integer :: ii, jj, kk

    call yu_ps_elastic_stiffness(E, nu, eps_eq, Ea, xi_param, C)
    call yu_ps_pmat(P)
    do ii = 1, 3
        eta(ii) = stress(ii) - theta(ii) - beta(ii)
    end do
    call yu_ps_dc_deq(E, Ea, xi_param, eps_eq, C, eta, dlambda, dC_deq, Peta)

    do jj = 1, 3
        do ii = 1, 3
            CP(ii,jj) = 0.0d0
            do kk = 1, 3
                CP(ii,jj) = CP(ii,jj) + C(ii,kk) * P(kk,jj)
            end do
        end do
    end do

    do jj = 1, 3
        do ii = 1, 3
            jmat(ii,jj) = dlambda * CP(ii,jj) + dlambda * dC_deq(ii) * Peta(jj)
        end do
        jmat(jj,jj) = jmat(jj,jj) + 1.0d0
    end do

end subroutine yu_ps_fe_fs


! =============================================================================
! yu_ps_fe_ft / yu_ps_fe_fb
!
! d(R_stress)/d(theta) = d/d(beta) = -dl*C@P - dl*outer(dC_deq, P@eta)
! =============================================================================
subroutine yu_ps_fe_ft(E, nu, Ea, xi_param, eps_eq, stress, theta, beta, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: E, nu, Ea, xi_param, eps_eq
    double precision, intent(in)  :: stress(3), theta(3), beta(3), dlambda
    double precision, intent(out) :: jmat(3,3)

    integer :: ii, jj

    call yu_ps_fe_fs(E, nu, Ea, xi_param, eps_eq, stress, theta, beta, dlambda, jmat)
    ! fe_fs = I + M  =>  fe_ft = -M
    do jj = 1, 3
        jmat(jj,jj) = jmat(jj,jj) - 1.0d0
    end do
    do jj = 1, 3
        do ii = 1, 3
            jmat(ii,jj) = -jmat(ii,jj)
        end do
    end do

end subroutine yu_ps_fe_ft


subroutine yu_ps_fe_fb(E, nu, Ea, xi_param, eps_eq, stress, theta, beta, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: E, nu, Ea, xi_param, eps_eq
    double precision, intent(in)  :: stress(3), theta(3), beta(3), dlambda
    double precision, intent(out) :: jmat(3,3)

    call yu_ps_fe_ft(E, nu, Ea, xi_param, eps_eq, stress, theta, beta, dlambda, jmat)

end subroutine yu_ps_fe_fb


! =============================================================================
! yu_ps_fe_fl
!
! d(R_stress)/d(dlambda) = C@P@eta + dl*dC_dl@P@eta, where
!   deq_dl = sqrt(2/3 * eta^T P eta)
!   dC_dl  = fb/f * deq_dl * C
! =============================================================================
subroutine yu_ps_fe_fl(E, nu, Ea, xi_param, eps_eq, stress, theta, beta, dlambda, jvec)
    implicit none
    double precision, intent(in)  :: E, nu, Ea, xi_param, eps_eq
    double precision, intent(in)  :: stress(3), theta(3), beta(3), dlambda
    double precision, intent(out) :: jvec(3)

    double precision :: C(3,3), eta(3), Peta(3), CPeta(3)
    double precision :: fb, f, etaPeta, deq_dl, scale
    integer :: ii, jj

    call yu_ps_elastic_stiffness(E, nu, eps_eq, Ea, xi_param, C)
    do ii = 1, 3
        eta(ii) = stress(ii) - theta(ii) - beta(ii)
    end do
    call yu_ps_pmul(eta, Peta)
    do ii = 1, 3
        CPeta(ii) = 0.0d0
        do jj = 1, 3
            CPeta(ii) = CPeta(ii) + C(ii,jj) * Peta(jj)
        end do
    end do

    fb = -xi_param * (1.0d0 - Ea / E) * exp(-xi_param * eps_eq)
    f  = 1.0d0 - (1.0d0 - Ea / E) * (1.0d0 - exp(-xi_param * eps_eq))
    call yu_ps_dev_inner(eta, eta, etaPeta)
    call yu_ps_smooth_sqrt(2.0d0 / 3.0d0 * etaPeta, deq_dl)

    ! dC_dl = fb/f * deq_dl * C, so dl*dC_dl@P@eta = dl*(fb/f*deq_dl)*C@P@eta
    scale = 1.0d0 + dlambda * fb / f * deq_dl
    do ii = 1, 3
        jvec(ii) = scale * CPeta(ii)
    end do

end subroutine yu_ps_fe_fl


! =============================================================================
! yu_ps_ft_fs / yu_ps_ft_fb
!
! d(R_theta)/d(sigma) = -2/3*C_k*a*dl*I
! d(R_theta)/d(beta)  = +2/3*C_k*a*dl*I
! =============================================================================
subroutine yu_ps_ft_fs(B_bnd, Y, C_1, C_2, R_new, theta_max, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: B_bnd, Y, C_1, C_2, R_new, theta_max, dlambda
    double precision, intent(out) :: jmat(3,3)

    double precision :: a, C_k
    integer :: ii, jj

    a = B_bnd + R_new - Y
    call yu_ps_ck(C_1, C_2, B_bnd, Y, theta_max, C_k)

    do jj = 1, 3
        do ii = 1, 3
            jmat(ii,jj) = 0.0d0
        end do
        jmat(jj,jj) = -2.0d0 / 3.0d0 * C_k * a * dlambda
    end do

end subroutine yu_ps_ft_fs


subroutine yu_ps_ft_fb(B_bnd, Y, C_1, C_2, R_new, theta_max, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: B_bnd, Y, C_1, C_2, R_new, theta_max, dlambda
    double precision, intent(out) :: jmat(3,3)

    integer :: ii, jj

    call yu_ps_ft_fs(B_bnd, Y, C_1, C_2, R_new, theta_max, dlambda, jmat)
    do jj = 1, 3
        do ii = 1, 3
            jmat(ii,jj) = -jmat(ii,jj)
        end do
    end do

end subroutine yu_ps_ft_fb


! =============================================================================
! yu_ps_ft_ft
!
! d(R_theta)/d(theta) = f1*I + f2*outer(theta, dthb_dth) with
!   f1 = 1 + 2/3*C_k*a*dl + 2/3*C_k*Y*dl*sqrt(a/theta_bar)
!   f2 = -C_k*Y*dl/(3*theta_bar) * sqrt(a/theta_bar)
!   dthb_dth = sqrt(1.5) * P@theta / sqrt(theta^T P theta)
! =============================================================================
subroutine yu_ps_ft_ft(B_bnd, Y, C_1, C_2, R_new, theta, theta_max, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: B_bnd, Y, C_1, C_2, R_new
    double precision, intent(in)  :: theta(3), theta_max, dlambda
    double precision, intent(out) :: jmat(3,3)

    double precision :: a, C_k, theta_bar, sq_a_tb, f1, f2
    double precision :: Pth(3), thPth, sq_thPth, dthb_dth(3)
    integer :: ii, jj

    a = B_bnd + R_new - Y
    call yu_ps_ck(C_1, C_2, B_bnd, Y, theta_max, C_k)
    call yu_ps_vonmises_norm(theta, theta_bar)
    call yu_ps_smooth_sqrt(a / theta_bar, sq_a_tb)

    f1 = 1.0d0 + 2.0d0 / 3.0d0 * C_k * a * dlambda &
       + 2.0d0 / 3.0d0 * C_k * Y * dlambda * sq_a_tb
    f2 = -C_k * Y * dlambda / 3.0d0 / theta_bar * sq_a_tb

    call yu_ps_pmul(theta, Pth)
    call yu_ps_dev_inner(theta, theta, thPth)
    call yu_ps_smooth_sqrt(thPth, sq_thPth)
    do ii = 1, 3
        dthb_dth(ii) = sqrt(1.5d0) * Pth(ii) / sq_thPth
    end do

    do jj = 1, 3
        do ii = 1, 3
            jmat(ii,jj) = f2 * theta(ii) * dthb_dth(jj)
        end do
        jmat(jj,jj) = jmat(jj,jj) + f1
    end do

end subroutine yu_ps_ft_ft


! =============================================================================
! yu_ps_ft_fl
!
! d(R_theta)/d(dlambda).  da_dl is gated: R is frozen unless the stagnation
! surface is active, so differentiating its evolution law unconditionally is
! wrong on exactly those steps.  Same test as yu_kinematic_3d's
! yu_prepare_rtheta: abs(R - R_n) > 1e-15*max(|R_n|, 1).
! =============================================================================
subroutine yu_ps_ft_fl(B_bnd, Y, k, Rsat, C_1, C_2, &
                       stress, theta, beta, R_new, R_n, theta_max, dlambda, jvec)
    implicit none
    double precision, intent(in)  :: B_bnd, Y, k, Rsat, C_1, C_2
    double precision, intent(in)  :: stress(3), theta(3), beta(3)
    double precision, intent(in)  :: R_new, R_n, theta_max, dlambda
    double precision, intent(out) :: jvec(3)

    double precision :: a, C_k, theta_bar, eta(3)
    double precision :: s, ds_dl, da_dl, active
    double precision :: f1, f2, f3, sq_a_tb, sq_inv
    integer :: ii

    a = B_bnd + R_new - Y
    call yu_ps_ck(C_1, C_2, B_bnd, Y, theta_max, C_k)
    call yu_ps_vonmises_norm(theta, theta_bar)
    do ii = 1, 3
        eta(ii) = stress(ii) - theta(ii) - beta(ii)
    end do

    s = 1.0d0 / (1.0d0 + 2.0d0 / 3.0d0 * k * Y * dlambda)
    ds_dl = -2.0d0 / 3.0d0 * k * Y * s * s

    if (abs(R_new - R_n) > 1.0d-15 * max(abs(R_n), 1.0d0)) then
        active = 1.0d0
    else
        active = 0.0d0
    end if
    da_dl = active * (ds_dl * (R_n + 2.0d0 / 3.0d0 * k * Y * Rsat * dlambda) &
                      + 2.0d0 / 3.0d0 * s * k * Y * Rsat)

    call yu_ps_smooth_sqrt(a / theta_bar, sq_a_tb)
    call yu_ps_smooth_sqrt(1.0d0 / a / theta_bar, sq_inv)

    f1 = -2.0d0 / 3.0d0 * C_k * dlambda * da_dl
    f2 = -2.0d0 / 3.0d0 * C_k * a
    f3 =  2.0d0 / 3.0d0 * C_k * Y * (dlambda / 2.0d0 * sq_inv * da_dl + sq_a_tb)

    do ii = 1, 3
        jvec(ii) = f1 * eta(ii) + f2 * eta(ii) + f3 * theta(ii)
    end do

end subroutine yu_ps_ft_fl


! =============================================================================
! yu_ps_fb_fs / yu_ps_fb_ft / yu_ps_fb_fb / yu_ps_fb_fl
!
! d(R_beta)/d(sigma) = -2/3*k*b_kin*dl*I
! d(R_beta)/d(theta) = +2/3*k*b_kin*dl*I
! d(R_beta)/d(beta)  = (1 + 2/3*k*b_kin*dl + 2/3*k*Y*dl)*I
! d(R_beta)/d(dl)    = 2/3*(k*Y*beta - k*b_kin*eta)
! =============================================================================
subroutine yu_ps_fb_fs(Y, k, b_kin, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: Y, k, b_kin, dlambda
    double precision, intent(out) :: jmat(3,3)

    integer :: ii, jj

    do jj = 1, 3
        do ii = 1, 3
            jmat(ii,jj) = 0.0d0
        end do
        jmat(jj,jj) = -2.0d0 / 3.0d0 * k * b_kin * dlambda
    end do

end subroutine yu_ps_fb_fs


subroutine yu_ps_fb_ft(Y, k, b_kin, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: Y, k, b_kin, dlambda
    double precision, intent(out) :: jmat(3,3)

    integer :: ii, jj

    call yu_ps_fb_fs(Y, k, b_kin, dlambda, jmat)
    do jj = 1, 3
        do ii = 1, 3
            jmat(ii,jj) = -jmat(ii,jj)
        end do
    end do

end subroutine yu_ps_fb_ft


subroutine yu_ps_fb_fb(Y, k, b_kin, dlambda, jmat)
    implicit none
    double precision, intent(in)  :: Y, k, b_kin, dlambda
    double precision, intent(out) :: jmat(3,3)

    double precision :: diag
    integer :: ii, jj

    diag = 1.0d0 + 2.0d0 / 3.0d0 * k * b_kin * dlambda &
         + 2.0d0 / 3.0d0 * k * Y * dlambda
    do jj = 1, 3
        do ii = 1, 3
            jmat(ii,jj) = 0.0d0
        end do
        jmat(jj,jj) = diag
    end do

end subroutine yu_ps_fb_fb


subroutine yu_ps_fb_fl(Y, k, b_kin, stress, theta, beta, jvec)
    implicit none
    double precision, intent(in)  :: Y, k, b_kin, stress(3), theta(3), beta(3)
    double precision, intent(out) :: jvec(3)

    double precision :: eta(3)
    integer :: ii

    do ii = 1, 3
        eta(ii) = stress(ii) - theta(ii) - beta(ii)
    end do
    do ii = 1, 3
        jvec(ii) = 2.0d0 / 3.0d0 * (k * Y * beta(ii) - k * b_kin * eta(ii))
    end do

end subroutine yu_ps_fb_fl


! =============================================================================
! yu_ps_calc_jacobian
!
! Assembles the 10x10 Newton-Raphson Jacobian.
!   rows/cols: stress(1..3), dlambda(4), theta(5..7), beta(8..10)
! =============================================================================
subroutine yu_ps_calc_jacobian(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                                stress_new, theta_new, beta_new, R_new, eps_eq_new, &
                                theta_max_n, R_n, dlambda, &
                                jac)
    implicit none
    double precision, intent(in)  :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in)  :: stress_new(3), theta_new(3), beta_new(3)
    double precision, intent(in)  :: R_new, eps_eq_new, theta_max_n, R_n, dlambda
    double precision, intent(out) :: jac(10,10)

    double precision :: Rs_s(3,3), Rs_t(3,3), Rs_b(3,3), Rs_l(3)
    double precision :: Rt_s(3,3), Rt_t(3,3), Rt_b(3,3), Rt_l(3)
    double precision :: Rb_s(3,3), Rb_t(3,3), Rb_b(3,3), Rb_l(3)
    double precision :: Ry_s(3), Ry_t(3), Ry_b(3), Ry_l
    integer :: ii, jj

    call yu_ps_fe_fs(E, nu, Ea, xi_param, eps_eq_new, stress_new, theta_new, beta_new, dlambda, Rs_s)
    call yu_ps_fe_ft(E, nu, Ea, xi_param, eps_eq_new, stress_new, theta_new, beta_new, dlambda, Rs_t)
    call yu_ps_fe_fb(E, nu, Ea, xi_param, eps_eq_new, stress_new, theta_new, beta_new, dlambda, Rs_b)
    call yu_ps_fe_fl(E, nu, Ea, xi_param, eps_eq_new, stress_new, theta_new, beta_new, dlambda, Rs_l)

    call yu_ps_ft_fs(B_bnd, Y, C_1, C_2, R_new, theta_max_n, dlambda, Rt_s)
    call yu_ps_ft_ft(B_bnd, Y, C_1, C_2, R_new, theta_new, theta_max_n, dlambda, Rt_t)
    call yu_ps_ft_fb(B_bnd, Y, C_1, C_2, R_new, theta_max_n, dlambda, Rt_b)
    call yu_ps_ft_fl(B_bnd, Y, k, Rsat, C_1, C_2, &
                     stress_new, theta_new, beta_new, R_new, R_n, theta_max_n, dlambda, Rt_l)

    call yu_ps_fb_fs(Y, k, b_kin, dlambda, Rb_s)
    call yu_ps_fb_ft(Y, k, b_kin, dlambda, Rb_t)
    call yu_ps_fb_fb(Y, k, b_kin, dlambda, Rb_b)
    call yu_ps_fb_fl(Y, k, b_kin, stress_new, theta_new, beta_new, Rb_l)

    call yu_ps_fy_fs(stress_new, theta_new, beta_new, Ry_s)
    call yu_ps_fy_ft(stress_new, theta_new, beta_new, Ry_t)
    call yu_ps_fy_fb(stress_new, theta_new, beta_new, Ry_b)
    call yu_ps_fy_fl(Ry_l)

    do jj = 1, 10
        do ii = 1, 10
            jac(ii,jj) = 0.0d0
        end do
    end do

    ! Rows 1..3: R_stress
    do ii = 1, 3
        do jj = 1, 3
            jac(ii, jj)     = Rs_s(ii,jj)
            jac(ii, 4 + jj) = Rs_t(ii,jj)
            jac(ii, 7 + jj) = Rs_b(ii,jj)
        end do
        jac(ii, 4) = Rs_l(ii)
    end do

    ! Row 4: R_yield
    do jj = 1, 3
        jac(4, jj)     = Ry_s(jj)
        jac(4, 4 + jj) = Ry_t(jj)
        jac(4, 7 + jj) = Ry_b(jj)
    end do
    jac(4, 4) = Ry_l

    ! Rows 5..7: R_theta
    do ii = 1, 3
        do jj = 1, 3
            jac(4 + ii, jj)     = Rt_s(ii,jj)
            jac(4 + ii, 4 + jj) = Rt_t(ii,jj)
            jac(4 + ii, 7 + jj) = Rt_b(ii,jj)
        end do
        jac(4 + ii, 4) = Rt_l(ii)
    end do

    ! Rows 8..10: R_beta
    do ii = 1, 3
        do jj = 1, 3
            jac(7 + ii, jj)     = Rb_s(ii,jj)
            jac(7 + ii, 4 + jj) = Rb_t(ii,jj)
            jac(7 + ii, 7 + jj) = Rb_b(ii,jj)
        end do
        jac(7 + ii, 4) = Rb_l(ii)
    end do

end subroutine yu_ps_calc_jacobian


! =============================================================================
! yu_ps_solve  (internal helper, not f2py-callable)
!
! In-place LU solve of an N x N system with NRHS right-hand sides, Gaussian
! elimination with partial pivoting.  B is overwritten with the solution.
! info = 0 on success, k if pivot k is essentially zero.
! =============================================================================
subroutine yu_ps_solve(N, A, LDB, B, NRHS, info)
    implicit none
    integer,          intent(in)    :: N, LDB, NRHS
    double precision, intent(inout) :: A(N,N)
    double precision, intent(inout) :: B(LDB,NRHS)
    integer,          intent(out)   :: info

    integer :: i, j, kk, piv
    double precision :: amax, tmp, factor

    info = 0

    do kk = 1, N
        piv = kk
        amax = abs(A(kk,kk))
        do i = kk + 1, N
            if (abs(A(i,kk)) > amax) then
                amax = abs(A(i,kk))
                piv = i
            end if
        end do
        if (amax < 1.0d-300) then
            info = kk
            return
        end if
        if (piv /= kk) then
            do j = 1, N
                tmp = A(kk,j)
                A(kk,j) = A(piv,j)
                A(piv,j) = tmp
            end do
            do j = 1, NRHS
                tmp = B(kk,j)
                B(kk,j) = B(piv,j)
                B(piv,j) = tmp
            end do
        end if
        do i = kk + 1, N
            factor = A(i,kk) / A(kk,kk)
            if (factor /= 0.0d0) then
                do j = kk, N
                    A(i,j) = A(i,j) - factor * A(kk,j)
                end do
                do j = 1, NRHS
                    B(i,j) = B(i,j) - factor * B(kk,j)
                end do
            end if
        end do
    end do

    do j = 1, NRHS
        do i = N, 1, -1
            tmp = B(i,j)
            do kk = i + 1, N
                tmp = tmp - A(i,kk) * B(kk,j)
            end do
            B(i,j) = tmp / A(i,i)
        end do
    end do

end subroutine yu_ps_solve


! =============================================================================
! yu_ps_calc_ddsdde
!
! Consistent algorithmic tangent (3x3) from the converged state.
! Matches Python YUKinematicPS.calc_ddsdde.
!
!   1. Build the 10x10 Jacobian
!   2. Solve J X = [I_3; 0] for the leading 3 columns of J^-1
!   3. ddsdde = X[1:3,1:3] @ C_n
!
! C_n (step-start stiffness) is the correct scaling: sigma_trial is built with
! C(state_n), so J*dx/de = [C_n; 0; ...].  Using C(state_new) is a bug when E
! varies with eps_eq.
!
! Only the first THREE columns of J^-1 are needed and C_n enters as a plain
! right-multiply.  The equivalent route via M = blockdiag(C_n^-1, I_7),
!   (M J)^-1 = J^-1 M^-1 = J^-1 @ blockdiag(C_n, I_7),
! costs an explicit 3x3 inverse, a 3x10 row premultiply and 7 extra right-hand
! sides for the same answer, so it is not taken.
!
! Parameters
! ----------
! (12 props + converged state, as yu_ps_calc_jacobian)
! eps_eq_n     [in]  : step-start eps_eq, for C_n
! ddsdde(3,3)  [out] : consistent tangent
! =============================================================================
subroutine yu_ps_calc_ddsdde(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                              stress_new, theta_new, beta_new, R_new, eps_eq_new, &
                              theta_max_n, R_n, dlambda, eps_eq_n, &
                              ddsdde)
    implicit none
    double precision, intent(in)  :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in)  :: stress_new(3), theta_new(3), beta_new(3)
    double precision, intent(in)  :: R_new, eps_eq_new, theta_max_n, R_n, dlambda, eps_eq_n
    double precision, intent(out) :: ddsdde(3,3)

    double precision :: jac(10,10), C_n(3,3), X(10,3)
    integer :: ii, jj, kk, info_lu

    call yu_ps_calc_jacobian(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                             stress_new, theta_new, beta_new, R_new, eps_eq_new, &
                             theta_max_n, R_n, dlambda, jac)

    call yu_ps_elastic_stiffness(E, nu, eps_eq_n, Ea, xi_param, C_n)

    do jj = 1, 3
        do ii = 1, 10
            X(ii,jj) = 0.0d0
        end do
        X(jj,jj) = 1.0d0
    end do
    call yu_ps_solve(10, jac, 10, X, 3, info_lu)
    if (info_lu /= 0) then
        do jj = 1, 3
            do ii = 1, 3
                ddsdde(ii,jj) = C_n(ii,jj)
            end do
        end do
        return
    end if

    do jj = 1, 3
        do ii = 1, 3
            ddsdde(ii,jj) = 0.0d0
            do kk = 1, 3
                ddsdde(ii,jj) = ddsdde(ii,jj) + X(ii,kk) * C_n(kk,jj)
            end do
        end do
    end do

end subroutine yu_ps_calc_ddsdde


! =============================================================================
! yu_ps_inner_mu_newton  (internal helper, not f2py-callable)
!
! Inner Newton loop for mu (stagnation surface update).  Mirrors
! yu_inner_mu_newton in yu_kinematic_3d.f90; the P metric enters only through
! the caller's Gn / Fn.
! =============================================================================
subroutine yu_ps_inner_mu_newton(h, r_n, Gn, Fn, mu_out, info)
    implicit none
    double precision, intent(in)  :: h, r_n, Gn, Fn
    double precision, intent(out) :: mu_out
    integer,          intent(out) :: info

    integer :: i
    double precision :: mu, H_mu, F_mu, F_mu_prime

    mu = 0.0d0
    info = 0

    ! When r_n = 0 the mu equation has no physical solution (mu >= 0) and the
    ! sqrt argument can go negative; mu = 0 (no stagnation update) is correct.
    if (r_n < 1.0d-14) then
        mu_out = 0.0d0
        return
    end if

    info = 1
    do i = 1, 10
        call yu_ps_smooth_sqrt(r_n * r_n + 6.0d0 * h * Fn / (1.0d0 + mu), H_mu)
        F_mu = 3.0d0 * Gn - r_n * (r_n + H_mu) * (1.0d0 + mu)**2 &
             - 3.0d0 * h * Fn * (1.0d0 + mu)
        ! F_mu decreases in mu, so F_mu(0) < 0 puts the root at mu < 0: beta is
        ! inside the surface and the stagnation state holds, so mu = 0 is the
        ! answer.  Otherwise only the magnitude may stop the iteration -- a
        ! signed test accepts the first step past the root, which leaves beta
        ! off the stagnation surface by ~1e-1 (see the Python counterpart).
        if (F_mu < 0.0d0 .and. mu <= 0.0d0) then
            mu   = 0.0d0
            info = 0
            exit
        end if
        if (abs(F_mu) < 1.0d-12 * max(abs(3.0d0 * Gn), 1.0d0)) then
            info = 0
            exit
        end if
        F_mu_prime = 3.0d0 * h * Fn / H_mu * (r_n - H_mu) &
                   - 2.0d0 * r_n * (1.0d0 + mu) * (r_n + H_mu)
        mu = mu - F_mu / F_mu_prime
    end do

    mu_out = mu

end subroutine yu_ps_inner_mu_newton


! =============================================================================
! yu_kinematic_ps
!
! Main entry: one constitutive integration step (elastic trial -> yield check
! -> return mapping -> consistent tangent).  Equivalent to one UMAT call.
! Mirrors YUKinematicPS.user_defined_return_mapping + user_defined_tangent.
!
! The stagnation flag uses smooth_heaviside, re-evaluated every NR iteration --
! the same gate as update_state and the Python analytical route.  See
! yu_kinematic_3d.f90 for why the earlier hard-branch latch was removed.
!
! Parameters
! ----------
! E .. xi_param                  [in]  : 12 material parameters
! stress_n(3)                    [in]  : stress at step start
! theta_n(3), beta_n(3)          [in]  : backstresses at step start
! Rbnd_n, q_n(3), rstag_n        [in]  : boundary/stagnation state at step start
! eps_eq_n, theta_max_n          [in]  : scalar history at step start
! dstran(3)                      [in]  : strain increment (engineering shear)
! stress_out(3)                  [out] : updated stress
! theta_out(3), beta_out(3)      [out] : updated backstresses
! Rbnd_out, q_out(3), rstag_out  [out] : updated boundary/stagnation state
! eps_eq_out, theta_max_out      [out] : updated scalar history
! ddsdde(3,3)                    [out] : consistent algorithmic tangent
! n_iter                         [out] : outer NR iterations performed
! converged                      [out] : 1 = converged, 0 = not converged
! r_hist(50)                     [out] : residual norm per NR iteration
! fail_code                      [out] : 0 = converged, 1 = outer NR exhausted,
!                                        2 = mu Newton failed, 3 = linear solve
!                                        failed.  n_iter alone cannot separate
!                                        2 from 3 -- both exit early.
! fail_diag(6)                   [out] : state at the failing iteration:
!                                        [r_n, Fn, Gn, sqrt_arg, dbeta_norm,
!                                        dlambda].  sqrt_arg = r_n^2 + 6*h*Fn
!                                        is the mu Newton radicand: negative
!                                        means the mu equation has no real
!                                        root, which is the failure mode a
!                                        cutback cannot fix.  Zero on success.
! =============================================================================
subroutine yu_kinematic_ps( &
        E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
        stress_n, &
        theta_n, beta_n, Rbnd_n, q_n, rstag_n, eps_eq_n, theta_max_n, &
        dstran, &
        stress_out, &
        theta_out, beta_out, Rbnd_out, q_out, rstag_out, eps_eq_out, theta_max_out, &
        ddsdde, &
        n_iter, converged, r_hist, fail_code, fail_diag)
    implicit none
    double precision, intent(in) :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in) :: stress_n(3)
    double precision, intent(in) :: theta_n(3), beta_n(3), Rbnd_n, q_n(3), rstag_n
    double precision, intent(in) :: eps_eq_n, theta_max_n
    double precision, intent(in) :: dstran(3)
    double precision, intent(out) :: stress_out(3)
    double precision, intent(out) :: theta_out(3), beta_out(3), Rbnd_out, q_out(3)
    double precision, intent(out) :: rstag_out, eps_eq_out, theta_max_out
    double precision, intent(out) :: ddsdde(3,3)
    integer,          intent(out) :: n_iter, converged
    double precision, intent(out) :: r_hist(50)
    integer,          intent(out) :: fail_code
    double precision, intent(out) :: fail_diag(6)

    double precision :: C(3,3), stress_trial(3)
    double precision :: stress_new(3), theta_new(3), beta_new(3)
    double precision :: Rbnd_new, q_new(3), rstag_new, eps_eq_new
    double precision :: xi_trial(3), f_trial, xiPxi
    double precision :: r_vec(10), jac(10,10), dx(10,1)
    double precision :: dlambda, r_norm
    double precision :: eta(3), g, delta_eps_eq, s_fac
    double precision :: d_beta(3), g_xi(3), stag_norm, g_stag, g_flag
    ! delta_rstag / delta_Rbnd, not delta_r / delta_R: Fortran is
    ! case-insensitive, so the Python names for the stagnation radius and the
    ! bound-surface radius would be the same symbol.
    double precision :: Gn, Fn, mu, delta_q(3), delta_rstag, delta_Rbnd, H_val
    double precision :: theta_norm_final, dbeta_norm
    integer :: ii, jj, iter, info_lu, info_mu
    double precision, parameter :: TOL_NR = 1.0d-10

    do ii = 1, 50
        r_hist(ii) = 0.0d0
    end do
    fail_code = 0
    do ii = 1, 6
        fail_diag(ii) = 0.0d0
    end do

    ! ---- elastic predictor -------------------------------------------------
    call yu_ps_elastic_stiffness(E, nu, eps_eq_n, Ea, xi_param, C)
    do ii = 1, 3
        stress_trial(ii) = stress_n(ii)
        do jj = 1, 3
            stress_trial(ii) = stress_trial(ii) + C(ii,jj) * dstran(jj)
        end do
    end do

    do ii = 1, 3
        xi_trial(ii) = stress_trial(ii) - theta_n(ii) - beta_n(ii)
    end do
    call yu_ps_dev_inner(xi_trial, xi_trial, xiPxi)
    f_trial = 0.5d0 * xiPxi - Y * Y / 3.0d0

    if (f_trial <= 0.0d0) then
        ! Elastic step: state unchanged, tangent is the secant stiffness.
        do ii = 1, 3
            stress_out(ii)  = stress_trial(ii)
            theta_out(ii)   = theta_n(ii)
            beta_out(ii)    = beta_n(ii)
            q_out(ii)       = q_n(ii)
            do jj = 1, 3
                ddsdde(ii,jj) = C(ii,jj)
            end do
        end do
        Rbnd_out      = Rbnd_n
        rstag_out     = rstag_n
        eps_eq_out    = eps_eq_n
        theta_max_out = theta_max_n
        n_iter        = 0
        converged     = 1
        return
    end if

    ! ---- plastic: Newton-Raphson on [stress, dlambda, theta, beta] ---------
    do ii = 1, 3
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
        call yu_ps_calc_residual(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                                 stress_new, theta_new, beta_new, Rbnd_new, eps_eq_new, &
                                 theta_n, beta_n, theta_max_n, &
                                 stress_trial, dlambda, r_vec)

        r_norm = 0.0d0
        do ii = 1, 10
            r_norm = r_norm + r_vec(ii)**2
        end do
        r_norm = sqrt(r_norm)
        r_hist(iter) = r_norm

        if (r_norm < TOL_NR) then
            converged = 1
            n_iter    = iter - 1
            exit
        end if

        call yu_ps_calc_jacobian(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                                 stress_new, theta_new, beta_new, Rbnd_new, eps_eq_new, &
                                 theta_max_n, Rbnd_n, dlambda, jac)

        do ii = 1, 10
            dx(ii,1) = r_vec(ii)
        end do
        call yu_ps_solve(10, jac, 10, dx, 1, info_lu)
        if (info_lu /= 0) then
            converged = 0
            n_iter    = iter
            fail_code = 3
            fail_diag(6) = dlambda
            exit
        end if

        do ii = 1, 3
            stress_new(ii) = stress_new(ii) - dx(ii, 1)
            theta_new(ii)  = theta_new(ii)  - dx(4 + ii, 1)
            beta_new(ii)   = beta_new(ii)   - dx(7 + ii, 1)
        end do
        dlambda = dlambda - dx(4, 1)

        ! ---- explicit state updates (stagnation surface) -------------------
        do ii = 1, 3
            eta(ii)    = stress_new(ii) - theta_new(ii) - beta_new(ii)
            d_beta(ii) = beta_new(ii) - beta_n(ii)
            g_xi(ii)   = beta_new(ii) - q_n(ii)
        end do

        call yu_ps_dev_inner(eta, eta, g)
        call yu_ps_smooth_sqrt(2.0d0 / 3.0d0 * g, delta_eps_eq)
        delta_eps_eq = dlambda * delta_eps_eq

        s_fac = 1.0d0 / (1.0d0 + 2.0d0 / 3.0d0 * k * Y * dlambda)

        ! Smooth gate, re-evaluated every iteration -- see yu_kinematic_3d.f90
        ! for why the hard-branch latch was removed.
        call yu_ps_vonmises_norm(g_xi, stag_norm)
        g_stag = stag_norm - rstag_n
        call yu_ps_smooth_heaviside(g_stag + 1.0d-10, g_flag)

        call yu_ps_dev_inner(g_xi, g_xi, Gn)
        call yu_ps_dev_inner(g_xi, d_beta, Fn)
        call yu_ps_inner_mu_newton(h, rstag_n, Gn, Fn, mu, info_mu)
        ! An unconverged mu leaves delta_q / delta_rstag meaningless, yet the
        ! outer NR can still reach its tolerance and report success -- the
        ! caller would then store a corrupt stagnation surface.  Python raises
        ! here (update_state / user_defined_return_mapping); the Fortran
        ! equivalent is to fail the step so the UMAT can request a cutback.
        if (info_mu /= 0) then
            converged = 0
            n_iter    = iter
            fail_code = 2
            call yu_ps_dev_inner(d_beta, d_beta, dbeta_norm)
            dbeta_norm = sqrt(max(dbeta_norm, 0.0d0))
            fail_diag(1) = rstag_n
            fail_diag(2) = Fn
            fail_diag(3) = Gn
            fail_diag(4) = rstag_n * rstag_n + 6.0d0 * h * Fn
            fail_diag(5) = dbeta_norm
            fail_diag(6) = dlambda
            exit
        end if

        do ii = 1, 3
            delta_q(ii) = mu * g_xi(ii) / (1.0d0 + mu)
        end do
        call yu_ps_smooth_sqrt(rstag_n * rstag_n + 6.0d0 * h * Fn / (1.0d0 + mu), H_val)
        delta_rstag = 0.5d0 * (rstag_n + H_val) - rstag_n
        delta_Rbnd  = s_fac * (Rbnd_n + 2.0d0 / 3.0d0 * Y * k * Rsat * dlambda) - Rbnd_n

        Rbnd_new  = Rbnd_n  + g_flag * delta_Rbnd
        rstag_new = rstag_n + g_flag * delta_rstag
        do ii = 1, 3
            q_new(ii) = q_n(ii) + g_flag * delta_q(ii)
        end do
        eps_eq_new = eps_eq_n + delta_eps_eq
    end do

    ! Exhausting the loop leaves n_iter at its initial 0; 50 distinguishes
    ! "outer NR ran out of iterations" from the early exits above, which the
    ! UMAT diagnostics use to separate NR from internal (mu / solve) failures.
    if (converged == 0 .and. n_iter == 0) then
        n_iter    = 50
        fail_code = 1
        call yu_ps_dev_inner(d_beta, d_beta, dbeta_norm)
        dbeta_norm = sqrt(max(dbeta_norm, 0.0d0))
        fail_diag(1) = rstag_n
        fail_diag(2) = Fn
        fail_diag(3) = Gn
        fail_diag(4) = rstag_n * rstag_n + 6.0d0 * h * Fn
        fail_diag(5) = dbeta_norm
        fail_diag(6) = dlambda
    end if

    ! theta_max is updated once, after convergence
    call yu_ps_vonmises_norm(theta_new, theta_norm_final)
    theta_max_out = max(theta_max_n, theta_norm_final)

    do ii = 1, 3
        stress_out(ii) = stress_new(ii)
        theta_out(ii)  = theta_new(ii)
        beta_out(ii)   = beta_new(ii)
        q_out(ii)      = q_new(ii)
    end do
    Rbnd_out   = Rbnd_new
    rstag_out  = rstag_new
    eps_eq_out = eps_eq_new

    call yu_ps_calc_ddsdde(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                           stress_new, theta_new, beta_new, Rbnd_new, eps_eq_new, &
                           theta_max_n, Rbnd_n, dlambda, eps_eq_n, ddsdde)

end subroutine yu_kinematic_ps


! =============================================================================
! umat -- ABAQUS UMAT interface for YUKinematicPS
!
! Thin shim that unpacks PROPS(12) and STATEV(13) into named arguments and
! calls yu_kinematic_ps.  Non-convergence is signalled to ABAQUS via
! PNEWDT = 0.5 (request to halve the time increment).
!
! PROPS / STATEV layouts: see file header.
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

    double precision :: theta_n(3), beta_n(3), Rbnd_n, q_n(3), rstag_n
    double precision :: eps_eq_n, theta_max_n
    double precision :: stress_out(3), theta_out(3), beta_out(3), Rbnd_out
    double precision :: q_out(3), rstag_out, eps_eq_out, theta_max_out
    double precision :: ddsdde_local(3,3)
    double precision :: theta_rot(3), beta_rot(3), q_rot(3)
    double precision :: r_hist_val(50)
    double precision :: fail_diag(6)
    integer :: i, j, n_iter, converged, fail_code
    ! Debug output control: change to .true. to enable YU-* diagnostic writes
    logical, parameter :: YU_DEBUG = .false.
    ! Detail lines per increment.  Failures usually arrive in bulk (a whole
    ! plastic band at once), so an unbounded dump buries the pattern; the true
    ! count still appears in the YU-NC summary.
    integer, parameter :: YU_MAX_DETAIL = 20

    ! Diagnostic: per-increment failure counters (retained across calls via save).
    ! NOTE these are not thread-safe.  Under domain-level parallelism
    ! (mp_mode=threads) the counters race and the detail cap is approximate;
    ! results are unaffected.  Run cpus=1 when reading these numbers.
    integer,          save :: yu_kstep = -1
    integer,          save :: yu_kinc  = -1
    integer,          save :: yu_nfail = 0
    integer,          save :: yu_nnr   = 0
    integer,          save :: yu_nmu   = 0
    integer,          save :: yu_nlu   = 0
    integer,          save :: yu_ndet  = 0
    double precision, save :: yu_time  = 0.0d0
    double precision, save :: yu_dtime = 0.0d0

    ! Guard: plane stress / conventional shell only (S4, S3, S4R, S3R, CPS4...).
    ! The NTENS=5 shell path (transverse shear carried by the material) is
    ! rejected: use *TRANSVERSE SHEAR STIFFNESS on the section instead.
    if (NTENS /= 3 .or. NDI /= 2 .or. NSHR /= 1 .or. &
        NSTATV < 13 .or. NPROPS < 12) then
        write(7,'(A)') 'YUKinematicPS UMAT: incompatible element/material definition.'
        write(7,'(A,I0,A,I0,A,I0)') '  Expected NTENS=3 NDI=2 NSHR=1, got NTENS=', &
            NTENS, ' NDI=', NDI, ' NSHR=', NSHR
        write(7,'(A,I0,A,I0)') '  Expected NSTATV>=13 NPROPS>=12, got NSTATV=', &
            NSTATV, ' NPROPS=', NPROPS
        PNEWDT = 0.0d0
        return
    end if

    ! STATEV unpack + co-rotate tensor state variables.
    ! STRESS is already co-rotated by ABAQUS before UMAT entry (no ROTSIG needed).
    ! theta, beta, q are stress-like tensors stored in STATEV and must be
    ! co-rotated here so the return mapping operates in the rotated frame.
    ! LSTR=1 (stress): PLANE_STRESS_P stores the raw tensor shear, so no
    ! engineering-shear halving applies.
    do i = 1, 3
        theta_n(i) = STATEV(i)
        beta_n(i)  = STATEV(3 + i)
        q_n(i)     = STATEV(7 + i)
    end do
    Rbnd_n      = STATEV(7)
    rstag_n     = STATEV(11)
    eps_eq_n    = STATEV(12)
    theta_max_n = STATEV(13)

    call ROTSIG(theta_n, DROT, theta_rot, 1, NDI, NSHR)
    call ROTSIG(beta_n,  DROT, beta_rot,  1, NDI, NSHR)
    call ROTSIG(q_n,     DROT, q_rot,     1, NDI, NSHR)

    call yu_kinematic_ps( &
        PROPS(1), PROPS(2), PROPS(3), PROPS(4), PROPS(5), PROPS(6), &
        PROPS(7), PROPS(8), PROPS(9), PROPS(10), PROPS(11), PROPS(12), &
        STRESS, &
        theta_rot, beta_rot, Rbnd_n, q_rot, rstag_n, eps_eq_n, theta_max_n, &
        DSTRAN, &
        stress_out, &
        theta_out, beta_out, Rbnd_out, q_out, rstag_out, eps_eq_out, theta_max_out, &
        ddsdde_local, n_iter, converged, r_hist_val, fail_code, fail_diag)

    ! Non-convergence: set PNEWDT and return WITHOUT updating STRESS/STATEV, so
    ! the retry starts from the original state rather than a partially-converged
    ! one.  DDSDDE still gets the secant stiffness at eps_eq_n -- the state the
    ! retry will start from -- because the ABAQUS global NR needs a valid matrix.
    if (converged == 0) then
        PNEWDT = min(PNEWDT, 0.5d0)
        call yu_ps_elastic_stiffness( &
            PROPS(1), PROPS(2), eps_eq_n, PROPS(11), PROPS(12), DDSDDE)

        if (YU_DEBUG) then
            ! Summary of the PREVIOUS increment, flushed when the increment
            ! changes (grep "YU-NC"):
            !   YU-NC  kstep  kinc  time  dtime  n_fail  n_nr  n_mu  n_lu
            ! The final increment never gets a summary -- a UMAT has no
            ! end-of-analysis hook -- but its YU-FL lines are written live.
            if (KSTEP /= yu_kstep .or. KINC /= yu_kinc) then
                if (yu_kinc /= -1) then
                    write(7,'(A,2I6,2ES11.3,4I8)') 'YU-NC ', &
                        yu_kstep, yu_kinc, yu_time, yu_dtime, &
                        yu_nfail, yu_nnr, yu_nmu, yu_nlu
                end if
                yu_kstep = KSTEP
                yu_kinc  = KINC
                yu_time  = TIME(1)
                yu_dtime = DTIME
                yu_nfail = 0
                yu_nnr   = 0
                yu_nmu   = 0
                yu_nlu   = 0
                yu_ndet  = 0
                ! Increment context, written once: the first failing point of
                ! the increment, in full.
                write(7,'(A,4I6,ES11.3)') 'YU-DT ', KSTEP, KINC, NOEL, NPT, DTIME
                write(7,'(A,3ES22.14)') 'YU-DS ', (DSTRAN(i), i=1,3)
                write(7,'(A,3ES22.14)') 'YU-SS ', (STRESS(i), i=1,3)
                write(7,'(A,3ES22.14)') 'YU-TH ', (STATEV(i), i=1,3)
                write(7,'(A,3ES22.14)') 'YU-BT ', (STATEV(3+i), i=1,3)
                write(7,'(A,4ES22.14)') 'YU-RQ ', STATEV(7), STATEV(11), &
                    STATEV(12), STATEV(13)
                write(7,'(A,10ES10.3)') 'YU-RH ', (r_hist_val(i), i=1,10)
            end if

            yu_nfail = yu_nfail + 1
            if (fail_code == 1) then
                yu_nnr = yu_nnr + 1
            else if (fail_code == 2) then
                yu_nmu = yu_nmu + 1
            else
                yu_nlu = yu_nlu + 1
            end if

            ! Per-point detail, capped (grep "YU-FL"):
            !   YU-FL  code  noel  npt  n_iter  r_n  Fn  Gn  sqrt_arg  |dbeta|  dlambda
            ! code: 1=NR exhausted, 2=mu Newton, 3=linear solve.
            ! sqrt_arg = r_n^2 + 6*h*Fn < 0 means the mu equation has no real
            ! root, so no cutback can rescue the point -- the stagnation
            ! formulation itself is out of range there.
            if (yu_ndet < YU_MAX_DETAIL) then
                yu_ndet = yu_ndet + 1
                write(7,'(A,4I7,6ES13.5)') 'YU-FL ', &
                    fail_code, NOEL, NPT, n_iter, (fail_diag(i), i=1,6)
            end if
        end if

        return   ! leave STRESS and STATEV unchanged for the retry
    end if

    ! Write-back stress and tangent (converged == 1 only)
    do i = 1, NTENS
        STRESS(i) = stress_out(i)
        do j = 1, NTENS
            DDSDDE(i, j) = ddsdde_local(i, j)
        end do
    end do

    ! STATEV repack (converged == 1 only).
    ! theta_out, beta_out, q_out are already in the rotated frame (the return
    ! mapping ran there); no further rotation before storing.
    do i = 1, 3
        STATEV(i)     = theta_out(i)
        STATEV(3 + i) = beta_out(i)
        STATEV(7 + i) = q_out(i)
    end do
    STATEV(7)  = Rbnd_out
    STATEV(11) = rstag_out
    STATEV(12) = eps_eq_out
    STATEV(13) = theta_max_out

    ! Zero unused output fields
    SSE = 0.0d0; SPD = 0.0d0; SCD = 0.0d0; RPL = 0.0d0; DRPLDT = 0.0d0
    do i = 1, NTENS
        DDSDDT(i) = 0.0d0
        DRPLDE(i) = 0.0d0
    end do

end subroutine umat
