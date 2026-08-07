! =============================================================================
! manforge -- timing harness for yu_kinematic_3d
!
! Companion to yu_projps_bench.f90; same reasoning for keeping it in a separate
! compilation unit (the optimiser must not see the callee bodies, or it would
! hoist the calls out of the repeat loops).
!
! The 3-D system carries 19 unknowns against the plane-stress 10, and
! yu_calc_ddsdde inverts it with NRHS=19 while only the leading 6 columns are
! used -- so the right-hand-side reduction that barely registered in plane
! stress is measured here on its own terms rather than extrapolated.
! =============================================================================


! =============================================================================
! yu_3d_bench_full -- time the complete constitutive update (one UMAT call)
! =============================================================================
subroutine yu_3d_bench_full( &
        E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
        stress_n, theta_n, beta_n, Rbnd_n, q_n, rstag_n, eps_eq_n, theta_max_n, &
        dstran, n_repeat, elapsed, sink, n_iter_out)
    implicit none
    double precision, intent(in) :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in) :: stress_n(6)
    double precision, intent(in) :: theta_n(6), beta_n(6), Rbnd_n, q_n(6), rstag_n
    double precision, intent(in) :: eps_eq_n, theta_max_n
    double precision, intent(in) :: dstran(6)
    integer,          intent(in) :: n_repeat
    double precision, intent(out) :: elapsed, sink
    integer,          intent(out) :: n_iter_out

    double precision :: stress_out(6), theta_out(6), beta_out(6)
    double precision :: Rbnd_out, q_out(6), rstag_out, eps_eq_out, theta_max_out
    double precision :: ddsdde(6,6), r_hist(50), stag_vals(6), iter_dump(50,22)
    double precision :: xi_trial_norm_out
    integer :: n_iter, converged
    integer :: rep, ii
    integer(8) :: t0, t1, rate

    sink = 0.0d0
    n_iter_out = 0
    call system_clock(t0, rate)
    do rep = 1, n_repeat
        call yu_kinematic_3d( &
            E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
            stress_n, theta_n, beta_n, Rbnd_n, q_n, rstag_n, eps_eq_n, theta_max_n, &
            dstran, &
            stress_out, theta_out, beta_out, Rbnd_out, q_out, rstag_out, &
            eps_eq_out, theta_max_out, ddsdde, &
            n_iter, converged, xi_trial_norm_out, r_hist, stag_vals, iter_dump)
        do ii = 1, 6
            sink = sink + stress_out(ii) + ddsdde(ii,ii)
        end do
        n_iter_out = n_iter
    end do
    call system_clock(t1, rate)
    elapsed = dble(t1 - t0) / dble(rate)

end subroutine yu_3d_bench_full


! =============================================================================
! yu_3d_bench_jac -- time Jacobian assembly alone
! =============================================================================
subroutine yu_3d_bench_jac( &
        E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
        stress_new, theta_new, beta_new, R_new, eps_eq_new, &
        theta_max_new, R_n, dlambda, n_repeat, elapsed, sink)
    implicit none
    double precision, intent(in) :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in) :: stress_new(6), theta_new(6), beta_new(6)
    double precision, intent(in) :: R_new, eps_eq_new, theta_max_new, R_n, dlambda
    integer,          intent(in) :: n_repeat
    double precision, intent(out) :: elapsed, sink

    double precision :: jac(19,19)
    integer :: rep, ii
    integer(8) :: t0, t1, rate

    sink = 0.0d0
    call system_clock(t0, rate)
    do rep = 1, n_repeat
        call yu_calc_jacobian(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                              stress_new, theta_new, beta_new, R_new, eps_eq_new, &
                              theta_max_new, R_n, dlambda, jac)
        do ii = 1, 19
            sink = sink + jac(ii,ii)
        end do
    end do
    call system_clock(t1, rate)
    elapsed = dble(t1 - t0) / dble(rate)

end subroutine yu_3d_bench_jac


! =============================================================================
! yu_3d_bench_resid -- time residual evaluation alone
! =============================================================================
subroutine yu_3d_bench_resid( &
        E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
        stress_new, theta_new, beta_new, R_new, eps_eq_new, &
        theta_n, beta_n, theta_max_n, stress_trial, dlambda, n_repeat, elapsed, sink)
    implicit none
    double precision, intent(in) :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in) :: stress_new(6), theta_new(6), beta_new(6)
    double precision, intent(in) :: R_new, eps_eq_new
    double precision, intent(in) :: theta_n(6), beta_n(6), theta_max_n, stress_trial(6), dlambda
    integer,          intent(in) :: n_repeat
    double precision, intent(out) :: elapsed, sink

    double precision :: r_vec(19)
    integer :: rep, ii
    integer(8) :: t0, t1, rate

    sink = 0.0d0
    call system_clock(t0, rate)
    do rep = 1, n_repeat
        call yu_calc_residual(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                              stress_new, theta_new, beta_new, R_new, eps_eq_new, &
                              theta_n, beta_n, theta_max_n, stress_trial, dlambda, r_vec)
        do ii = 1, 19
            sink = sink + r_vec(ii)
        end do
    end do
    call system_clock(t1, rate)
    elapsed = dble(t1 - t0) / dble(rate)

end subroutine yu_3d_bench_resid


! =============================================================================
! yu_3d_bench_ddsdde -- time the existing consistent tangent
! =============================================================================
subroutine yu_3d_bench_ddsdde( &
        E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
        stress_new, theta_new, beta_new, R_new, eps_eq_new, &
        theta_max_new, R_n, dlambda, eps_eq_n, n_repeat, elapsed, sink)
    implicit none
    double precision, intent(in) :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in) :: stress_new(6), theta_new(6), beta_new(6)
    double precision, intent(in) :: R_new, eps_eq_new, theta_max_new, R_n, dlambda, eps_eq_n
    integer,          intent(in) :: n_repeat
    double precision, intent(out) :: elapsed, sink

    double precision :: ddsdde(6,6)
    integer :: rep, ii
    integer(8) :: t0, t1, rate

    sink = 0.0d0
    call system_clock(t0, rate)
    do rep = 1, n_repeat
        call yu_calc_ddsdde(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                            stress_new, theta_new, beta_new, R_new, eps_eq_new, &
                            theta_max_new, R_n, dlambda, ddsdde, eps_eq_n)
        do ii = 1, 6
            sink = sink + ddsdde(ii,ii)
        end do
    end do
    call system_clock(t1, rate)
    elapsed = dble(t1 - t0) / dble(rate)

end subroutine yu_3d_bench_ddsdde


! =============================================================================
! yu_3d_ddsdde_fast -- prototype: no explicit C inverse, NRHS = 6
!
! The existing routine forms M @ J with M = blockdiag(C_n^-1, I_13) and inverts
! that.  Since (M J)^-1 = J^-1 @ blockdiag(C_n, I_13), the leading 6x6 block is
! (J^-1)[1:6,1:6] @ C_n -- so the 6x6 inverse and the 6x19 row premultiply are
! both redundant, and only the first six COLUMNS of J^-1 are needed.
! =============================================================================
subroutine yu_3d_ddsdde_fast(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                             stress_new, theta_new, beta_new, R_new, eps_eq_new, &
                             theta_max_new, R_n, dlambda, eps_eq_n, ddsdde)
    implicit none
    double precision, intent(in)  :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in)  :: stress_new(6), theta_new(6), beta_new(6)
    double precision, intent(in)  :: R_new, eps_eq_new, theta_max_new, R_n, dlambda, eps_eq_n
    double precision, intent(out) :: ddsdde(6,6)

    double precision :: jac(19,19), C_n(6,6)

    call yu_calc_jacobian(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                          stress_new, theta_new, beta_new, R_new, eps_eq_new, &
                          theta_max_new, R_n, dlambda, jac)
    call yu_kinematic_3d_elastic_stiffness(E, nu, eps_eq_n, Ea, xi_param, C_n)
    call yu_3d_ddsdde_from_jac(jac, C_n, ddsdde)

end subroutine yu_3d_ddsdde_fast


! =============================================================================
! yu_3d_ddsdde_from_jac -- the linear-algebra core
! =============================================================================
subroutine yu_3d_ddsdde_from_jac(jac_in, C_n, ddsdde)
    implicit none
    double precision, intent(in)  :: jac_in(19,19), C_n(6,6)
    double precision, intent(out) :: ddsdde(6,6)

    double precision :: A(19,19), X(19,6)
    integer :: ipiv(19), info
    integer :: ii, jj, kk

    do jj = 1, 19
        do ii = 1, 19
            A(ii,jj) = jac_in(ii,jj)
        end do
    end do
    do jj = 1, 6
        do ii = 1, 19
            X(ii,jj) = 0.0d0
        end do
        X(jj,jj) = 1.0d0
    end do

    call dgesv(19, 6, A, 19, ipiv, X, 19, info)
    if (info /= 0) then
        do jj = 1, 6
            do ii = 1, 6
                ddsdde(ii,jj) = C_n(ii,jj)
            end do
        end do
        return
    end if

    do jj = 1, 6
        do ii = 1, 6
            ddsdde(ii,jj) = 0.0d0
            do kk = 1, 6
                ddsdde(ii,jj) = ddsdde(ii,jj) + X(ii,kk) * C_n(kk,jj)
            end do
        end do
    end do

end subroutine yu_3d_ddsdde_from_jac


! =============================================================================
! yu_3d_bench_ddsdde_fast -- time the prototype (Jacobian rebuilt)
! =============================================================================
subroutine yu_3d_bench_ddsdde_fast( &
        E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
        stress_new, theta_new, beta_new, R_new, eps_eq_new, &
        theta_max_new, R_n, dlambda, eps_eq_n, n_repeat, elapsed, sink)
    implicit none
    double precision, intent(in) :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in) :: stress_new(6), theta_new(6), beta_new(6)
    double precision, intent(in) :: R_new, eps_eq_new, theta_max_new, R_n, dlambda, eps_eq_n
    integer,          intent(in) :: n_repeat
    double precision, intent(out) :: elapsed, sink

    double precision :: ddsdde(6,6)
    integer :: rep, ii
    integer(8) :: t0, t1, rate

    sink = 0.0d0
    call system_clock(t0, rate)
    do rep = 1, n_repeat
        call yu_3d_ddsdde_fast(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                               stress_new, theta_new, beta_new, R_new, eps_eq_new, &
                               theta_max_new, R_n, dlambda, eps_eq_n, ddsdde)
        do ii = 1, 6
            sink = sink + ddsdde(ii,ii)
        end do
    end do
    call system_clock(t1, rate)
    elapsed = dble(t1 - t0) / dble(rate)

end subroutine yu_3d_bench_ddsdde_fast


! =============================================================================
! yu_3d_bench_dgesv -- dgesv on the 19x19 system at a chosen NRHS
! =============================================================================
subroutine yu_3d_bench_dgesv(A_in, nrhs, n_repeat, elapsed, sink)
    implicit none
    double precision, intent(in) :: A_in(19,19)
    integer,          intent(in) :: nrhs, n_repeat
    double precision, intent(out) :: elapsed, sink

    double precision :: A(19,19), b(19,19)
    integer :: ipiv(19), info
    integer :: rep, ii, jj
    integer(8) :: t0, t1, rate

    sink = 0.0d0
    call system_clock(t0, rate)
    do rep = 1, n_repeat
        do jj = 1, 19
            do ii = 1, 19
                A(ii,jj) = A_in(ii,jj)
                b(ii,jj) = 0.0d0
            end do
            b(jj,jj) = 1.0d0
        end do
        call dgesv(19, nrhs, A, 19, ipiv, b, 19, info)
        do ii = 1, 19
            sink = sink + b(ii,1)
        end do
    end do
    call system_clock(t1, rate)
    elapsed = dble(t1 - t0) / dble(rate)

end subroutine yu_3d_bench_dgesv
