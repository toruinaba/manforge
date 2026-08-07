! =============================================================================
! manforge -- timing harness for yu_kinematic_proj_ps
!
! Deliberately a SEPARATE file from yu_kinematic_proj_ps.f90: gfortran cannot
! see the callee bodies from here (no LTO), so it cannot hoist the calls out of
! the repeat loop or discard them as dead.  Merging this into the implementation
! file would let the optimiser delete the whole benchmark.
!
! Each routine returns wall time for n_repeat calls plus a `sink` value derived
! from the outputs, so nothing is provably unused.
! =============================================================================


! =============================================================================
! yu_projps_bench_full -- time the complete constitutive update (one UMAT call)
! =============================================================================
subroutine yu_projps_bench_full( &
        E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
        stress_n, theta_n, beta_n, Rbnd_n, q_n, rstag_n, eps_eq_n, theta_max_n, &
        dstran, n_repeat, elapsed, sink, n_iter_out)
    implicit none
    double precision, intent(in) :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in) :: stress_n(3)
    double precision, intent(in) :: theta_n(3), beta_n(3), Rbnd_n, q_n(3), rstag_n
    double precision, intent(in) :: eps_eq_n, theta_max_n
    double precision, intent(in) :: dstran(3)
    integer,          intent(in) :: n_repeat
    double precision, intent(out) :: elapsed, sink
    integer,          intent(out) :: n_iter_out

    double precision :: stress_out(3), theta_out(3), beta_out(3)
    double precision :: Rbnd_out, q_out(3), rstag_out, eps_eq_out, theta_max_out
    double precision :: ddsdde(3,3), r_hist(50), fail_diag(6)
    integer :: n_iter, converged, fail_code
    integer :: rep, ii
    integer(8) :: t0, t1, rate

    sink = 0.0d0
    n_iter_out = 0
    call system_clock(t0, rate)
    do rep = 1, n_repeat
        call yu_kinematic_proj_ps( &
            E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
            stress_n, theta_n, beta_n, Rbnd_n, q_n, rstag_n, eps_eq_n, theta_max_n, &
            dstran, &
            stress_out, theta_out, beta_out, Rbnd_out, q_out, rstag_out, &
            eps_eq_out, theta_max_out, ddsdde, &
            n_iter, converged, r_hist, fail_code, fail_diag)
        do ii = 1, 3
            sink = sink + stress_out(ii) + ddsdde(ii,ii)
        end do
        n_iter_out = n_iter
    end do
    call system_clock(t1, rate)
    elapsed = dble(t1 - t0) / dble(rate)

end subroutine yu_projps_bench_full


! =============================================================================
! yu_projps_bench_jac -- time Jacobian assembly alone
! =============================================================================
subroutine yu_projps_bench_jac( &
        E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
        stress_new, theta_new, beta_new, R_new, eps_eq_new, &
        theta_max_n, R_n, dlambda, n_repeat, elapsed, sink)
    implicit none
    double precision, intent(in) :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in) :: stress_new(3), theta_new(3), beta_new(3)
    double precision, intent(in) :: R_new, eps_eq_new, theta_max_n, R_n, dlambda
    integer,          intent(in) :: n_repeat
    double precision, intent(out) :: elapsed, sink

    double precision :: jac(10,10)
    integer :: rep, ii
    integer(8) :: t0, t1, rate

    sink = 0.0d0
    call system_clock(t0, rate)
    do rep = 1, n_repeat
        call yu_projps_calc_jacobian( &
            E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
            stress_new, theta_new, beta_new, R_new, eps_eq_new, &
            theta_max_n, R_n, dlambda, jac)
        do ii = 1, 10
            sink = sink + jac(ii,ii)
        end do
    end do
    call system_clock(t1, rate)
    elapsed = dble(t1 - t0) / dble(rate)

end subroutine yu_projps_bench_jac


! =============================================================================
! yu_projps_bench_resid -- time residual evaluation alone
! =============================================================================
subroutine yu_projps_bench_resid( &
        E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
        stress_new, theta_new, beta_new, R_new, eps_eq_new, &
        theta_n, beta_n, theta_max_n, stress_trial, dlambda, n_repeat, elapsed, sink)
    implicit none
    double precision, intent(in) :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in) :: stress_new(3), theta_new(3), beta_new(3)
    double precision, intent(in) :: R_new, eps_eq_new
    double precision, intent(in) :: theta_n(3), beta_n(3), theta_max_n, stress_trial(3), dlambda
    integer,          intent(in) :: n_repeat
    double precision, intent(out) :: elapsed, sink

    double precision :: r_vec(10)
    integer :: rep, ii
    integer(8) :: t0, t1, rate

    sink = 0.0d0
    call system_clock(t0, rate)
    do rep = 1, n_repeat
        call yu_projps_calc_residual( &
            E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
            stress_new, theta_new, beta_new, R_new, eps_eq_new, &
            theta_n, beta_n, theta_max_n, stress_trial, dlambda, r_vec)
        do ii = 1, 10
            sink = sink + r_vec(ii)
        end do
    end do
    call system_clock(t1, rate)
    elapsed = dble(t1 - t0) / dble(rate)

end subroutine yu_projps_bench_resid


! =============================================================================
! yu_projps_bench_ddsdde -- time consistent-tangent extraction alone
! =============================================================================
subroutine yu_projps_bench_ddsdde( &
        E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
        stress_new, theta_new, beta_new, R_new, eps_eq_new, &
        theta_max_n, R_n, dlambda, eps_eq_n, n_repeat, elapsed, sink)
    implicit none
    double precision, intent(in) :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in) :: stress_new(3), theta_new(3), beta_new(3)
    double precision, intent(in) :: R_new, eps_eq_new, theta_max_n, R_n, dlambda, eps_eq_n
    integer,          intent(in) :: n_repeat
    double precision, intent(out) :: elapsed, sink

    double precision :: ddsdde(3,3)
    integer :: rep, ii
    integer(8) :: t0, t1, rate

    sink = 0.0d0
    call system_clock(t0, rate)
    do rep = 1, n_repeat
        call yu_projps_calc_ddsdde( &
            E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
            stress_new, theta_new, beta_new, R_new, eps_eq_new, &
            theta_max_n, R_n, dlambda, eps_eq_n, ddsdde)
        do ii = 1, 3
            sink = sink + ddsdde(ii,ii)
        end do
    end do
    call system_clock(t1, rate)
    elapsed = dble(t1 - t0) / dble(rate)

end subroutine yu_projps_bench_ddsdde


! =============================================================================
! yu_projps_ddsdde_fast -- prototype of the improved consistent tangent
!
! Same result as yu_projps_calc_ddsdde, reached without the explicit C inverse.
! The existing routine premultiplies the R_stress rows by C_inv, i.e. it forms
! M @ J with M = blockdiag(C_inv, I_7), then inverts:
!
!   (M J)^-1 = J^-1 M^-1 = J^-1 @ blockdiag(C_n, I_7)
!
! whose leading 3x3 block is just  (J^-1)[1:3,1:3] @ C_n.  So the 3x3 inverse
! and the 3x10 row premultiply are both unnecessary, and only the first three
! COLUMNS of J^-1 are needed -- NRHS drops from 10 to 3.
!
! Lives here rather than in the implementation file so the claim can be timed
! and diffed against the current routine before anything is committed.
! =============================================================================
subroutine yu_projps_ddsdde_fast(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                                 stress_new, theta_new, beta_new, R_new, eps_eq_new, &
                                 theta_max_n, R_n, dlambda, eps_eq_n, ddsdde)
    implicit none
    double precision, intent(in)  :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in)  :: stress_new(3), theta_new(3), beta_new(3)
    double precision, intent(in)  :: R_new, eps_eq_new, theta_max_n, R_n, dlambda, eps_eq_n
    double precision, intent(out) :: ddsdde(3,3)

    double precision :: jac(10,10), C_n(3,3)

    call yu_projps_calc_jacobian(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                             stress_new, theta_new, beta_new, R_new, eps_eq_new, &
                             theta_max_n, R_n, dlambda, jac)
    call yu_projps_elastic_stiffness(E, nu, eps_eq_n, Ea, xi_param, C_n)
    call yu_projps_ddsdde_from_jac(jac, C_n, ddsdde)

end subroutine yu_projps_ddsdde_fast


! =============================================================================
! yu_projps_ddsdde_from_jac -- the linear-algebra core, jac consumed in place
! =============================================================================
subroutine yu_projps_ddsdde_from_jac(jac_in, C_n, ddsdde)
    implicit none
    double precision, intent(in)  :: jac_in(10,10), C_n(3,3)
    double precision, intent(out) :: ddsdde(3,3)

    double precision :: A(10,10), X(10,3)
    integer :: info
    integer :: ii, jj, kk

    do jj = 1, 10
        do ii = 1, 10
            A(ii,jj) = jac_in(ii,jj)
        end do
    end do
    do jj = 1, 3
        do ii = 1, 10
            X(ii,jj) = 0.0d0
        end do
        X(jj,jj) = 1.0d0
    end do

    ! yu_projps_solve, not dgesv: at 10x10 the hand-rolled elimination measures
    ! ~3x faster, so the prototype uses what the implementation will use.
    call yu_projps_solve(10, A, 10, X, 3, info)
    if (info /= 0) then
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

end subroutine yu_projps_ddsdde_from_jac


! =============================================================================
! yu_projps_bench_ddsdde_fast -- time the prototype, Jacobian recomputed
! =============================================================================
subroutine yu_projps_bench_ddsdde_fast( &
        E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
        stress_new, theta_new, beta_new, R_new, eps_eq_new, &
        theta_max_n, R_n, dlambda, eps_eq_n, n_repeat, elapsed, sink)
    implicit none
    double precision, intent(in) :: E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param
    double precision, intent(in) :: stress_new(3), theta_new(3), beta_new(3)
    double precision, intent(in) :: R_new, eps_eq_new, theta_max_n, R_n, dlambda, eps_eq_n
    integer,          intent(in) :: n_repeat
    double precision, intent(out) :: elapsed, sink

    double precision :: ddsdde(3,3)
    integer :: rep, ii
    integer(8) :: t0, t1, rate

    sink = 0.0d0
    call system_clock(t0, rate)
    do rep = 1, n_repeat
        call yu_projps_ddsdde_fast(E, nu, Y, B_bnd, C_1, C_2, Rsat, k, b_kin, h, Ea, xi_param, &
                                   stress_new, theta_new, beta_new, R_new, eps_eq_new, &
                                   theta_max_n, R_n, dlambda, eps_eq_n, ddsdde)
        do ii = 1, 3
            sink = sink + ddsdde(ii,ii)
        end do
    end do
    call system_clock(t1, rate)
    elapsed = dble(t1 - t0) / dble(rate)

end subroutine yu_projps_bench_ddsdde_fast


! =============================================================================
! yu_projps_bench_dgesv -- LAPACK dgesv on the same 10x10 system
! =============================================================================
subroutine yu_projps_bench_dgesv(A_in, nrhs, n_repeat, elapsed, sink)
    implicit none
    double precision, intent(in) :: A_in(10,10)
    integer,          intent(in) :: nrhs, n_repeat
    double precision, intent(out) :: elapsed, sink

    double precision :: A(10,10), b(10,10)
    integer :: ipiv(10), info
    integer :: rep, ii, jj
    integer(8) :: t0, t1, rate

    sink = 0.0d0
    call system_clock(t0, rate)
    do rep = 1, n_repeat
        do jj = 1, 10
            do ii = 1, 10
                A(ii,jj) = A_in(ii,jj)
                b(ii,jj) = 0.0d0
            end do
            b(jj,jj) = 1.0d0
        end do
        call dgesv(10, nrhs, A, 10, ipiv, b, 10, info)
        do ii = 1, 10
            sink = sink + b(ii,1)
        end do
    end do
    call system_clock(t1, rate)
    elapsed = dble(t1 - t0) / dble(rate)

end subroutine yu_projps_bench_dgesv


! =============================================================================
! yu_projps_bench_solve -- time the 10x10 LU solve alone (NRHS=1)
! =============================================================================
subroutine yu_projps_bench_solve(A_in, nrhs, n_repeat, elapsed, sink)
    implicit none
    double precision, intent(in) :: A_in(10,10)
    integer,          intent(in) :: nrhs, n_repeat
    double precision, intent(out) :: elapsed, sink

    double precision :: A(10,10), b(10,10)
    integer :: rep, ii, jj, info
    integer(8) :: t0, t1, rate

    sink = 0.0d0
    call system_clock(t0, rate)
    do rep = 1, n_repeat
        do jj = 1, 10
            do ii = 1, 10
                A(ii,jj) = A_in(ii,jj)
                b(ii,jj) = 0.0d0
            end do
            b(jj,jj) = 1.0d0
        end do
        call yu_projps_solve(10, A, 10, b, nrhs, info)
        do ii = 1, 10
            sink = sink + b(ii,1)
        end do
    end do
    call system_clock(t1, rate)
    elapsed = dble(t1 - t0) / dble(rate)

end subroutine yu_projps_bench_solve
