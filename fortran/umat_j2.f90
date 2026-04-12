! =============================================================================
! manforge -- J2 isotropic hardening UMAT (standalone, no ABAQUS solver)
!
! Implements the full ABAQUS UMAT interface for J2 plasticity with linear
! isotropic hardening.  Algorithm is identical to the Python reference:
!   - return_mapping  (src/manforge/core/return_mapping.py)
!   - consistent_tangent (src/manforge/core/tangent.py)
!
! Material properties (PROPS):
!   PROPS(1) = E        Young's modulus
!   PROPS(2) = nu       Poisson's ratio
!   PROPS(3) = sigma_y0 initial yield stress
!   PROPS(4) = H        linear isotropic hardening modulus
!
! State variables (STATEV):
!   STATEV(1) = ep      equivalent plastic strain
!
! Voigt convention: [s11, s22, s33, s12, s13, s23]
!   Normal components: indices 1-3
!   Shear components:  indices 4-6 (engineering shear = 2 * tensor shear)
!
! Flow direction (critical -- see derivation in docs):
!   n(1:3) = (3/2) * s(i) / sigma_vm   (normal)
!   n(4:6) = 3     * s(i) / sigma_vm   (shear -- factor 2 from Mandel norm)
!
! Consistent tangent: 7x7 linear system solved by internal LU solver
!   [ I + dl*C*H_f   C*n ] [dstress/de]   [C]
!   [ n^T            -H  ] [ddl/de    ] = [0]
!
! Build with f2py (meson backend, no external LAPACK needed):
!   python -m numpy.f2py -c abaqus_stubs.f90 umat_j2.f90 -m manforge_umat
! =============================================================================


! -----------------------------------------------------------------------------
! solve7 -- in-place LU solver for a 7x7 system with NRHS right-hand sides
!
! Solves A*X = B using Gaussian elimination with partial pivoting.
! B is overwritten with the solution X.
! Returns info=0 on success, info=k if pivot k is essentially zero.
!
! Parameters
! ----------
! A(7,7)    [inout] : coefficient matrix (overwritten with LU)
! B(7,NRHS) [inout] : right-hand sides (overwritten with solution)
! NRHS      [in]    : number of right-hand sides
! info      [out]   : 0 = success, k = singular at pivot k
! -----------------------------------------------------------------------------
subroutine solve7(A, B, NRHS, info)
    implicit none
    integer,          intent(in)    :: NRHS
    double precision, intent(inout) :: A(7,7), B(7,NRHS)
    integer,          intent(out)   :: info

    integer          :: i, j, k, piv
    double precision :: maxval_loc, tmp_d, factor
    double precision, parameter :: EPS = 1.0d-14

    info = 0

    do k = 1, 7
        ! -- find pivot row
        maxval_loc = abs(A(k,k))
        piv = k
        do i = k+1, 7
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
            do j = 1, 7
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
        do i = k+1, 7
            factor = A(i,k) / A(k,k)
            do j = k, 7
                A(i,j) = A(i,j) - factor * A(k,j)
            end do
            do j = 1, NRHS
                B(i,j) = B(i,j) - factor * B(k,j)
            end do
        end do
    end do

    ! -- back substitution
    do k = 7, 1, -1
        do j = 1, NRHS
            do i = k+1, 7
                B(k,j) = B(k,j) - A(k,i) * B(i,j)
            end do
            B(k,j) = B(k,j) / A(k,k)
        end do
    end do

end subroutine solve7


! -----------------------------------------------------------------------------
! umat_j2_run -- f2py-callable wrapper for the J2 UMAT
!
! Simplified interface (no ABAQUS boilerplate) designed for f2py and testing.
!
! Parameters
! ----------
! E, nu, sigma_y0, H [in]  : material parameters
! stress_in(6)       [in]  : stress at start of increment
! ep_in              [in]  : equivalent plastic strain at start
! dstran(6)          [in]  : strain increment (engineering shear)
! stress_out(6)      [out] : updated stress
! ep_out             [out] : updated equivalent plastic strain
! ddsdde(6,6)        [out] : consistent algorithmic tangent
! -----------------------------------------------------------------------------
subroutine umat_j2_run(E, nu, sigma_y0, H, &
                        stress_in, ep_in, dstran, &
                        stress_out, ep_out, ddsdde)
    implicit none
    double precision, intent(in)  :: E, nu, sigma_y0, H
    double precision, intent(in)  :: stress_in(6), ep_in, dstran(6)
    double precision, intent(out) :: stress_out(6), ep_out, ddsdde(6,6)

    ! -- local variables
    double precision :: mu, lam, C(6,6), stress_trial(6), s(6), p_mean
    double precision :: sigma_vm, sigma_vm_new, sss, f, dlambda
    double precision :: n_flow(6), Cn(6), P_ij, M(6)
    double precision :: H_f(6,6), A_mat(7,7), rhs(7,6)
    integer          :: i, j, k, info_lu

    ! -- elastic moduli
    mu  = E / (2.0d0 * (1.0d0 + nu))
    lam = E * nu / ((1.0d0 + nu) * (1.0d0 - 2.0d0 * nu))

    ! -- elastic stiffness C (6x6 Voigt)
    C = 0.0d0
    do i = 1, 3
        do j = 1, 3
            C(i,j) = lam
        end do
        C(i,i) = lam + 2.0d0 * mu
    end do
    do i = 4, 6
        C(i,i) = mu
    end do

    ! -- trial stress: stress_trial = stress_in + C @ dstran
    do i = 1, 6
        stress_trial(i) = stress_in(i)
        do j = 1, 6
            stress_trial(i) = stress_trial(i) + C(i,j) * dstran(j)
        end do
    end do

    ! -- deviatoric stress of trial state
    p_mean = (stress_trial(1) + stress_trial(2) + stress_trial(3)) / 3.0d0
    do i = 1, 6
        s(i) = stress_trial(i)
    end do
    do i = 1, 3
        s(i) = s(i) - p_mean
    end do

    ! -- von Mises: sqrt(1.5 * (s11^2+s22^2+s33^2 + 2*(s12^2+s13^2+s23^2)))
    sss = 0.0d0
    do i = 1, 3
        sss = sss + s(i)**2
    end do
    do i = 4, 6
        sss = sss + 2.0d0 * s(i)**2
    end do
    sigma_vm = sqrt(1.5d0 * sss)

    ! -- yield function at trial state
    f = sigma_vm - (sigma_y0 + H * ep_in)

    if (f <= 0.0d0 .or. sigma_vm < 1.0d-14) then
        ! ----------------------------------------------------------------
        ! Elastic step: accept trial stress, return elastic tangent
        ! ----------------------------------------------------------------
        do i = 1, 6
            stress_out(i) = stress_trial(i)
        end do
        ep_out = ep_in
        do i = 1, 6
            do j = 1, 6
                ddsdde(i,j) = C(i,j)
            end do
        end do

    else
        ! ----------------------------------------------------------------
        ! Plastic step: closed-form radial return (J2 + linear hardening)
        ! ----------------------------------------------------------------

        ! Delta-lambda = f / (3*mu + H)
        dlambda = f / (3.0d0 * mu + H)

        ! von Mises at converged stress (sigma_vm decreases by 3*mu*dlambda)
        ! H_f must be evaluated at converged state, not trial state
        sigma_vm_new = sigma_vm - 3.0d0 * mu * dlambda

        ! -- flow direction n
        ! n(1:3) = (3/2) * s(i) / sigma_vm    (normal components)
        ! n(4:6) = 3     * s(i) / sigma_vm    (shear -- x2 from Mandel norm)
        do i = 1, 3
            n_flow(i) = 1.5d0 * s(i) / sigma_vm
        end do
        do i = 4, 6
            n_flow(i) = 3.0d0 * s(i) / sigma_vm
        end do

        ! -- C @ n_flow
        do i = 1, 6
            Cn(i) = 0.0d0
            do j = 1, 6
                Cn(i) = Cn(i) + C(i,j) * n_flow(j)
            end do
        end do

        ! -- updated stress: sigma = sigma_trial - dlambda * C * n
        do i = 1, 6
            stress_out(i) = stress_trial(i) - dlambda * Cn(i)
        end do

        ! -- updated equivalent plastic strain
        ep_out = ep_in + dlambda

        ! ----------------------------------------------------------------
        ! Consistent tangent via 7x7 linear system
        !
        ! [ I + dl*C*H_f   C*n ] [dsigma/de]   [C]
        ! [ n^T             -H  ] [ddl/de   ] = [0]
        !
        ! H_f_ij = M_i * (3/2) * P_ij / sigma_vm - n_i * n_j / sigma_vm
        ! M_i = 1 for i=1..3 (normal), 2 for i=4..6 (shear)
        ! P_ij = deviatoric projector:
        !   i,j <= 3: delta_ij - 1/3
        !   cross normal-shear: 0
        !   i,j >  3: delta_ij
        ! ----------------------------------------------------------------

        ! -- Mandel factors M
        do i = 1, 3
            M(i) = 1.0d0
        end do
        do i = 4, 6
            M(i) = 2.0d0
        end do

        ! -- Hessian of yield function H_f (6x6)
        do i = 1, 6
            do j = 1, 6
                P_ij = 0.0d0
                if (i == j)                   P_ij = 1.0d0
                if (i <= 3 .and. j <= 3) P_ij = P_ij - 1.0d0/3.0d0
                H_f(i,j) = M(i) * 1.5d0 * P_ij / sigma_vm_new &
                          - n_flow(i) * n_flow(j) / sigma_vm_new
            end do
        end do

        ! -- Assemble 7x7 matrix A_mat
        !    (1:6, 1:6) = I + dlambda * C * H_f
        !    (1:6, 7)   = Cn
        !    (7,   1:6) = n_flow
        !    (7,   7)   = -H

        do i = 1, 6
            do j = 1, 6
                A_mat(i,j) = 0.0d0
                if (i == j) A_mat(i,j) = 1.0d0
                do k = 1, 6
                    A_mat(i,j) = A_mat(i,j) + dlambda * C(i,k) * H_f(k,j)
                end do
            end do
        end do

        do i = 1, 6
            A_mat(i,7) = Cn(i)
        end do
        do j = 1, 6
            A_mat(7,j) = n_flow(j)
        end do
        A_mat(7,7) = -H

        ! -- Assemble RHS (7 x 6): [C; zeros]
        do i = 1, 6
            do j = 1, 6
                rhs(i,j) = C(i,j)
            end do
        end do
        do j = 1, 6
            rhs(7,j) = 0.0d0
        end do

        ! -- Solve 7x7 system in-place (A_mat overwritten, rhs -> solution)
        call solve7(A_mat, rhs, 6, info_lu)

        if (info_lu /= 0) then
            ! Degenerate: fall back to elastic tangent
            do i = 1, 6
                do j = 1, 6
                    ddsdde(i,j) = C(i,j)
                end do
            end do
        else
            ! DDSDDE = upper-left 6x6 block of solution
            do i = 1, 6
                do j = 1, 6
                    ddsdde(i,j) = rhs(i,j)
                end do
            end do
        end if

    end if

end subroutine umat_j2_run


! -----------------------------------------------------------------------------
! umat_j2 -- full ABAQUS UMAT interface
!
! Delegates all logic to umat_j2_run; implements the standard ABAQUS signature
! for reference and potential solver integration.
! -----------------------------------------------------------------------------
subroutine umat_j2(STRESS, STATEV, DDSDDE, SSE, SPD, SCD, &
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

    double precision :: stress_out(6), ep_out
    integer          :: i

    call umat_j2_run(PROPS(1), PROPS(2), PROPS(3), PROPS(4), &
                     STRESS, STATEV(1), DSTRAN, &
                     stress_out, ep_out, DDSDDE)

    do i = 1, NTENS
        STRESS(i) = stress_out(i)
    end do
    STATEV(1) = ep_out

    ! -- zero unused output fields
    SSE = 0.0d0; SPD = 0.0d0; SCD = 0.0d0; RPL = 0.0d0; DRPLDT = 0.0d0
    do i = 1, NTENS
        DDSDDT(i) = 0.0d0
        DRPLDE(i) = 0.0d0
    end do

end subroutine umat_j2


! -----------------------------------------------------------------------------
! umat_j2_elastic_stiffness -- return the isotropic elastic stiffness C (6x6)
!
! Exposes the elastic stiffness calculation from umat_j2_run as a standalone
! f2py-callable subroutine.  Useful for component-level cross-validation:
! compare this output against J2Isotropic3D.elastic_stiffness(params) in Python
! to confirm that the elastic part of the Fortran implementation is correct
! before debugging the plastic return-mapping logic.
!
! Parameters
! ----------
! E      [in]  : Young's modulus
! nu     [in]  : Poisson's ratio
! C      [out] : 6x6 Voigt stiffness tensor (Fortran column-major order)
! -----------------------------------------------------------------------------
subroutine umat_j2_elastic_stiffness(E, nu, C)
    implicit none
    double precision, intent(in)  :: E, nu
    double precision, intent(out) :: C(6,6)

    double precision :: lam, mu
    integer :: i, j

    mu  = E / (2.0d0 * (1.0d0 + nu))
    lam = E * nu / ((1.0d0 + nu) * (1.0d0 - 2.0d0 * nu))

    do j = 1, 6
        do i = 1, 6
            C(i,j) = 0.0d0
        end do
    end do

    do i = 1, 3
        do j = 1, 3
            C(i,j) = lam
        end do
        C(i,i) = lam + 2.0d0 * mu
    end do

    do i = 4, 6
        C(i,i) = mu
    end do

end subroutine umat_j2_elastic_stiffness
