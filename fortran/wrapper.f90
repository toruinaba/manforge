! =============================================================================
! manforge — Fortran UMAT wrapper template
!
! This file provides:
!   1. A standard ABAQUS UMAT subroutine skeleton (umat_j2)
!   2. An ISO_C_BINDING wrapper (umat_j2_c) callable from Python via
!      f2py or ctypes
!
! To compile with f2py (see README.md for full instructions):
!   python -m numpy.f2py -c wrapper.f90 -m manforge_umat
!
! =============================================================================

! -----------------------------------------------------------------------------
! Helper: isotropic elastic stiffness in Voigt notation
!
! C[i,j] is stored as the (ntens x ntens) array DDSDDE.
! Convention: Voigt stress/strain with engineering shear (gamma = 2*eps_shear).
!   sigma = [s11, s22, s33, s12, s13, s23]
!   C = lambda * (delta_ij delta_kl) + mu * (delta_ik delta_jl + delta_il delta_jk)
!
! In Voigt notation (3 normal + 3 shear):
!   C_normal-normal block  = lambda * I + 2*mu * diag
!   C_normal-shear block   = 0
!   C_shear-shear block    = mu * I   (factor of 2 from engineering shear)
! -----------------------------------------------------------------------------
subroutine elastic_stiffness(E, nu, NTENS, DDSDDE)
    implicit none
    double precision, intent(in)  :: E, nu
    integer,          intent(in)  :: NTENS
    double precision, intent(out) :: DDSDDE(NTENS, NTENS)

    double precision :: lam, mu
    integer :: i, j

    mu  = E / (2.0d0 * (1.0d0 + nu))
    lam = E * nu / ((1.0d0 + nu) * (1.0d0 - 2.0d0 * nu))

    ! Initialise
    do j = 1, NTENS
        do i = 1, NTENS
            DDSDDE(i, j) = 0.0d0
        end do
    end do

    ! Normal-normal block (indices 1-3)
    do i = 1, 3
        do j = 1, 3
            DDSDDE(i, j) = lam
        end do
        DDSDDE(i, i) = lam + 2.0d0 * mu
    end do

    ! Shear-shear block (indices 4-NTENS); factor of 1 (engineering shear)
    do i = 4, NTENS
        DDSDDE(i, i) = mu
    end do
end subroutine elastic_stiffness


! -----------------------------------------------------------------------------
! UMAT skeleton — J2 isotropic hardening (to be completed)
!
! This subroutine follows the ABAQUS UMAT interface exactly.
! The current implementation contains only an elastic predictor as a
! placeholder.  The full return-mapping loop is TODO.
!
! Parameters
! ----------
! STRESS(NTENS)   : stress at start; updated to stress at end on exit
! STATEV(NSTATV)  : state variables; STATEV(1) = equivalent plastic strain ep
! DDSDDE(NTENS,NTENS) : consistent tangent on exit
! SSE, SPD, SCD   : specific elastic strain energy, plastic dissipation, creep
! STRAN(NTENS)    : total strain at start of increment
! DSTRAN(NTENS)   : strain increment
! PROPS(NPROPS)   : material constants [E, nu, sigma_y0, H]
!                   PROPS(1)=E, PROPS(2)=nu, PROPS(3)=sigma_y0, PROPS(4)=H
! -----------------------------------------------------------------------------
subroutine umat_j2(STRESS, STATEV, DDSDDE, SSE, SPD, SCD,        &
                   RPL, DDSDDT, DRPLDE, DRPLDT,                   &
                   STRAN, DSTRAN, TIME, DTIME,                    &
                   TEMP, DTEMP, PREDEF, DPRED,                    &
                   CMNAME, NDI, NSHR, NTENS, NSTATV,              &
                   PROPS, NPROPS, COORDS, DROT, PNEWDT,           &
                   CELENT, DFGRD0, DFGRD1,                        &
                   NOEL, NPT, LAYER, KSPT, JSTEP, KINC)
    implicit none

    ! --- Arguments (ABAQUS UMAT interface) ---
    character(len=80), intent(in)    :: CMNAME
    integer,           intent(in)    :: NDI, NSHR, NTENS, NSTATV, NPROPS
    integer,           intent(in)    :: NOEL, NPT, LAYER, KSPT, KINC
    integer,           intent(in)    :: JSTEP(4)
    double precision,  intent(inout) :: STRESS(NTENS)
    double precision,  intent(inout) :: STATEV(NSTATV)
    double precision,  intent(out)   :: DDSDDE(NTENS, NTENS)
    double precision,  intent(out)   :: SSE, SPD, SCD
    double precision,  intent(out)   :: RPL
    double precision,  intent(out)   :: DDSDDT(NTENS)
    double precision,  intent(out)   :: DRPLDE(NTENS)
    double precision,  intent(out)   :: DRPLDT
    double precision,  intent(in)    :: STRAN(NTENS), DSTRAN(NTENS)
    double precision,  intent(in)    :: TIME(2), DTIME
    double precision,  intent(in)    :: TEMP, DTEMP
    double precision,  intent(in)    :: PREDEF(1), DPRED(1)
    double precision,  intent(in)    :: PROPS(NPROPS)
    double precision,  intent(in)    :: COORDS(3)
    double precision,  intent(in)    :: DROT(3, 3)
    double precision,  intent(inout) :: PNEWDT
    double precision,  intent(in)    :: CELENT
    double precision,  intent(in)    :: DFGRD0(3, 3), DFGRD1(3, 3)

    ! --- Local variables ---
    double precision :: E, nu, sigma_y0, H
    double precision :: stress_trial(NTENS)
    integer          :: i

    ! --- Extract material constants ---
    E        = PROPS(1)
    nu       = PROPS(2)
    sigma_y0 = PROPS(3)
    H        = PROPS(4)

    ! --- Elastic stiffness ---
    call elastic_stiffness(E, nu, NTENS, DDSDDE)

    ! --- Elastic predictor (trial stress) ---
    ! TODO: Replace this loop with the full matrix-vector product
    !       stress_trial = STRESS + DDSDDE @ DSTRAN
    !       For now, copy the initial stress as a placeholder.
    do i = 1, NTENS
        stress_trial(i) = STRESS(i)
    end do

    ! TODO: Compute elastic trial stress (uncommment when implementing):
    !   integer :: j
    !   do i = 1, NTENS
    !       stress_trial(i) = STRESS(i)
    !       do j = 1, NTENS
    !           stress_trial(i) = stress_trial(i) + DDSDDE(i, j) * DSTRAN(j)
    !       end do
    !   end do
    !
    ! TODO: Evaluate yield function
    !       f_trial = vonmises(stress_trial) - (sigma_y0 + H * STATEV(1))
    !
    ! TODO: If f_trial > 0 (plastic), perform radial return mapping:
    !       1. Newton-Raphson on delta_lambda
    !       2. Update STRESS, STATEV(1) (equivalent plastic strain)
    !       3. Compute consistent tangent (see manforge Python implementation
    !          in src/manforge/core/tangent.py for the algorithm)
    !
    ! For now: elastic update only (placeholder)
    do i = 1, NTENS
        STRESS(i) = STRESS(i)   ! no-op placeholder
    end do

    SSE   = 0.0d0
    SPD   = 0.0d0
    SCD   = 0.0d0
    RPL   = 0.0d0
    DRPLDT = 0.0d0
    do i = 1, NTENS
        DDSDDT(i) = 0.0d0
        DRPLDE(i) = 0.0d0
    end do
end subroutine umat_j2


! -----------------------------------------------------------------------------
! ISO_C_BINDING wrapper — exposes umat_j2 to ctypes / f2py with C linkage
!
! Argument layout (flat C arrays, no assumed-shape):
!   stress(ntens), statev(nstatv), ddsdde(ntens*ntens),
!   stran(ntens), dstran(ntens), props(nprops)
! -----------------------------------------------------------------------------
subroutine umat_j2_c(stress, statev, ddsdde,                      &
                     sse, spd, scd,                               &
                     stran, dstran, dtime,                        &
                     props, nprops, ntens, nstatv) bind(C)
    use iso_c_binding, only: c_double, c_int
    implicit none

    integer(c_int),    intent(in),    value :: ntens, nstatv, nprops
    real(c_double),    intent(inout)        :: stress(ntens)
    real(c_double),    intent(inout)        :: statev(nstatv)
    real(c_double),    intent(out)          :: ddsdde(ntens, ntens)
    real(c_double),    intent(out)          :: sse, spd, scd
    real(c_double),    intent(in)           :: stran(ntens), dstran(ntens)
    real(c_double),    intent(in),    value :: dtime
    real(c_double),    intent(in)           :: props(nprops)

    ! Dummy ABAQUS arguments not needed for standalone use
    double precision :: rpl, ddsddt(ntens), drplde(ntens), drpldt
    double precision :: time(2), temp, dtemp, predef(1), dpred(1)
    double precision :: coords(3), drot(3,3), pnewdt, celent
    double precision :: dfgrd0(3,3), dfgrd1(3,3)
    integer          :: ndi, nshr, noel, npt, layer, kspt, jstep(4), kinc
    character(len=80) :: cmname

    ndi    = 3
    nshr   = ntens - 3
    time   = 0.0d0
    temp   = 0.0d0
    dtemp  = 0.0d0
    pnewdt = 1.0d0
    cmname = "UMAT_J2"
    noel   = 1;  npt  = 1;  layer = 1;  kspt = 1
    jstep  = 0;  kinc = 1

    call umat_j2(stress, statev, ddsdde, sse, spd, scd,           &
                 rpl, ddsddt, drplde, drpldt,                     &
                 stran, dstran, time, dtime,                      &
                 temp, dtemp, predef, dpred,                      &
                 cmname, ndi, nshr, ntens, nstatv,                &
                 props, nprops, coords, drot, pnewdt,             &
                 celent, dfgrd0, dfgrd1,                          &
                 noel, npt, layer, kspt, jstep, kinc)
end subroutine umat_j2_c
