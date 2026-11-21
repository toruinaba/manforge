! =============================================================================
! J2 plasticity with Voce isotropic hardening -- 3D solid
!
! The port of mymodels.voce.model.J2Voce3D: the same radial return, the same
! scalar Newton iteration, the same closed-form tangent.
!
!   voce_3d   the constitutive update.  This is what is built with f2py and
!             compared with the Python model, step by step.
!   umat      the ABAQUS entry point.  It only unpacks PROPS / STATEV and calls
!             voce_3d.
!
! PROPS:  1 = E, 2 = nu, 3 = sigma_y0, 4 = Q, 5 = b      (J2Voce.param_names)
! STATEV: 1 = ep  (equivalent plastic strain)
!
! Voigt order [11, 22, 33, 12, 13, 23].  Stress carries the tensor shear
! components, strain the engineering shear (gamma = 2 * epsilon).
! =============================================================================

subroutine voce_3d(E, nu, sigma_y0, Q, b, &
                   stress_in, ep_in, dstran, &
                   stress_out, ep_out, ddsdde, converged)
    implicit none
    double precision, intent(in)  :: E, nu, sigma_y0, Q, b
    double precision, intent(in)  :: stress_in(6), ep_in, dstran(6)
    double precision, intent(out) :: stress_out(6), ep_out, ddsdde(6,6)
    integer,          intent(out) :: converged     ! 1 = converged, 0 = not

    ! The same limits as J2Voce3D.MAX_ITER and J2Voce3D.TOL.
    integer,          parameter :: MAX_ITER = 25
    double precision, parameter :: TOL = 1.0d-12

    double precision :: mu, lam, bulk, C(6,6), stress_trial(6), s(6), p_mean
    double precision :: q_trial, g, dlambda, sigma_y, hard, theta, beta
    integer          :: i, j, iter

    mu   = E / (2.0d0 * (1.0d0 + nu))
    lam  = E * nu / ((1.0d0 + nu) * (1.0d0 - 2.0d0 * nu))
    bulk = lam + 2.0d0 * mu / 3.0d0

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

    ! --- elastic predictor
    do i = 1, 6
        stress_trial(i) = stress_in(i)
        do j = 1, 6
            stress_trial(i) = stress_trial(i) + C(i,j) * dstran(j)
        end do
    end do

    p_mean = (stress_trial(1) + stress_trial(2) + stress_trial(3)) / 3.0d0
    s = stress_trial
    do i = 1, 3
        s(i) = s(i) - p_mean
    end do
    q_trial = sqrt(1.5d0 * (s(1)**2 + s(2)**2 + s(3)**2 &
                            + 2.0d0 * (s(4)**2 + s(5)**2 + s(6)**2)))

    converged = 1
    sigma_y = sigma_y0 + Q * (1.0d0 - exp(-b * ep_in))

    ! --- elastic step
    if (q_trial - sigma_y <= 0.0d0) then
        stress_out = stress_trial
        ep_out = ep_in
        ddsdde = C
        return
    end if

    ! --- radial return: g(dlambda) = q_trial - 3 mu dlambda - sigma_y(ep_n + dlambda) = 0
    ! g is convex and decreasing and g(0) > 0, so Newton from 0 approaches the
    ! root from below.
    dlambda = 0.0d0
    converged = 0
    do iter = 0, MAX_ITER
        sigma_y = sigma_y0 + Q * (1.0d0 - exp(-b * (ep_in + dlambda)))
        hard    = Q * b * exp(-b * (ep_in + dlambda))
        g = q_trial - 3.0d0 * mu * dlambda - sigma_y
        if (abs(g) < TOL * q_trial) then
            converged = 1
            exit
        end if
        dlambda = dlambda + g / (3.0d0 * mu + hard)
    end do

    do i = 1, 6
        stress_out(i) = stress_trial(i) - (3.0d0 * mu * dlambda / q_trial) * s(i)
    end do
    ep_out = ep_in + dlambda

    ! --- consistent tangent: I_vol C + theta I_dev C - beta s (x) s,
    ! with sigma_y and hard at the end of the step.
    sigma_y = sigma_y0 + Q * (1.0d0 - exp(-b * ep_out))
    hard    = Q * b * exp(-b * ep_out)
    theta = 1.0d0 - 3.0d0 * mu * dlambda / q_trial
    beta  = 9.0d0 * mu**2 * (sigma_y - hard * dlambda) &
            / ((3.0d0 * mu + hard) * q_trial**3)

    ddsdde = 0.0d0
    do i = 1, 3
        do j = 1, 3
            ddsdde(i,j) = bulk - theta * 2.0d0 * mu / 3.0d0
        end do
        ddsdde(i,i) = ddsdde(i,i) + theta * 2.0d0 * mu
    end do
    do i = 4, 6
        ddsdde(i,i) = theta * mu
    end do
    do i = 1, 6
        do j = 1, 6
            ddsdde(i,j) = ddsdde(i,j) - beta * s(i) * s(j)
        end do
    end do

end subroutine voce_3d


! -----------------------------------------------------------------------------
! umat -- ABAQUS entry point
! -----------------------------------------------------------------------------
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

    double precision :: stress_out(6), ep_out
    integer          :: i, converged

    call voce_3d(PROPS(1), PROPS(2), PROPS(3), PROPS(4), PROPS(5), &
                 STRESS, STATEV(1), DSTRAN, &
                 stress_out, ep_out, DDSDDE, converged)

    SSE = 0.0d0; SPD = 0.0d0; SCD = 0.0d0; RPL = 0.0d0; DRPLDT = 0.0d0
    do i = 1, NTENS
        DDSDDT(i) = 0.0d0
        DRPLDE(i) = 0.0d0
    end do

    if (converged /= 1) then
        ! Ask ABAQUS for a smaller increment; STRESS and STATEV are left as
        ! they were, since the iterate is not a solution.
        PNEWDT = 0.5d0
        return
    end if

    do i = 1, NTENS
        STRESS(i) = stress_out(i)
    end do
    STATEV(1) = ep_out

end subroutine umat
