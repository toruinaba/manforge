! =============================================================================
! manforge -- basic Fortran subroutine for f2py smoke test (Step 9)
!
! Provides a single subroutine: elastic_stress
!   Computes stress = C @ dstran for an isotropic linear elastic material.
!   Used to verify that f2py compilation and Python interop work correctly
!   before implementing the full UMAT in Step 10.
!
! Compile:
!   python -m numpy.f2py -c test_basic.f90 -m manforge_test_basic
! =============================================================================

! -----------------------------------------------------------------------------
! elastic_stress -- isotropic elastic stress increment
!
! Computes stress(ntens) = C(ntens,ntens) @ dstran(ntens)
! where C is the isotropic elastic stiffness in Voigt notation with
! engineering shear convention (gamma_ij = 2 * eps_ij).
!
! Parameters
! ----------
! E      [in]  : Young's modulus
! nu     [in]  : Poisson's ratio
! dstran [in]  : strain increment, shape (ntens,)
! stress [out] : stress increment, shape (ntens,)
! ntens  [in]  : number of stress/strain components (6 for 3D)
! -----------------------------------------------------------------------------
subroutine elastic_stress(E, nu, dstran, stress, ntens)
    implicit none
    integer,          intent(in)  :: ntens
    double precision, intent(in)  :: E, nu
    double precision, intent(in)  :: dstran(ntens)
    double precision, intent(out) :: stress(ntens)

    double precision :: lam, mu
    integer :: i, j

    mu  = E / (2.0d0 * (1.0d0 + nu))
    lam = E * nu / ((1.0d0 + nu) * (1.0d0 - 2.0d0 * nu))

    ! stress = C @ dstran  (Voigt, engineering shear)
    do i = 1, ntens
        stress(i) = 0.0d0
    end do

    ! Normal-normal block: C_ij = lam  (i/=j, both in 1:3)
    !                      C_ii = lam + 2*mu  (i in 1:3)
    do i = 1, 3
        do j = 1, 3
            if (i == j) then
                stress(i) = stress(i) + (lam + 2.0d0 * mu) * dstran(j)
            else
                stress(i) = stress(i) + lam * dstran(j)
            end if
        end do
    end do

    ! Shear-shear block: C_ii = mu  (i in 4:ntens)
    do i = 4, ntens
        stress(i) = stress(i) + mu * dstran(i)
    end do
end subroutine elastic_stress


! -----------------------------------------------------------------------------
! elastic_stiffness -- return the full (ntens x ntens) stiffness matrix
!
! Parameters
! ----------
! E      [in]  : Young's modulus
! nu     [in]  : Poisson's ratio
! C      [out] : stiffness tensor, shape (ntens, ntens)
! ntens  [in]  : number of stress/strain components
! -----------------------------------------------------------------------------
subroutine elastic_stiffness(E, nu, C, ntens)
    implicit none
    integer,          intent(in)  :: ntens
    double precision, intent(in)  :: E, nu
    double precision, intent(out) :: C(ntens, ntens)

    double precision :: lam, mu
    integer :: i, j

    mu  = E / (2.0d0 * (1.0d0 + nu))
    lam = E * nu / ((1.0d0 + nu) * (1.0d0 - 2.0d0 * nu))

    do j = 1, ntens
        do i = 1, ntens
            C(i, j) = 0.0d0
        end do
    end do

    do i = 1, 3
        do j = 1, 3
            C(i, j) = lam
        end do
        C(i, i) = lam + 2.0d0 * mu
    end do

    do i = 4, ntens
        C(i, i) = mu
    end do
end subroutine elastic_stiffness
