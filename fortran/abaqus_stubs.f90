! =============================================================================
! manforge -- ABAQUS internal function stubs for standalone UMAT testing
!
! Provides mock implementations of ABAQUS utility subroutines so that
! UMAT code can be compiled and tested without the ABAQUS solver.
!
! Subroutines provided:
!   SINV   -- stress invariants (trace and von Mises equivalent)
!   SPRINC -- principal values of a symmetric Voigt tensor (Jacobi method)
!   ROTSIG -- rotate a Voigt tensor by an incremental rotation matrix
!
! Convention: Voigt order [s11, s22, s33, s12, s13, s23]
!             Engineering shear (gamma = 2*eps_shear)
!
! NOTE: These stubs are linked into the f2py module for symbol resolution
!       (see Makefile target "fortran-build-umat").  The current umat_j2_run
!       subroutine does NOT call them, but they satisfy the linker when
!       compiling UMATs that reference ABAQUS utility functions.  Future
!       UMAT implementations (e.g., Drucker-Prager) may call them directly.
! =============================================================================

! -----------------------------------------------------------------------------
! SINV -- compute stress invariants
!
! Parameters
! ----------
! STRESS(NTENS) [in]  : stress in Voigt notation
! SINV1         [out] : first invariant I1 = trace(sigma) = s11+s22+s33
! SINV2         [out] : second invariant = von Mises equivalent stress
!                       sigma_vm = sqrt(1.5 * s:s)  where s = dev(sigma)
! NDI           [in]  : number of direct components (3 for 3D)
! NSHR          [in]  : number of shear components (3 for 3D)
! -----------------------------------------------------------------------------
subroutine SINV(STRESS, SINV1, SINV2, NDI, NSHR)
    implicit none
    integer,          intent(in)  :: NDI, NSHR
    double precision, intent(in)  :: STRESS(NDI + NSHR)
    double precision, intent(out) :: SINV1, SINV2

    double precision :: p, s(6), sss
    integer :: i, ntens

    ntens = NDI + NSHR

    ! First invariant: trace
    SINV1 = 0.0d0
    do i = 1, NDI
        SINV1 = SINV1 + STRESS(i)
    end do

    ! Deviatoric stress
    p = SINV1 / dble(NDI)
    do i = 1, ntens
        s(i) = STRESS(i)
    end do
    do i = 1, NDI
        s(i) = s(i) - p
    end do

    ! s:s in Voigt notation (shear components doubled for double contraction)
    sss = 0.0d0
    do i = 1, NDI
        sss = sss + s(i)**2
    end do
    do i = NDI+1, ntens
        sss = sss + 2.0d0 * s(i)**2
    end do

    ! Von Mises = sqrt(3/2 * s:s)
    SINV2 = sqrt(1.5d0 * sss)
end subroutine SINV


! -----------------------------------------------------------------------------
! SPRINC -- compute principal values of a symmetric tensor in Voigt notation
!
! Uses the Jacobi iterative method for the 3x3 symmetric eigenvalue problem.
!
! Parameters
! ----------
! S(NTENS) [in]  : tensor in Voigt notation [s11,s22,s33,s12,s13,s23]
! PS(3)    [out] : three principal values (eigenvalues), unsorted
! LSTR     [in]  : 1 = stress (shear = stress), 2 = strain (shear = gamma/2)
! NDI      [in]  : number of direct components (3)
! NSHR     [in]  : number of shear components (3)
! -----------------------------------------------------------------------------
subroutine SPRINC(S, PS, LSTR, NDI, NSHR)
    implicit none
    integer,          intent(in)  :: LSTR, NDI, NSHR
    double precision, intent(in)  :: S(NDI + NSHR)
    double precision, intent(out) :: PS(3)

    ! 3x3 symmetric matrix A
    double precision :: A(3,3), G(3,3), tmp(3,3)
    double precision :: theta, t, c, s12, tau
    double precision :: off, aij
    integer :: iter, i, j, k, p, q
    integer, parameter :: MAX_ITER = 100
    double precision, parameter :: TOL = 1.0d-12

    ! Shear factor: 1 for stress, 0.5 for strain (engineering -> tensor)
    double precision :: sfac
    if (LSTR == 2) then
        sfac = 0.5d0
    else
        sfac = 1.0d0
    end if

    ! Build 3x3 matrix from Voigt [s11,s22,s33,s12,s13,s23]
    A(1,1) = S(1);  A(2,2) = S(2);  A(3,3) = S(3)
    A(1,2) = sfac * S(4);  A(2,1) = A(1,2)
    A(1,3) = sfac * S(5);  A(3,1) = A(1,3)
    A(2,3) = sfac * S(6);  A(3,2) = A(2,3)

    ! Jacobi iteration
    do iter = 1, MAX_ITER
        ! Check off-diagonal norm
        off = 0.0d0
        do i = 1, 3
            do j = 1, 3
                if (i /= j) off = off + A(i,j)**2
            end do
        end do
        if (sqrt(off) < TOL) exit

        ! Sweep all off-diagonal pairs (p,q) with p < q
        do p = 1, 2
            do q = p+1, 3
                aij = A(p,q)
                if (abs(aij) < TOL) cycle
                tau = (A(q,q) - A(p,p)) / (2.0d0 * aij)
                if (tau >= 0.0d0) then
                    t = 1.0d0 / (tau + sqrt(1.0d0 + tau**2))
                else
                    t = -1.0d0 / (-tau + sqrt(1.0d0 + tau**2))
                end if
                c = 1.0d0 / sqrt(1.0d0 + t**2)
                s12 = t * c

                ! Build Givens rotation matrix G
                do i = 1, 3
                    do j = 1, 3
                        if (i == j) then
                            G(i,j) = 1.0d0
                        else
                            G(i,j) = 0.0d0
                        end if
                    end do
                end do
                G(p,p) = c;   G(q,q) = c
                G(p,q) = s12; G(q,p) = -s12

                ! A = G^T A G
                ! tmp = A G
                do i = 1, 3
                    do j = 1, 3
                        tmp(i,j) = 0.0d0
                        do k = 1, 3
                            tmp(i,j) = tmp(i,j) + A(i,k) * G(k,j)
                        end do
                    end do
                end do
                ! A = G^T tmp
                do i = 1, 3
                    do j = 1, 3
                        A(i,j) = 0.0d0
                        do k = 1, 3
                            A(i,j) = A(i,j) + G(k,i) * tmp(k,j)
                        end do
                    end do
                end do
            end do
        end do
    end do

    PS(1) = A(1,1)
    PS(2) = A(2,2)
    PS(3) = A(3,3)
end subroutine SPRINC


! -----------------------------------------------------------------------------
! ROTSIG -- rotate a symmetric Voigt tensor by an incremental rotation
!
! Computes SROT = DROT * S_mat * DROT^T in 3x3 form, then converts back.
!
! Parameters
! ----------
! S(NTENS)    [in]  : input tensor in Voigt notation
! DROT(3,3)   [in]  : incremental rotation matrix
! SROT(NTENS) [out] : rotated tensor in Voigt notation
! LSTR        [in]  : 1 = stress, 2 = strain
! NDI         [in]  : number of direct components (3)
! NSHR        [in]  : number of shear components (3)
! -----------------------------------------------------------------------------
subroutine ROTSIG(S, DROT, SROT, LSTR, NDI, NSHR)
    implicit none
    integer,          intent(in)  :: LSTR, NDI, NSHR
    double precision, intent(in)  :: S(NDI + NSHR)
    double precision, intent(in)  :: DROT(3, 3)
    double precision, intent(out) :: SROT(NDI + NSHR)

    double precision :: A(3,3), B(3,3), C(3,3)
    double precision :: sfac
    integer :: i, j, k

    if (LSTR == 2) then
        sfac = 0.5d0
    else
        sfac = 1.0d0
    end if

    ! Build 3x3 from Voigt
    A(1,1) = S(1);  A(2,2) = S(2);  A(3,3) = S(3)
    A(1,2) = sfac * S(4);  A(2,1) = A(1,2)
    A(1,3) = sfac * S(5);  A(3,1) = A(1,3)
    A(2,3) = sfac * S(6);  A(3,2) = A(2,3)

    ! B = DROT * A
    do i = 1, 3
        do j = 1, 3
            B(i,j) = 0.0d0
            do k = 1, 3
                B(i,j) = B(i,j) + DROT(i,k) * A(k,j)
            end do
        end do
    end do

    ! C = B * DROT^T
    do i = 1, 3
        do j = 1, 3
            C(i,j) = 0.0d0
            do k = 1, 3
                C(i,j) = C(i,j) + B(i,k) * DROT(j,k)
            end do
        end do
    end do

    ! Convert back to Voigt
    SROT(1) = C(1,1);  SROT(2) = C(2,2);  SROT(3) = C(3,3)
    SROT(4) = C(1,2) / sfac
    SROT(5) = C(1,3) / sfac
    SROT(6) = C(2,3) / sfac
end subroutine ROTSIG
