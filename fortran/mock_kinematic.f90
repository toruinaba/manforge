! =============================================================================
! mock_kinematic -- minimal dummy UMAT with two state variables (alpha, ep)
!
! Non-physical linear update used solely to test crosscheck_umat's ability to
! handle multi-state-variable models with ndarray state (alpha: 6-component
! back-stress vector, ep: scalar equivalent plastic strain).
!
! Algorithm (linear, non-physical):
!   stress_out = stress_in + E * dstran
!   alpha_out  = alpha_in  + H_kin * dstran
!   ep_out     = ep_in     + H_iso * sum(abs(dstran))
!
! Subroutines:
!   mock_kinematic  -- f2py-callable core logic
! =============================================================================

subroutine mock_kinematic(E, H_kin, H_iso, stress_in, alpha_in, ep_in, dstran, &
                          stress_out, alpha_out, ep_out)
    implicit none
    real(8), intent(in)  :: E, H_kin, H_iso
    real(8), intent(in)  :: stress_in(6), alpha_in(6), ep_in, dstran(6)
    real(8), intent(out) :: stress_out(6), alpha_out(6), ep_out

    stress_out = stress_in + E * dstran
    alpha_out  = alpha_in  + H_kin * dstran
    ep_out     = ep_in     + H_iso * sum(abs(dstran))

end subroutine mock_kinematic
