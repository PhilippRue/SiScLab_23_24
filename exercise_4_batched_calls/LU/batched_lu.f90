program test

  use nvtx
  use mod_lu, only: linearsolve_dc, linearsolve_dc_batched
  use mod_matrix_tools, only: init_matrices_batched
  use mod_read_cmdline, only: get_mat_size_from_input
  implicit none
  integer, parameter :: dp = kind(0.0d0)
  integer, parameter :: m = 2
  integer :: n = 64, num_mult = 1000
  complex (kind=dp), parameter :: cone = (1.0_dp, 0.0_dp), czero = (0.0_dp, 0.0_dp)
  integer :: i, irun
  complex (kind=dp), allocatable :: a_array(:,:,:), a2_array(:,:,:), b_array(:,:,:), b2_array(:,:,:), c_array(:,:,:)

  integer :: ierr
  integer :: clock_rate, start_time, start_time0, stop_time
  real (kind=dp), allocatable :: timings(:)


  ! find clock rate for timing measurements
  call system_clock(count_rate=clock_rate) ! Find the rate

  ! read matrix sizes from commandline
  call get_mat_size_from_input(n, num_mult)

  ! ================== loop over multiple LU calls ==================

  ! initialize matrices
  call init_matrices_batched(n, num_mult, a_array, a2_array, b_array, b2_array, c_array, timings)

  ! start measurement of total runtime
  call system_clock(count=start_time0)

  ! start a measurement region in nvtx
  call system_clock(count=start_time)
  call nvtxStartRange('LU - CPU', 1) ! icolor argument is different for better visibility

  ! solve linear system of equations using LU decomposition
  ! b is overwritten in here
  do irun = 1, num_mult
    call linearsolve_dc(a_array(:,:,irun), b_array(:,:,irun))
  end do

  ! end a measurement reagion in nvtx and measure time
  call nvtxEndRange()
  call system_clock(count=stop_time) ! Stop timing
  timings(1) = (stop_time-start_time)/real(clock_rate)

  ! ==================

  ! start region for ZGEMM
  call system_clock(count=start_time)
  call nvtxStartRange('ZGEMM - CPU', 2) ! icolor argument is different for better visibility

  ! check result with a zgemm (we solve a*b=c which we can check here)
  ! c = matmul(a2, b)
  do irun = 1, num_mult
    call zgemm('N', 'N', n, m, n, cone, a2_array(:,:,irun), n, b_array(:,:,irun), n, czero, c_array(:,:,irun), n)
  end do

  ! end region for ZGEMM
  call nvtxEndRange()
  call system_clock(count=stop_time) ! Stop timing
  timings(2) = (stop_time-start_time)/real(clock_rate)

  ! check c == b2
  write(*, *)
  write(*, *) 'max difference', maxval(real(c_array-b2_array, kind=dp))

  ! ================== end loop over multiple LU calls ==================

  ! ================== batched LU calls ==================

  ! initialize matrices
  call init_matrices_batched(n, num_mult, a_array, a2_array, b_array, b2_array, c_array, timings)

  ! start a measurement region in nvtx
  call system_clock(count=start_time)
  call nvtxStartRange('LU - CPU - batched', 3) ! icolor argument is different for better visibility

  ! solve linear system of equations using LU decomposition
  ! b is overwritten in here
  call linearsolve_dc_batched(a_array, b_array)

  ! end a measurement reagion in nvtx and measure time
  call nvtxEndRange()
  call system_clock(count=stop_time) ! Stop timing
  timings(3) = (stop_time-start_time)/real(clock_rate)

  ! ==================

  ! start region for ZGEMM
  call system_clock(count=start_time)
  call nvtxStartRange('ZGEMM - CPU - batched', 4) ! icolor argument is different for better visibility

  ! check result with a zgemm (we solve a*b=c which we can check here)
  ! c = matmul(a2, b)
  call zgemm_batch_strided('N', 'N', n, n, n, cone, a2_array, n, n*n, b_array, n, n*n, czero, c_array, n, n*n, num_mult)

  ! end region for ZGEMM
  call nvtxEndRange()
  call system_clock(count=stop_time) ! Stop timing
  timings(4) = (stop_time-start_time)/real(clock_rate)

  ! check c == b2
  write(*, *)
  write(*, *) 'max difference', maxval(real(c_array-b2_array, kind=dp))

  ! ================== end batched LU calls ==================

  ! total runtime
  call system_clock(count=stop_time) ! Stop timing
  timings(5) = (stop_time-start_time0)/real(clock_rate)

  ! print timings
  write(*, *) 
  write(*, '(A,5ES12.5)') 'timings:', timings(1:5)


  ! clean up memory allocation
  deallocate(a_array, b_array, c_array)

end program test
