program test_batched

  use nvtx
  use mod_read_cmdline, only: get_mat_size_from_input

  implicit none
  integer, parameter :: dp = kind(0.0d0)
  integer :: n = 64, nmax = 2, num_mult = 1000
  complex (kind=dp), parameter :: cone = (1.0_dp, 0.0_dp), czero = (0.0_dp, 0.0_dp)
  integer :: i
  real (kind=dp), allocatable :: tmp(:, :), tmp2(:,:)
  complex (kind=dp), allocatable :: a(:,:), b(:,:), c(:,:)
  complex (kind=dp), allocatable :: a_array(:,:,:), b_array(:,:,:), c_array(:,:,:)
  integer :: irun, stridea, strideb, stridec, batch_size
  integer :: ierr
  integer :: clock_rate, start_time, stop_time
  real (kind=dp) :: timing1, timing2, timing3

  ! read matrix sizes from commandline
  call get_mat_size_from_input(n, num_mult)

  ! allocate memory of working arrays
  allocate(a(n,n), b(n,n), c(n,n))
  allocate(a_array(n,n,num_mult), b_array(n,n,num_mult), c_array(n,n,num_mult))
  allocate(tmp(n,n), tmp2(n,n))

  ! find clock rate for timing measurements
  call system_clock(count_rate=clock_rate) ! Find the rate


  ! initialize A matrix with random numbers + unity matrix
  a = czero
  do i = 1, n
    a(i, i) = cone
  end do
  call random_number(tmp)
  call random_number(tmp2)
  a = a + cmplx(tmp, tmp2)

  ! initialize B matrix with random numbers
  b = cone
  call random_number(tmp)
  call random_number(tmp2)
  b = b + cmplx(tmp, tmp2)

  ! set array inputs
  do irun = 1, num_mult
    call random_number(tmp)
    call random_number(tmp2)
    a = a + cmplx(tmp, tmp2)
    call random_number(tmp)
    call random_number(tmp2)
    b = b + cmplx(tmp, tmp2)
    a_array(:,:, irun) = a(:,:)
    b_array(:,:, irun) = b(:,:)
    c_array(:,:, irun) = c(:,:)
  end do


  ! single matrix call for reference
  call zgemm('N', 'N', n, n, n, cone, a, n, b, n, czero, c, n)

  ! #################################################

  ! start a measurement reagion in nvtx
  call nvtxStartRange('ZGEMM - loop', 2) ! icolor argument is different for better visibility

  call system_clock(count=start_time)

  ! for validation compare with ZGEMM on CPU from LAPACK
  do irun = 1, num_mult
    call zgemm('N', 'N', n, n, n, cone, a_array(:,:,irun), n, b_array(:,:,irun), n, czero, c_array(:,:,irun), n)
  end do

  call system_clock(count=stop_time) ! Stop timing
  timing1 = (stop_time-start_time)/real(clock_rate)

  ! end last measurement region in nvtx
  call nvtxEndRange()

  ! #################################################

  ! start a measurement reagion in nvtx
  call nvtxStartRange('ZGEMM - batched', 1) ! icolor argument is different for better visibility

  call system_clock(count=start_time)

  ! compare with batched ZGEMM
  ! https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2024-0/gemm-batch-strided.html
  c_array = czero
  stridea = n*n
  strideb = n*n
  stridec = n*n
  batch_size = num_mult
  call zgemm_batch_strided('N', 'N', n, n, n, cone, a_array, n, stridea, b_array, n, strideb, czero, c_array, n, stridec, batch_size)

  call system_clock(count=stop_time) ! Stop timing
  timing2 = (stop_time-start_time)/real(clock_rate)

  ! end last measurement region in nvtx
  call nvtxEndRange()

  write(*, *) 
  write(*, *) 'timings batch strided', timing1, timing2

  ! write out alternative result and compare with GPU version
  write(*, *)
  write(*, *) 'max difference', maxval(real(c_array(:,:,num_mult)-c, kind=dp))

  ! clean up memory allocations
  deallocate(a, b, c)

end program test_batched