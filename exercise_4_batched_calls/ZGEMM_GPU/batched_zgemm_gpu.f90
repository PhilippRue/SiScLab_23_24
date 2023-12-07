program test_batched

  use cublas_v2
  use openacc
  use nvtx

  implicit none
  integer, parameter :: dp = kind(0.0d0)
  integer :: n = 64, nmax = 2, num_mult = 1000
  complex (kind=dp), parameter :: cone = (1.0_dp, 0.0_dp), czero = (0.0_dp, 0.0_dp)
  integer :: i
  real (kind=dp), allocatable :: tmp(:, :), tmp2(:,:)
  complex (kind=dp), allocatable :: a(:,:), b(:,:), c(:,:)
  complex (kind=dp), allocatable :: a_array(:,:,:), b_array(:,:,:), c_array(:,:,:)
  complex (kind=dp), allocatable :: a_work(:,:), b_work(:,:), c_work(:,:)
  integer :: irun, stridea, strideb, stridec, batch_size
  integer :: ierr
  integer :: clock_rate, start_time, stop_time
  real (kind=dp) :: timing1, timing2, timing3

  ! command line options
  integer :: iarg
  character(100) :: optchar, optchar2

  ! GPU libraries
  type (cublasHandle) :: handle_cublas


  ! create cublas handle
  ierr = cublasCreate(handle_cublas)
  ierr = ierr + cublasSetStream(handle_cublas, acc_get_cuda_stream(acc_async_sync))
  if (ierr/=CUBLAS_STATUS_SUCCESS) stop 'Error in cublasCreate'


  ! check for command line arguments
  if (command_argument_count() > 0) then
    do iarg = 1, command_argument_count()
      call get_command_argument(iarg, optchar)
      if (trim(optchar)=='--mat-size' .or. trim(optchar)=='-n') then
        call get_command_argument(iarg+1, optchar2)
        read(optchar2, *) n
      end if
      if (trim(optchar)=='--num-mult' .or. trim(optchar)=='-m') then
        call get_command_argument(iarg+1, optchar2)
        read(optchar2, *) num_mult
      end if
      if (trim(optchar)=='--help' .or. trim(optchar)=='-h') then
        write(*,'(A)') 'available command line options:'
        write(*,'(A)') '  --help or -h          print this help message and exit'
        write(*,'(A)') '  --mat-size or -n <I>  matrix size, default is 64'
        write(*,'(A)') '  --num-mult or -m <I>  number of matrices for batching, default 1000'
        stop
      end if
    end do
  endif

  write (*, '(A)') '  === settings ==='
  write (*, *) '  matrix size:', n
  write (*, *) '  number of matrices:', num_mult

  ! allocate memory of working arrays
  allocate(a(n,n), b(n,n), c(n,n))
  allocate(a_work(n,n), b_work(n,n), c_work(n,n))
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

  do irun = 1, num_mult
    ! copy input data
    a_work(:,:) = a_array(:,:,irun)
    b_work(:,:) = b_array(:,:,irun)

    !$acc data create(c_work)
    !$acc host_data use_device(a_work, b_work, c_work)
    ierr = cublasZgemm_v2(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, cone, a_work, n, b_work, n, czero, c_work, n)
    if (ierr /= CUBLAS_STATUS_SUCCESS) stop 'Error in cublasZgemm_v2'
    !$acc end host_data
    !$acc end data
    !$acc wait

    ! copy output data
    c_array(:,:,irun) = c_work(:,:)
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

  ! cublasZgemmStridedBatched(cublasHandle_t handle,
  !                                   cublasOperation_t transa,
  !                                   cublasOperation_t transb,
  !                                   int m, int n, int k,
  !                                   const cuDoubleComplex *alpha,
  !                                   const cuDoubleComplex *A, int lda,
  !                                   long long int          strideA,
  !                                   const cuDoubleComplex *B, int ldb,
  !                                   long long int          strideB,
  !                                   const cuDoubleComplex *beta,
  !                                   cuDoubleComplex       *C, int ldc,
  !                                   long long int          strideC,
  !                                   int batchCount)

  ! reinitialize
  c_array = czero

  !$acc data create(c_array)
  !$acc host_data use_device(a_array, b_array, c_array)
  ierr = cublasZgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, cone, a_array, n, stridea, b_array, n, strideb, czero, c_array, n, stridec, batch_size)
  if (ierr /= CUBLAS_STATUS_SUCCESS) stop 'Error in cublasZgemm_v2'
  !$acc end host_data
  !$acc end data
  !$acc wait

  call system_clock(count=stop_time) ! Stop timing
  timing2 = (stop_time-start_time)/real(clock_rate)

  ! end last measurement region in nvtx
  call nvtxEndRange()

  ! #################################################

  write(*, *) 
  write(*, *) 'timings batch strided', timing1, timing2

  ! write out comparison of results
  write(*, *)
  write(*, *) 'max difference loop:   ', maxval(real(c_work(:,:)-c, kind=dp))
  write(*, *) 'max difference batched:', maxval(real(c_array(:,:,num_mult)-c, kind=dp))

  ! clean up memory allocations
  deallocate(a, b, c, a_work, b_work, c_work, a_array, b_array, c_array)

  ! destroy cublas handle
  ierr = cublasDestroy(handle_cublas)
  if (ierr/=CUBLAS_STATUS_SUCCESS) stop 'Error in cublasDestroy'

end program test_batched