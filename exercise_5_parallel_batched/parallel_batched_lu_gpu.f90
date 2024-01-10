program test

  use nvtx
  use cublas_v2
  use cusolverDn
  use openacc
  use omp_lib
  use mod_lu, only: linearsolve_dc_gpu, linearsolve_dc_gpu_batched
  use mod_matrix_tools, only: init_matrices_batched
  use mod_read_cmdline, only: get_mat_size_from_input

  implicit none

  integer, parameter :: dp = kind(0.0d0)
  integer, parameter :: m = 2
  integer :: n = 64, batch_size = 1000
  integer :: n_outer = 4 !! outer loop over which OpenMP parallel region is done
  complex (kind=dp), parameter :: cone = (1.0_dp, 0.0_dp), czero = (0.0_dp, 0.0_dp)
  integer :: i, irun, iloop, ithread, nthreads, num_dev, dev_id
  complex (kind=dp), allocatable :: a_array(:,:,:), a2_array(:,:,:), b_array(:,:,:), b2_array(:,:,:), c_array(:,:,:)
  complex (kind=dp), allocatable :: a_array_big(:,:,:,:), a2_array_big(:,:,:,:), b_array_big(:,:,:,:), b2_array_big(:,:,:,:), c_array_big(:,:,:,:)

  integer :: ierr
  integer :: clock_rate, start_time, start_time0, stop_time
  real (kind=dp), allocatable :: timings(:)

  type (cublasHandle) :: handle_cublas
  type (cusolverDnHandle) :: handle_cusolver


  ! find clock rate for timing measurements
  call system_clock(count_rate=clock_rate) ! Find the rate

  ! read matrix sizes from commandline
  call get_mat_size_from_input(n, batch_size, n_outer)

  ! move allocation of arrays here to be able to share it
  call nvtxStartRange('init_mat', 2)
  allocate(a_array_big(n,n,batch_size,n_outer), a2_array_big(n,n,batch_size,n_outer), &
           b_array_big(n,n,batch_size,n_outer), b2_array_big(n,n,batch_size,n_outer), &
           c_array_big(n,n,batch_size,n_outer))
  do iloop = 1, n_outer
    call init_matrices_batched(n, batch_size, a_array, a2_array, b_array, b2_array, c_array, timings)
    a_array_big(:,:,:,iloop) = a_array(:,:,:)
    a2_array_big(:,:,:,iloop) = a2_array(:,:,:)
    b_array_big(:,:,:,iloop) = b_array(:,:,:)
    b2_array_big(:,:,:,iloop) = b2_array(:,:,:)
    c_array_big(:,:,:,iloop) = c_array(:,:,:)
  end do ! iloop
  ! clean up 'small' matrices
  deallocate(a_array, a2_array, b_array, b2_array, c_array)
  call nvtxEndRange()

  ! print GPU info
  write(*, *) 'num GPU devices', acc_get_num_devices(acc_get_device_type())
  
  ! measure timing of all of the parallel region
  call system_clock(count=start_time0)

  ! start parallel region where a private handle is created for each CPU thread, these then submit batched calls to the GPUs
  !$omp parallel default(private) shared(n, batch_size, n_outer, timings, clock_rate, a_array_big, a2_array_big, b_array_big, b2_array_big, c_array_big) reduction(+:timings)

  ! index of OpenMP thread
  ithread = omp_get_thread_num()
  nthreads = omp_get_num_threads()
  !$omp single
  write(*,'(A, 1I3)') '# OpenMP threads', nthreads
  !$omp end single

  ! divide OpenMP threads to different GPUs on the node in a round-robin fashion
  num_dev = acc_get_num_devices(acc_get_device_type())
  dev_id = mod(ithread, num_dev)
  call acc_set_device_num(dev_id, acc_get_device_type())

  !$omp critical
  write(*,'(4(A,1I3))') 'My OpenMP thread', ithread + 1, ' /', nthreads, &
                        ', GPU device', acc_get_device_num(acc_get_device_type()) + 1, ' /', num_dev
  !$omp end critical


  ! create cublas/cusolver handles for ZGEMM and cusolver handle for LU decomposition
  ! handles are created with the ithread stread id to have separate streams for each thread

  call system_clock(count=start_time)
  call nvtxStartRange('create_handle', 1) ! icolor argument is different for better visibility

  ierr = cublasCreate(handle_cublas)
  ierr = ierr + cublasSetStream(handle_cublas, acc_get_cuda_stream(ithread))
  if (ierr/=CUBLAS_STATUS_SUCCESS) stop 'Error in cublasCreate'

  ierr = cusolverDnCreate(handle_cusolver)
  ierr = ierr + cusolverDnSetStream(handle_cusolver, acc_get_cuda_stream(ithread))
  if (ierr/=0) stop 'Error in cusovlerCreate'

  call nvtxEndRange()
  call system_clock(count=stop_time) ! Stop timing
  timings(1) = timings(1) + (stop_time-start_time)/real(clock_rate)

  ! ================== batched LU calls ==================

  ! now start the OpenMP loop (do the same thing for simplicity here)
  !$omp do
  do iloop = 1, n_outer
  
    ! start a measurement region in nvtx
    call system_clock(count=start_time)
    call nvtxStartRange('LU - GPU - batched', 3)

    ! solve linear system of equations using LU decomposition
    ! b is overwritten in here
    call linearsolve_dc_gpu_batched(a_array_big(:,:,:,iloop), b_array_big(:,:,:,iloop), handle_cublas, ithread)

    ! end a measurement reagion in nvtx and measure time
    call nvtxEndRange()
    call system_clock(count=stop_time) ! Stop timing
    timings(2) = timings(2) + (stop_time-start_time)/real(clock_rate)

    ! ==================

    ! start region for ZGEMM
    call system_clock(count=start_time)
    call nvtxStartRange('ZGEMM - GPU - batched', 4)

    ! check result with a zgemm (we solve a*b=c which we can check here)
    ! c = matmul(a2, b)

    !$acc data create(c_array_big)
    !$acc host_data use_device(a2_array_big, b_array_big, c_array_big)
    ierr = cublasZgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, cone, a2_array_big(:,:,:,iloop), n, n*n, b_array_big(:,:,:,iloop), n, n*n, czero, c_array_big(:,:,:,iloop), n, n*n, batch_size)
    if (ierr /= CUBLAS_STATUS_SUCCESS) stop 'Error in cublasZgemm_v2'
    !$acc end host_data
    !$acc end data
    !$acc wait(ithread)

    ! end region for ZGEMM
    call nvtxEndRange()
    call system_clock(count=stop_time) ! Stop timing
    timings(3) = timings(3) + (stop_time-start_time)/real(clock_rate)

    ! check c == b2
    !$omp critical
    write(*, *) 'max difference', maxval(real(c_array_big(:,:,:,iloop)-b2_array_big(:,:,:,iloop), kind=dp))
    !$omp end critical

    ! ================== end batched LU calls ==================

  end do ! iloop
  !$omp end do


  ! destroy cublas and cusolver handles
  call system_clock(count=start_time)
  call nvtxStartRange('destroy_handle', 5) ! icolor argument is different for better visibility
  ierr = cublasDestroy(handle_cublas)
  if (ierr/=CUBLAS_STATUS_SUCCESS) stop 'Error in cublasDestroy'

  ierr = cusolverDnDestroy(handle_cusolver)
  if (ierr/=0) stop 'Error in cusolverDestroy'

  call nvtxEndRange()
  call system_clock(count=stop_time) ! Stop timing
  timings(4) = timings(4) + (stop_time-start_time)/real(clock_rate)

  !$omp end parallel
  ! done with parallel region
  
  ! clean up memory allocation
  deallocate(a_array_big, b_array_big, c_array_big, a2_array_big, b2_array_big)
  
  call system_clock(count=stop_time) ! Stop timing
  timings(5) = (stop_time-start_time0)/real(clock_rate)
  
  ! print timings
  write(*, *)
  write(*, '(A,5ES12.5)') 'timings (total, h_create, LU, ZGEMM, h_destroy):', timings(5), timings(1:4)

end program test
