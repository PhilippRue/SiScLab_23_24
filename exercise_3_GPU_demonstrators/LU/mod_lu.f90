
module mod_lu

  integer, parameter :: dp = kind(0.0d0)

contains

  subroutine linearsolve_dc(Amat, Bmat)
    implicit none
    !interface variables
    complex (kind=dp), intent(in)    :: Amat(:,:)
    complex (kind=dp), intent(inout) :: Bmat(:,:)
    !local variables
    integer :: nrow, ncol
    integer :: info
    integer :: ipiv(size(Bmat, 1)) !! pivot array
   
    nrow = size(Bmat, 1)
    ncol = size(Bmat, 2)
     
    if (nrow /= size(Amat, 1) .or. nrow /= size(Amat, 2)) then
      stop '[linearsolve] dimension error while solving Ax=b' 
    end if
  
    !--------------------------------------------------
    !-- solve the system of linear equations         --
    !--------------------------------------------------
    call zgetrf(nrow, nrow, Amat, nrow, ipiv, info)
    if(info /= 0) then 
      WRITE(*,*) 'info', info
      stop '[linearsolve_dc] zgetrf failed' 
    endif
  
    call zgetrs('n', nrow, ncol, Amat, nrow, ipiv, Bmat, nrow, info)
    if(info /= 0) then 
      WRITE(*,*) 'info',info
      stop '[linearsolve_dc] zgetrf failed' 
    endif

  end subroutine linearsolve_dc


  subroutine linearsolve_dc_batched(Amat, Bmat)
    implicit none
    !interface variables
    complex (kind=dp), intent(in)    :: Amat(:,:,:)
    complex (kind=dp), intent(inout) :: Bmat(:,:,:)
    !local variables
    integer :: nrow, ncol, batch_size
    integer :: info, irun
    integer :: ipiv(size(Bmat, 1)) !! pivot array
   
    nrow = size(Bmat, 1)
    ncol = size(Bmat, 2)
    batch_size = size(Bmat, 3)
     
    if (nrow /= size(Amat, 1) .or. nrow /= size(Amat, 2)) then
      stop '[linearsolve] dimension error while solving Ax=b' 
    end if
  
    !--------------------------------------------------
    !-- solve the system of linear equations         --
    !--------------------------------------------------
    ! TODO: change to batched version
    do irun = 1, batch_size

      call zgetrf(nrow, nrow, Amat(:,:,irun), nrow, ipiv, info)
      if(info /= 0) then 
        WRITE(*,*) 'info', info
        stop '[linearsolve_dc] zgetrf failed' 
      endif
    
      call zgetrs('n', nrow, ncol, Amat(:,:,irun), nrow, ipiv, Bmat(:,:,irun), nrow, info)
      if(info /= 0) then 
        WRITE(*,*) 'info',info
        stop '[linearsolve_dc] zgetrf failed' 
      endif

    end do 

  end subroutine linearsolve_dc_batched


  subroutine linearsolve_dc_GPU(Amat, Bmat, handle_cusolver, ithread)
    use cublas_v2
    use cusolverDn
    use openacc

    implicit none
    !interface variables
    complex(kind=dp),intent(in)    :: Amat(:,:)
    complex(kind=dp),intent(inout) :: Bmat(:,:)
    type (cusolverDnHandle) :: handle_cusolver
    integer, optional :: ithread !! thread index for syncronizing correct queue with outer OpenMP parallism
    !local variables
    integer :: nrow,ncol
    integer :: info
    integer :: ipiv(size(Bmat,1)) !! pivot array
    ! for GPU code
    integer :: lwork, ierr
    complex(kind=dp), dimension(:), allocatable :: workspace

    nrow = size(Bmat, 1)
    ncol = size(Bmat, 2)
     
    if (nrow /= size(Amat, 1) .or. nrow /= size(Amat, 2)) then
      stop '[linearsolve] dimension error while solving Ax=b' 
    end if


    !--------------------------------------------------
    !-- solve the system of linear equations         --
    !--------------------------------------------------
    ! GPU code using cusolver
    !$acc data create(ipiv) copyout(info)
    !$acc host_data use_device(ipiv, info)
    ierr = cusolverDnZgetrf_buffersize(handle_cusolver, nrow, nrow, Amat, nrow, lwork)
    if (ierr/=0) stop 'Error cusolverZgetrf_buffersize'

    allocate(workspace(lwork))

    ierr = cusolverDnZgetrf(handle_cusolver, nrow, nrow, Amat, nrow, workspace, ipiv, info)
    if (ierr/=0) stop 'Error cusolverZgetrf'

    ierr = cusolverDnZgetrs(handle_cusolver, CUBLAS_OP_N, nrow, ncol, Amat, nrow, ipiv, Bmat, nrow, info)
    if (ierr/=0) stop 'Error cusolverZgetrs Bmat'

    deallocate(workspace)
    
    !$acc end host_data
    !$acc end data
    if (present(ithread)) then
      !$acc wait(ithread)
    else
      !$acc wait
    end if

  end subroutine linearsolve_dc_GPU


  subroutine linearsolve_dc_GPU_batched(Amat, Bmat, handle_cublas, ithread)
    use cublas_v2
    use cusolverDn
    use openacc

    implicit none
    !interface variables
    complex (kind=dp), intent(in)    :: Amat(:,:,:)
    complex (kind=dp), intent(inout) :: Bmat(:,:,:)
    type (cublasHandle) :: handle_cublas
    integer, optional :: ithread !! thread index for syncronizing correct queue with outer OpenMP parallism
    !local variables
    type(c_devptr), allocatable :: devPtr_Amat(:)
    type(c_devptr), allocatable :: devPtr_Bmat(:)
    integer :: nrow, ncol, batch_size
    integer :: ibatch
    integer, allocatable :: ipiv(:, :) !! pivot array, has additionally batch_size as second dimension
    integer, allocatable :: info(:) !! info array (size is batch_size)
    ! for GPU code
    integer :: lwork, ierr
    complex(kind=dp), dimension(:), allocatable :: workspace

    nrow = size(Bmat, 1)
    ncol = size(Bmat, 2)
    batch_size = size(Bmat, 3)
     
    if (nrow /= size(Amat, 1) .or. nrow /= size(Amat, 2)) then
      stop '[linearsolve] dimension error while solving Ax=b' 
    end if

    allocate(devPtr_Amat(batch_size), devPtr_Bmat(batch_size))
    allocate(ipiv(nrow, batch_size), info(batch_size))

    !--------------------------------------------------
    !-- solve the system of linear equations         --
    !--------------------------------------------------
    ! GPU code using cusolver

    ! set C device pointer that is used by the batched cublas LU routines instead of the arrays directly
    do ibatch = 1, batch_size
      devPtr_Amat(ibatch) = c_devloc(Amat(lbound(Amat,1), lbound(Amat,2), ibatch))
      devPtr_Bmat(ibatch) = c_devloc(Bmat(lbound(Bmat,1), lbound(Bmat,2), ibatch))
    end do

    !$acc data copyin(devPtr_Amat, devPtr_Bmat) create(ipiv, info)
    !$acc host_data use_device(ipiv, info, devPtr_Amat, devPtr_Bmat)

    ierr = cublasZgetrfBatched(handle_cublas, nrow, devPtr_Amat, nrow, ipiv, info, batch_size)
    if (ierr/=0) stop 'Error cublasCgetrfBatched'

    ierr = cublasZgetrsBatched(handle_cublas, CUBLAS_OP_N, nrow, ncol, devPtr_Amat, nrow, ipiv, devPtr_Bmat, nrow, info, batch_size)
    if (ierr/=0) stop 'Error cusolverZgetrs Bmat'
    
    if (present(ithread)) then
      !$acc wait(ithread)
    else
      !$acc wait
    end if

    !$acc end host_data
    !$acc end data

    deallocate(devPtr_Amat, devPtr_Bmat)
    deallocate(ipiv, info)

  end subroutine linearsolve_dc_GPU_batched

end module mod_lu
