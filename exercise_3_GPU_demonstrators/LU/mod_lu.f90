
module mod_lu

  integer, parameter :: dp = kind(0.0d0)

contains

  subroutine linearsolve_dc(Amat, bmat)
    implicit none
    !interface variables
    complex (kind=dp), intent(in)    :: Amat(:,:)
    complex (kind=dp), intent(inout) :: bmat(:,:)
    !local variables
    integer :: nrow, ncol
    integer :: info
    integer :: temp(size(bmat, 1))
   
    nrow = size(bmat, 1)
    ncol = size(bmat, 2)
     
    if (nrow /= size(Amat, 1) .or. nrow /= size(Amat, 2)) then
      stop '[linearsolve] dimension error while solving Ax=b' 
    end if
  
    !--------------------------------------------------
    !-- solve the system of linear equations         --
    !--------------------------------------------------
    call zgetrf(nrow, nrow, Amat, nrow, temp, info)
    if(info /= 0) then 
      WRITE(*,*) 'info', info
      stop '[linearsolve_dc] zgetrf failed' 
    endif
  
    call zgetrs('n', nrow, ncol, Amat, nrow, temp, bmat, nrow, info)
    if(info /= 0) then 
      WRITE(*,*) 'info',info
      stop '[linearsolve_dc] zgetrf failed' 
    endif

  end subroutine linearsolve_dc


  subroutine linearsolve_dc_batched(Amat, bmat)
    implicit none
    !interface variables
    complex (kind=dp), intent(in)    :: Amat(:,:,:)
    complex (kind=dp), intent(inout) :: bmat(:,:,:)
    !local variables
    integer :: nrow, ncol, batch_size
    integer :: info, irun
    integer :: temp(size(bmat, 1))
   
    nrow = size(bmat, 1)
    ncol = size(bmat, 2)
    batch_size = size(bmat, 3)
     
    if (nrow /= size(Amat, 1) .or. nrow /= size(Amat, 2)) then
      stop '[linearsolve] dimension error while solving Ax=b' 
    end if
  
    !--------------------------------------------------
    !-- solve the system of linear equations         --
    !--------------------------------------------------
    ! TODO: change to batched version
    do irun = 1, batch_size

      call zgetrf(nrow, nrow, Amat(:,:,irun), nrow, temp, info)
      if(info /= 0) then 
        WRITE(*,*) 'info', info
        stop '[linearsolve_dc] zgetrf failed' 
      endif
    
      call zgetrs('n', nrow, ncol, Amat(:,:,irun), nrow, temp, bmat(:,:,irun), nrow, info)
      if(info /= 0) then 
        WRITE(*,*) 'info',info
        stop '[linearsolve_dc] zgetrf failed' 
      endif

    end do 

  end subroutine linearsolve_dc_batched


  subroutine linearsolve_dc_GPU(Amat, bmat, handle_cusolver)
    use cublas_v2
    use cusolverDn
    use openacc

    implicit none
    !interface variables
    complex(kind=dp),intent(in)        :: Amat(:,:)
    complex(kind=dp),intent(inout)     :: bmat(:,:)
    type (cusolverDnHandle) :: handle_cusolver
    !local variables
    integer                             :: nrow,ncol
    integer                             :: info
    integer                             :: temp(size(bmat,1))
    ! for GPU code
    integer :: lwork, ierr
    complex(kind=dp), dimension(:), allocatable :: workspace

    nrow = size(bmat, 1)
    ncol = size(bmat, 2)
     
    if (nrow /= size(Amat, 1) .or. nrow /= size(Amat, 2)) then
      stop '[linearsolve] dimension error while solving Ax=b' 
    end if

    !--------------------------------------------------
    !-- solve the system of linear equations         --
    !--------------------------------------------------
    ! GPU code using cusolver
    !$acc data create(temp) copyout(info)
    !$acc host_data use_device(temp, info)
    ierr = cusolverDnZgetrf_buffersize(handle_cusolver, nrow, nrow, Amat, nrow, lwork)
    if (ierr/=0) stop 'Error cusolverZgetrf_buffersize'

    allocate(workspace(lwork))

    ierr = cusolverDnZgetrf(handle_cusolver, nrow, nrow, Amat, nrow, workspace, temp, info)
    if (ierr/=0) stop 'Error cusolverZgetrf'

    ierr = cusolverDnZgetrs(handle_cusolver, CUBLAS_OP_N, nrow, ncol, Amat, nrow, temp, bmat, nrow, info)
    if (ierr/=0) stop 'Error cusolverZgetrs bmat'

    deallocate(workspace)
    
    !$acc end host_data
    !$acc end data
    !$acc wait

  end subroutine linearsolve_dc_GPU


  subroutine linearsolve_dc_GPU_batched(Amat, bmat, handle_cusolver)
    use cublas_v2
    use cusolverDn
    use openacc

    implicit none
    !interface variables
    complex (kind=dp), intent(in)    :: Amat(:,:,:)
    complex (kind=dp), intent(inout) :: bmat(:,:,:)
    type (cusolverDnHandle) :: handle_cusolver
    !local variables
    integer :: nrow, ncol, batch_size
    integer :: info, irun
    integer :: temp(size(bmat, 1))
    ! for GPU code
    integer :: lwork, ierr
    complex(kind=dp), dimension(:), allocatable :: workspace

    nrow = size(bmat, 1)
    ncol = size(bmat, 2)
    batch_size = size(bmat, 3)
     
    if (nrow /= size(Amat, 1) .or. nrow /= size(Amat, 2)) then
      stop '[linearsolve] dimension error while solving Ax=b' 
    end if

    !--------------------------------------------------
    !-- solve the system of linear equations         --
    !--------------------------------------------------
    ! GPU code using cusolver

    ! TODO: change to batched version
    do irun = 1, batch_size

      !$acc data create(temp) copyout(info)
      !$acc host_data use_device(temp, info)
      ierr = cusolverDnZgetrf_buffersize(handle_cusolver, nrow, nrow, Amat(:,:,irun), nrow, lwork)
      if (ierr/=0) stop 'Error cusolverZgetrf_buffersize'

      allocate(workspace(lwork))

      ierr = cusolverDnZgetrf(handle_cusolver, nrow, nrow, Amat(:,:,irun), nrow, workspace, temp, info)
      if (ierr/=0) stop 'Error cusolverZgetrf'

      ierr = cusolverDnZgetrs(handle_cusolver, CUBLAS_OP_N, nrow, ncol, Amat(:,:,irun), nrow, temp, bmat(:,:,irun), nrow, info)
      if (ierr/=0) stop 'Error cusolverZgetrs bmat'

      deallocate(workspace)
      
      !$acc end host_data
      !$acc end data
      !$acc wait

    end do ! irun

  end subroutine linearsolve_dc_GPU_batched

end module mod_lu
