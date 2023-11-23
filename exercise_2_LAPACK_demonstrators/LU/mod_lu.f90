
module mod_lu

contains

  subroutine linearsolve_dc(Amat, bmat)
    implicit none
    integer, parameter :: dp = kind(0.0d0)
    !interface variables
    complex (kind=dp), intent(in)    :: Amat(:,:)
    complex (kind=dp), intent(inout) :: bmat(:,:)
    !local variables
    integer :: nrow, ncol
    integer :: info
    integer :: temp(size(bmat, 1))
   
    nrow = size(bmat, 1)
    ncol = size(bmat, 2)
  
    !write(*,*) 'nrow, ncol', nrow, ncol, shape(Amat), shape(Bmat)
   
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

end module mod_lu
