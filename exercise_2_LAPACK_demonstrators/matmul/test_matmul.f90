program test

  implicit none
  integer, parameter :: dp = kind(0.0d0)
  integer, parameter :: n = 10
  complex (kind=dp), parameter :: cone = (1.0_dp, 0.0_dp), czero = (0.0_dp, 0.0_dp)
  integer :: i
  complex (kind=dp), allocatable :: a(:,:), b(:,:), c(:,:), c2(:,:)
  character(100) :: fmt

  ! allocate memory
  allocate(a(n,n), b(n,n), c(n,n), c2(n,n))

  ! initialize a matrix
  a = czero
  do i = 1, n
    a(i, i) = cone
  end do
  a(1,3) = (0.0_dp, 1.0_dp)
  a(4,5) = (-1.0_dp, 0.0_dp)
  a(2,6) = (2.0_dp, 2.0_dp)
  a(7,8) = (3.0_dp, -5.0_dp)
  a(10,1) = (0.0_dp, 1.0_dp)

  ! initialize b matrix
  b = (1._dp, 0._dp)
  b(2,8) = (0.0_dp, -1.0_dp)
  b(5,2) = cone
  b(7,5) = (-2.0_dp, 2.0_dp)
  b(8,4) = (3.0_dp, 5.0_dp)
  b(10,1) = (0.0_dp, -1.0_dp)

  ! matrix-matrix multiply
  call zgemm('N', 'N', n, n, n, cone, a, n, b, n, czero, c, n)

  ! print output
  write(fmt, '(A, 1I5, A)') '(', 2*n, 'ES12.3)'
  write(*,'(A)') '# A matrix'
  do i = 1, n
    write(*,fmt) a(i, :)
  end do
  write(*,'(A)') '# B matrix'
  do i = 1, n
    write(*,fmt) b(i, :)
  end do
  write(*,'(A)') '# C matrix: C = A * B'
  do i = 1, n
    write(*,fmt) c(i, :)
  end do

  ! for validation compare with matmul instead of BLAS
  c2 = matmul(a, b)
  write(*, *)
  write(*,'(A)') '# C matrix from matmul instead of BLAS'
  do i = 1, n
    write(*,fmt) c2(i, :)
  end do
  write(*,'(A)') '# Difference to BLAS result'
  do i = 1, n
    write(*,fmt) c(i, :) - c2(i, :)
  end do
  write(*, *) 'max difference', maxval(real(c-c2, kind=dp))

  ! clean up memory allocations
  deallocate(a, b, c)

end program test
