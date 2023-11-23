! Test program for a simple LU decomposition

program test

  use mod_lu, only: linearsolve_dc
  implicit none
  integer, parameter :: dp = kind(0.0d0)
  integer, parameter :: n = 10, m = 2
  complex (kind=dp), parameter :: cone = (1.0_dp, 0.0_dp), czero = (0.0_dp, 0.0_dp)
  integer :: i
  complex (kind=dp), allocatable :: a(:,:), a2(:,:), b(:,:), b2(:,:), c(:,:)
  character(100) :: fmt

  ! allocate memory for working arrays
  allocate(a(n,n), a2(n,n), b(n,m), b2(n,m), c(n,m))

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
  b = cone
  b(1,1) = (5.0_dp, 0.0_dp)
  b(5,2) = (1.0_dp, 1.0_dp)
  b(6,1) = (2.0_dp, 1.0_dp)
  b(8,2) = (1.0_dp, 3.0_dp)
  b(10,1) = (0.0_dp, -1.0_dp)

  ! copy a and b matrice to a2,b2 to save it for later comparison
  a2 = a
  b2 = b

  ! working array
  c = czero

  ! print input
  write(*, *) 'input:'
  write(fmt, '(A, I,A)') '(', 2*n, 'ES12.3)'
  write(*,'(A)') '# A matrix'
  do i = 1, n
    write(*,fmt) a(i, :)
  end do
  write(*,'(A)') '# B matrix'
  do i = 1, n
    write(*,fmt) b(i, :)
  end do

  ! solve linear system of equations using LU decomposition
  ! b is overwritten in here
  call linearsolve_dc(a, b)

  ! print output
  write(*, *)
  write(*, *) 'output:'
  write(fmt, '(A, I,A)') '(', 2*n, 'ES12.3)'
  write(*,'(A)') '# A matrix'
  do i = 1, n
    write(*,fmt) a(i, :)
  end do
  write(*,'(A)') '# B matrix'
  do i = 1, n
    write(*,fmt) b(i, :)
  end do

  ! check result with a zgemm (we solve a*b=c which we can check here)
  ! a*b = c
  call zgemm('N', 'N', n, m, n, cone, a2, n, b, n, czero, c, n)
  !c = matmul(a2, b)
  ! c == b2
  write(*, *)
  write(fmt, '(A, I,A)') '(', 2*n, 'ES12.3)'
  write(*,'(A)') '# result after zgemm'
  do i = 1, n
    write(*,fmt) c(i, :)
  end do
  write(*,'(A)') '# Difference to input'
  do i = 1, n
    write(*,fmt) c(i, :) - b2(i, :)
  end do
  write(*, *) 'max difference', maxval(real(c-b2, kind=dp))


  ! clean up memory allocation
  deallocate(a, a2, b, b2, c)

end program test

