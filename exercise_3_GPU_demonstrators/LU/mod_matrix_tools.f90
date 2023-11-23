module mod_matrix_tools

  implicit none
  integer, parameter :: dp = kind(0.0d0)
  complex (kind=dp), parameter :: cone = (1.0_dp, 0.0_dp), czero = (0.0_dp, 0.0_dp)

contains

  subroutine init_matrices(n, m, a, a2, b, b2, c, timings)
    implicit none
    ! interface
    integer :: n, m
    complex (kind=dp), allocatable :: a(:,:), a2(:,:), b(:,:), b2(:,:), c(:,:)
    real (kind=dp), allocatable :: timings(:)
    ! local
    integer :: i
    character(100) :: fmt


    ! allocate memory for working arrays
    if (.not. allocated(a)) then
      allocate(a(n,n), a2(n,n), b(n,m), b2(n,m), c(n,m), timings(10))
    end if

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
      write(*, fmt) a(i, :)
    end do
    write(*,'(A)') '# B matrix'
    do i = 1, n
      write(*,fmt) b(i, :)
    end do

  end subroutine init_matrices

end module mod_matrix_tools
