module mod_matrix_tools

  implicit none
  integer, parameter :: dp = kind(0.0d0)
  complex (kind=dp), parameter :: cone = (1.0_dp, 0.0_dp), czero = (0.0_dp, 0.0_dp)

contains

  subroutine init_matrices(n, m, a, a2, b, b2, c, timings, init_random, print_input)
    implicit none
    ! interface
    integer :: n, m
    complex (kind=dp), allocatable :: a(:,:), a2(:,:), b(:,:), b2(:,:), c(:,:)
    real (kind=dp), allocatable :: timings(:)
    logical, optional :: print_input, init_random
    logical :: init_random_
    ! local
    integer :: i
    character(100) :: fmt
    real (kind=dp), allocatable :: tmp(:, :), tmp2(:,:)

    init_random_ = .false.
    if (present(init_random)) init_random_ = init_random


    ! allocate memory for working arrays
    if (.not. allocated(a)) then
      allocate(a(n,n), a2(n,n), b(n,m), b2(n,m), c(n,m))
    end if
    if (.not.allocated(timings)) allocate(timings(10))
    allocate(tmp(n,n), tmp2(n,n))

    ! initialize a matrix
    a = czero
    do i = 1, n
      a(i, i) = cone
    end do
    if (.not. init_random_) then
      a(1,3) = (0.0_dp, 1.0_dp)
      a(4,5) = (-1.0_dp, 0.0_dp)
      a(2,6) = (2.0_dp, 2.0_dp)
      a(7,8) = (3.0_dp, -5.0_dp)
      a(10,1) = (0.0_dp, 1.0_dp)
    else
      call random_number(tmp)
      call random_number(tmp2)
      a = a + cmplx(tmp, tmp2)
    end if

    ! initialize b matrix
    b = cone
    if (.not. init_random_) then
      b(1,1) = (5.0_dp, 0.0_dp)
      b(5,2) = (1.0_dp, 1.0_dp)
      b(6,1) = (2.0_dp, 1.0_dp)
      b(8,2) = (1.0_dp, 3.0_dp)
      b(10,1) = (0.0_dp, -1.0_dp)
    else
      call random_number(tmp)
      call random_number(tmp2)
      b = b + cmplx(tmp, tmp2)
    end if

    ! copy a and b matrice to a2,b2 to save it for later comparison
    a2 = a
    b2 = b

    ! working array
    c = czero

    ! print input
    if (present(print_input)) then
      if (print_input) then
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
      end if
    end if

    ! clean up memory allocation
    deallocate(tmp, tmp2)

  end subroutine init_matrices


  subroutine init_matrices_batched(n, num_mult, a_array, a2_array, b_array, b2_array, c_array, timings)
    implicit none
    ! interface
    integer, intent(in) :: n, num_mult
    complex (kind=dp), allocatable :: a_array(:,:,:), a2_array(:,:,:), b_array(:,:,:), b2_array(:,:,:), c_array(:,:,:)
    real (kind=dp), allocatable :: timings(:)
    ! local
    complex (kind=dp), allocatable :: a(:,:), a2(:,:), b(:,:), b2(:,:), c(:,:)
    real (kind=dp), allocatable :: tmp(:, :), tmp2(:,:)
    integer :: irun
    logical, parameter :: init_random = .true. ! for debugging this can be set to false

    ! allocate memory and initialize arrays
    call init_matrices(n, 1, a, a2, b, b2, c, timings, print_input=.false., init_random=init_random)

    ! set array inputs
    if (.not. allocated(a_array)) allocate(a_array(n,n,num_mult), a2_array(n,n,num_mult), b_array(n,n,num_mult), b2_array(n,n,num_mult), c_array(n,n,num_mult))
    allocate(tmp(n,n), tmp2(n,n))
    do irun = 1, num_mult
      if (init_random) then
        call random_number(tmp)
        call random_number(tmp2)
        a = a + cmplx(tmp, tmp2)
        call random_number(tmp)
        call random_number(tmp2)
        b = b + cmplx(tmp, tmp2)
      end if
      a_array(:,:, irun) = a(:,:)
      a2_array(:,:, irun) = a(:,:)
      b_array(:,:, irun) = b(:,:)
      b2_array(:,:, irun) = b(:,:)
    end do

    ! cleanup temporary working arrays
    deallocate(tmp, tmp2, a, a2, b, b2, c)

  end subroutine init_matrices_batched

end module mod_matrix_tools
