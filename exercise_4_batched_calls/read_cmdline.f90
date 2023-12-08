module mod_read_cmdline

contains

  subroutine get_mat_size_from_input(n, num_mult)

    implicit none
    integer :: n, num_mult
    ! command line options
    integer :: iarg
    character(100) :: optchar, optchar2

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

  end subroutine get_mat_size_from_input

end module mod_read_cmdline