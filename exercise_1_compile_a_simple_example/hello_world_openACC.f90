!! Hello_World_OpenACC.f90
subroutine Print_Hello_World()
  integer :: i
  !$acc kernels
  do i = 1, 5
    print *, "hello world"
  end do
 !$acc end kernels
end subroutine Print_Hello_World

program hello

  call print_hello_world()

end program hello
