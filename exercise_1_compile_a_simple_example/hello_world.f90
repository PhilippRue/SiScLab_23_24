!! Hello_World.f90
subroutine Print_Hello_World()
  integer :: i
  do i = 1, 5
     print *, "hello world"
  end do
end subroutine Print_Hello_World

program hello

  call print_hello_world()

end program hello
