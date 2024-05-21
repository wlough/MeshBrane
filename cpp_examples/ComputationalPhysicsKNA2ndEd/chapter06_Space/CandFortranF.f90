!----------------------------------------
real(8) function make_array1(v,N)
 implicit none
 integer N,i
 real(8) v(N)

 do i=1,N
  v(i) = i
 end do

 make_array1 = -11.0 ! a return value
 
end function make_array1
!----------------------------------------
!----------------------------------------
real(8) function make_array2(A,N,M)
 implicit none
 integer N,M,i,j
 real(8) A(M,N) ! Careful: N and M are interchanged!!

 do i=1,M
  do j=1,N
  !A(i,j) = i.j, e.g. A(2,3)=2.3
   A(i,j) = i + j/10.0  
  end do
 end do

 make_array2 = -22.0 ! a return value
 
end function make_array2
!----------------------------------------
