// Example of how to link C++ with Fortran programs:
// The main program is in C++ and calls Fortran functions
// and subroutines from the Fortran source code file CandFortranF.f90
// The main features of this test are:
//  1. The names of Fortran functions are declared within
//     extern "C" { ... }
//  2. The names of the Fortran functions are declared and used by
//     appending an underscore to their names. Arguments must be
//     passed by reference. Use only lowercase letters in their names,
//     even if in the Fortran source code they are declared with
//     capital letters.
//  3. Arrays in fortran are stored in column-major mode, whereas in
//     C++ in row major mode. Moreover, the arrays in fortran, e.g. a
//     A(N) array, are indexed from A(1) ... A(N), whereas in C,
//     an array A[N] from A[0] ... A[N-1]. Therefore
//     Fortran A(i,j) -> A[j-1][i-1] in C++
//  4. Compile the Fortran code using a Fortran compiler with the -c
//     flag. Then the .o (object) files are added in the C++
//     compiler command line, together with the C++ source code
//     files.
//  5. Fortran libraries must be linked when compiling with the C++
//     compiler. With g++/gfortran, this is done by adding
//     -lgfortran at the end of the compilation command.
//
// Compile using the commands:
// gfortran -c CandFortranF.f90
// g++ CandFortranC.cpp CandFortranF.o -o CandFortran -lgfortran
//
// Output: (notice the transposed way that the Fortran 2D array is
//          printed!)
//
// 1D array: Return value x= -11
// 1 2 3 4 
// -------------------------
// 2D array: Return value x= -22
// 1.1 2.1 3.1 
// 1.2 2.2 3.2 
// 1.3 2.3 3.3 
// 1.4 2.4 3.4 


#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>

using namespace std;

const int N=4, M=3;
//extern "C" is a linkage-specification
//extern "C" makes a function-name in C++ have 'C' linkage
//The same is done if you want to link C functions to C++ code.
extern "C" {
  double make_array1_(double v[N]   , const int& N);
  double make_array2_(double A[N][M], const int& N, const int& M);
}
int main(){
  double A[N][M], v[N];
  double x;

  //Make a  1D array using a fortran function:
  x = make_array1_(v,N);
  cout << "1D array: Return value x= " << x << endl;
  for(int i=0;i<N;i++)
    cout << v[i] << " ";
  cout << "\n-------------------------\n";
  //Make an 2D array using a fortran function:
  x = make_array2_(A,N,M);
  cout << "2D array: Return value x= " << x << endl;
  for(int i=0;i<N;i++){
    for(int j=0;j<M;j++)      //A is ... transposed!
      cout  << A[i][j] << " ";//A[i][j] = (j+1).(i+1), e.g. A[1][2] = 3.1
    cout << '\n';
  }
}
