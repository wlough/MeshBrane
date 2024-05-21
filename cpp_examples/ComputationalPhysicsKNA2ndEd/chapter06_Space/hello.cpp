// Compile and run:
// > gfortran -c helloF.f90
// > g++ hello.cpp helloF.o -o hello -lgfortran
// > ./hello
#include <iostream>
using namespace std;
//extern "C" is a linkage-specification
//extern "C" makes a function-name in C++ have 'C' linkage
//The same is done if you want to link C functions to C++ code.
extern "C" void   hello_();
extern "C" double squaremydouble_(double& x);
int main(){
  double x,x2;
  hello_();
  x  = 2.0;
  x2 = squaremydouble_(x);
  cout << "x   = "         << 2.0 << endl;
  cout << "2 x = "         << x   << endl;
  cout << "(2 x)*(2 x) = " << x2  << endl;
}
