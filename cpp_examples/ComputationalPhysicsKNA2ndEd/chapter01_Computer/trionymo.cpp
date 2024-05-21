// =============================================================
// Program to compute roots of a 2nd order polynomial
// Tasks: Input from user ,logical statements,
//        use of functions,exit
//  
// Tests: a,b,c= 1  2  3 D=   -8
//        a,b,c= 1 -8 16 D=    0   x1=   4
//        a,b,c= 1 -1 -2 D=    9.  x1=   2.  x2=  -1.
//        a,b,c= 2.3 -2.99 -16.422 x1=   3.4 x2=  -2.1
// =============================================================
#include <iostream>
#include <cstdlib>
#include <cmath>
using namespace std;

double Discriminant(double a, double b, double c);
void   roots(double a, double b, double c, double& x1,
	     double& x2);

int main(){
  double a,b,c,D;
  double x1,x2;

  cout << "Enter a,b,c: ";
  cin  >> a >> b >> c;
  cout << a << " " << b << " " << c << " " << '\n';
  
  // Test if we have a well defined polynomial of 2nd degree:  
  if( a == 0.0 ){
    cerr << "Trionymo: a=0\n";
    exit(1);
  }

  // Compute the discriminant (= diakrinousa)
  D = Discriminant(a,b,c);
  cout << "Discriminant: D=  " << D << '\n';
  
  // Compute the roots in each case: D>0, D=0, D<0 (no roots)
  if     (D >  0.0) {
    roots(a,b,c,x1,x2);
    cout << "Roots:  x1= " << x1 << " x2= "<< x2 << '\n';
  }
  else if(D == 0.0) {
    roots(a,b,c,x1,x2);
    cout << "Double Root:  x1= " << x1 << '\n';
  }
  else{
    cout << "No real roots\n";
    exit(1);
  }
  
} 
// =============================================================
// This is the function that computes the discriminant
// A function returns a value. This value is returned using
// the return statement
// =============================================================
double Discriminant(double a,double b,double c){
  return b * b - 4.0 * a * c;
}
// =============================================================
// The subroutine that computes the roots.
// a,b,c are passed by value: Their values cannot change within
//                            the function
// x1,x2 are passed by reference: Their values DO change within
//                            the function
// =============================================================
void  roots(double a,double b,double c, double& x1,double& x2){
  double D;

  D = Discriminant(a,b,c);
  if(D >= 0.0){
    D = sqrt(D);
  }else{
    cerr << "roots: Sorry, cannot compute roots, D<0=" << D << '\n';
  }

  x1 = (-b + D)/(2.0*a);
  x2 = (-b - D)/(2.0*a);
}
