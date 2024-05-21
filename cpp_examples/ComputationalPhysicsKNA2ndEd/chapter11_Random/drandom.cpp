//====================================================
//File: drandom.cpp
//Implementation of the Schrage algorithm for a 
//portable modulo generator for 32 bit signed integers
//(from numerical recipes)
//
//returns uniformly distributed pseudorandom numbers
// 0.0 < x < 1.0 (0 and 1 excluded)
//period: 2**31-2 = 2 147 483 646
//====================================================
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;
static  int seed = 323412; 
double drandom(){
  const int    a = 16807;     //a = 7**5
  const int    m = 2147483647;//m = a*q+r = 2**31-1
  const int    q = 127773;    //q = [m/a]
  const int    r = 2836;      //r = m % a 
  const double f = 1.0/m;
  int     p;
  double dr;

 compute:
  p    = seed/q;              //p = [seed/q]
  //seed = a*(seed % q)-r*[seed/q] = (a*seed) % m
  seed = a*(seed-q*p) - r*p;
  if(seed < 0) seed +=m;
  dr=f*(seed-1);
  if(dr <= 0.0 || dr >= 1.0) goto compute;
  return dr;
}
//===================================================
//small test program to see period
//===================================================
// int main(){
//   double x;
//   int    seed0;
//   long   i;
//
//   seed0 = 582375326;
//   seed  = seed0;
//   for(i=1;i<=3*2147483647l;i++){
//     x = drandom();
//     if(x == 0.0)
//       cout << "SOS: i,x=0 "<< i << " " << x << endl;
//     if(x <  0.0)
//       cout << "SOS: i,x<0 "<< i << " " << x << endl;
//     if(x == 1.0)
//       cout << "SOS: i,x=1 "<< i << " " << x << endl;
//     if(x >  1.0)
//       cout << "SOS: i,x>1 "<< i << " " << x << endl;
//     if(seed == seed0)
//       cout << "PERIOD: i,seed= "
//            << i << " " << seed << endl;
//   }
// }//main()
