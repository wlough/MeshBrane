//===================================================
//Function to produce random numbers distributed
//according to the gaussian distribution
//g(x) = 1/(sigma*sqrt(2*pi))*exp(-x^2/(2*sigma^2))
//===================================================
#include <cmath>
double drandom ();
double gaussran(){
  const  double sigma = 1.0;
  const  double PI2   = 6.28318530717958648;
  static bool   newx  = true;
  static double x;
  double        r,phi;

  if(newx){
    newx =  false;
    r    =  drandom();
    phi  =  drandom()*PI2;
    r    =  sigma*sqrt(-2.0*log(r));
    x    =  r*cos(phi);
    return  r*sin(phi);
  }else{
    newx =  true;
    return x;
  }
}//gaussran();
//===================================================
// #include <iostream>
// #include <fstream>
// #include <iomanip>
// using namespace std;
// int main(){
//   for(int i=1;i<=100000;i++)
//     cout << gaussran() << '\n';
// }
//  ---------------------------------------------------------------------
//  Copyright by Konstantinos N. Anagnostopoulos (2004-2014)
//  Physics Dept., National Technical University,
//  konstant@mail.ntua.gr, www.physics.ntua.gr/~konstant
//  
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, version 3 of the License.
//  
//  This program is distributed in the hope that it will be useful, but
//  WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  General Public License for more details.
//  
//  You should have received a copy of the GNU General Public Liense along
//  with this program.  If not, see <http://www.gnu.org/licenses/>.
//  -----------------------------------------------------------------------

