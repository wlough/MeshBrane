//===========================================================
//file: well.cpp
//
//Computation of energy eigenvalues and eigenfunctions
//of a particle in an infinite well with V(-x)=V(x)
//
//Input:  energy: initial guess for energy
//        parity: desired parity of solution (+/- 1)
//        Nx-1  : Number of RK4 steps from x=0 to x=1
//Output: energy: energy eigenvalue
//        psi.dat: final psi[x]
//        all.dat: all   psi[x] for trial energies
//===========================================================
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;
//--------------------------------------------------------
const int P = 10000;
double energy;
//--------------------------------------------------------
double
normalization(
         double*psi,const double&  dx,const int   & Nx,
                    const int   &   P);
void
RKSTEP(  double&  t,      double&  x1,
         double& x2,
         const            double&  dt);
//--------------------------------------------------------
int main(){
  double dx,x,epsilon,de;
  double psi,psip,psinew,psiold;
  double norm;
  double array_psifinal[2*P+1],array_xstep[2*P+1];
  double *psifinal = array_psifinal+P;
  double *xstep    = array_xstep   +P;
  //psifinal points to (array_psifinal+P) and one can
  //use psifinal[-P]...psifinal[P]. Similarly for xstep
  int    parity,Nx,iter,i,node;
  string buf;
  //------ Input:
  cout << "Enter energy,parity,Nx:\n";
  cin  >> energy  >> parity >> Nx;     getline(cin,buf);
  if(Nx > P){cerr << "Nx>P\n";exit(1);}
  if(parity > 0) parity= 1;
  else           parity=-1;
  cout << "# #######################################\n";
  cout << "# Estart= " << energy
       << "  parity= " << parity  << endl;
  dx       =  1.0/(Nx-1);
  epsilon  = 1.0e-6;
  cout << "# Nx=  "    << Nx      << " dx = " << dx
       << " eps=  "    << epsilon << endl;
  cout << "# #######################################\n";
  //------ Calculate:
  ofstream myfile("all.dat");
  myfile.precision(17);
  cout  .precision(17);
  iter     = 0;
  psiold   = 0.0;
  psinew   = 1.0;
  de       = 0.1 * abs(energy);
  while(iter < 10000){
    // Initial conditions at x=0
    x      = 0.0;
    if(parity == 1){
      psi  = 1.0;
      psip = 0.0;
    }else{
      psi  = 0.0;
      psip = 1.0;
    }
    myfile   << iter  << " " << energy << " "
             << x     << " " << psi    << " "
             << psip  << endl;
    // Use Runge-Kutta to forward to x=1
    for(i=2;i<=Nx;i++){
      x = (i-2)*dx;
      RKSTEP(x,psi,psip,dx);
      myfile << iter  << " " << energy << " "
             << x     << " " << psi    << " "
             << psip  << endl;
    }
    psinew = psi;
    cout     << iter  << " " << energy << " "
             << de    << " " << psinew << endl;
    // Stop if value of psi close to 0
    if(abs(psinew)    <= epsilon) break;
    // Change direction of energy search:
    if(psinew*psiold  <  0.0    ) de = -0.5*de;
    energy   += de;
    psiold    = psinew;
    iter++;
  }//while(iter < 10000)
  myfile.close();
  //We found the solution:
  //calculate it once again and store it
  if(parity   == 1){
    psi       =  1.0;
    psip      =  0.0;
    node      =  0; //count number of nodes of function
  }else{
    psi       =  0.0;
    psip      =  1.0;
    node      =  1;
  }
  x           =  0.0;
  xstep   [0] =  x;
  psifinal[0] =  psi; //array that stores psi(x)
  psiold      =  0.0;
  // Use Runge-Kutta to move to x=1
  for(i=2 ; i<=Nx ; i++ ){
    x               = (i-2)*dx;
    RKSTEP(x,psi,psip,dx);
    xstep   [i-1]   = x;
    psifinal[i-1]   = psi;
    // Use parity to compute psi(-x)
    xstep   [1-i]   = -x;
    psifinal[1-i]   = psi*parity;
    // Determine zeroes of psi(x):
    // psi should not be zero within epsilon:
    if(abs(psi)     > 2.0*epsilon){
      if(psi*psiold < 0.0) node += 2;
      psiold        = psi;
    }
  }//for(i=2;i<=Nx;i++)
  node++;
  //print final solution:
  myfile.open("psi.dat");
  norm = 1.0/normalization(psifinal,dx,Nx,P);
  cout     << "Final result: E= " << energy
           << " n= "              << node
           << " parity= "         << parity
           << " norm = "          << norm         << endl;
  myfile   << "# E= "             << energy
           << " n= "              << node
           << " parity= "         << parity
           << " norm = "          << norm         << endl;
  for(i=-(Nx-1);i<=(Nx-1);i++)
    myfile << xstep[i]            << " "
           << norm*psifinal[i]                    << endl;
  myfile.close();
}// main()
//--------------------------------------------------------
//===========================================================
//Simpson's rule to integrate psi(x)*psi(x) for proper
//normalization. For n intervals of width dx (n even)
//Simpson's rule is:
//int(f(x)dx) = 
// (dx/3)*(f(x_0)+4 f(x_1)+2 f(x_2)+...+4 f(x_{n-1})+f(x_n))
//
//Input:   Discrete values of function psi[-Nx:Nx]
//         Array psi[-P:P] and its dimension P
//         Integration step dx
//Returns: sqrt( Integral(psi(x)psi(x) dx) )
//===========================================================
double normalization(double   *psi,const double& dx,
                     const int& Nx,const int   & P){
  int n,k,i;
  double integral;
  n = 2*(Nx-1); // the Simpson's rule number of intervals
  //zeroth order point:
  k = 0;
  i = k-Nx+1;
  integral    =       psi[i] * psi[i];
  //odd  order points:
  for(k=1 ; k<=(n-1);k+=2){
    i = k-Nx+1;
    integral += 4.0 * psi[i] * psi[i];
  }
  //even order points:
  for(k=2 ; k<=(n-2);k+=2){
    i = k-Nx+1;
    integral += 2.0 * psi[i] * psi[i];
  }
  //last point:
  k = n;
  i = k-Nx+1;
  integral   +=       psi[i] * psi[i];
  //measure normalization:
  integral   *=       dx/3.0;
  //final result:
  return sqrt(integral);
}
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
