//===========================================================
//file: schHOC.cpp
//
//Functions used in RKSTEP routine. Here:
//f1 = psip(x) = psi(x)'
//f2 = psip(x)'= psi(x)''
//
//One has to set:
//  1. V(x), the potential     
//  2. The boundary conditions for psi,psip at x=xmin and x=xmax
//
//===========================================================
//----- potential:
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;
extern double energy;
//-------- 
double V  (const double& x ){
  return x*x;
}//V()
//-------- boundary conditions:
void
boundary  (const double& xmin,const double& xmax    ,
                 double& psixmin,   double& psipxmin,
                 double& psixmax,   double& psipxmax){
  
  psixmin  =  exp(-0.5*xmin*xmin);
  psipxmin = -xmin*psixmin;
  psixmax  =  exp(-0.5*xmax*xmax);
  psipxmax = -xmax*psixmax;
  //Initial values at xmin and xmax
}
//-------- trivial function: derivative of psi      
double
f1(const double& x, const double& psi,const double& psip){
  return psip;
}
//-------- the second derivative of wavefunction:
//psip(x)' = psi(x)'' = -(E-V) psi(x)
double
f2(const double& x, const double& psi,const double& psip){
  //------- Schroedinger eq: RHS
  return (V(x)-energy)*psi;
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
