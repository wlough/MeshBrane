//========================================================
//Program to solve an ODE system using the
//4th order Runge-Kutta Method
//NEQ: Number of equations
//User supplies two functions:
//f(t,x,xdot): with double  t,x[NEQ],xdot[NEQ] which
//given the time t and current values of functions x[NEQ]
//it returns the values of derivatives: xdot = dx/dt 
//The values of two coupling constants k1,k2 may be used
//in f which are read in the main program
//finit(NEQ) : sets the value of NEQ
//
//User Interface: 
//double k1,k2:  coupling constants in global scope
//Nt,Ti,Tf: Nt-1 integration steps, initial/final time
//double X0[NEQ]: initial conditions
//Output:
//rkA.dat with Nt lines consisting of: T[Nt],X[Nt][NEQ]
//========================================================
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;
//--------------------------------------------------------
double  *T;  // T[Nt]      stores the values of times
double **X;  // X[Nt][NEQ] stores the values of functions
double k1,k2;
//--------------------------------------------------------
void RK(const double& Ti, const double& Tf, double* X0,
        const int   & Nt, const int   & NEQ);
void RKSTEP(  double& t ,       double* x ,
        const double& dt, const int   & NEQ);
//The following functions are defined in rkA_XX.cpp:
void finit(   int   & NEQ); //Sets number of equations
//Derivative and energy functions:
void   f     (const double& t, double* X,double* dXdt);
double energy(const double& t, double* X);
//--------------------------------------------------------
int main(){
  string  buf;
  int     NEQ,Nt;
  double*  X0;
  double Ti,Tf;
  //Get number of equations and allocate memory for X0:
  finit(NEQ);
  X0 = new double [NEQ];
  //Input:
  cout << "Runge-Kutta Method for ODE Integration.\n";
  cout << "NEQ= " << NEQ << endl;
  cout << "Enter coupling constants:\n";
  cin  >> k1 >> k2;getline(cin,buf);
  cout << "k1= " << k1 << " k2= " << k2 << endl;
  cout << "Enter Nt, Ti, Tf, X0:\n";
  cin  >>        Nt>>Ti>>Tf;
  for(int i=0;i<NEQ;i++) cin >> X0[i];getline(cin,buf);
  cout << "Nt = " << Nt << endl;
  cout << "Time: Initial Ti =" << Ti << " " 
       << "Final Tf="          << Tf << endl;
  cout << "              X0 =";
  for(int i=0;i<NEQ;i++) cout  << X0[i] << " ";
  cout << endl;
  //Allocate memory for data arrays:
  T  = new double [Nt];
  X  = new double*[Nt];
  for(int i=0;i<Nt;i++) X[i] = new double[NEQ];
  //The Calculation:
  RK(Ti,Tf,X0,Nt,NEQ);
  //Output:
  ofstream  myfile("rkA.dat");
  myfile.precision(16);
  for(int i=0;i<Nt;i++){
    myfile   << T[i]      << " ";
    for(int jeq=0;jeq<NEQ;jeq++)
      myfile << X[i][jeq] << " ";
    myfile   << energy(T[i],X[i]) << '\n';
  }
  myfile.close();
  //------------------------------------------------------
  //Cleaning up dynamic memory: delete[] each array
  //created with the new operator (Not necessary in this
  //program,it is done at the end of the program anyway)
  delete[] X0;
  delete[] T ;
  for(int i=0;i<NEQ;i++) delete [] X[i];
  delete[] X ;
}//main()
//========================================================
//Driver of the RKSTEP routine
//========================================================
void RK(const double& Ti, const double& Tf, double* X0,
        const int   & Nt, const int   & NEQ){
  double  dt;
  double  TS;
  double* XS;

  XS     = new double[NEQ];
  //Initialize variables:
  dt     = (Tf-Ti)/(Nt-1);
  T [0]  = Ti;
  for(int ieq=0;ieq<NEQ;ieq++) X[0][ieq]=X0[ieq];
  TS     = Ti;
  for(int ieq=0;ieq<NEQ;ieq++) XS  [ieq]=X0[ieq];
  //Make RK steps: The arguments of RKSTEP are 
  //replaced with the new ones
  for(int i=1;i<Nt;i++){
    RKSTEP(TS,XS,dt,NEQ);
    T[i] = TS;
    for(int ieq=0;ieq<NEQ;ieq++) X[i][ieq] = XS[ieq];
  }
  // Clean up memory:
  delete [] XS;
}//RK()
//========================================================
//Function RKSTEP(t,X,dt)
//Runge-Kutta Integration routine of ODE
//========================================================
void RKSTEP(double& t, double* x, const double& dt ,
                                  const int   & NEQ){
  double  tt;
  double *k1, *k2, *k3, *k4, *xx;
  double  h,h2,h6;

  k1 = new double[NEQ];
  k2 = new double[NEQ];
  k3 = new double[NEQ];
  k4 = new double[NEQ];
  xx = new double[NEQ];
  
  h =dt;     // h =dt, integration step
  h2=0.5*h;  // h2=h/2
  h6=h/6.0;  // h6=h/6
  //1st step:
  f(t ,x ,k1);
  //2nd step:
  for(int ieq=0;ieq<NEQ;ieq++)
    xx[ieq] = x[ieq] + h2*k1[ieq];
  tt =t+h2;
  f(tt,xx,k2);
  //3rd step:
  for(int ieq=0;ieq<NEQ;ieq++)
    xx[ieq] = x[ieq] + h2*k2[ieq];
  tt =t+h2;
  f(tt,xx,k3);
  //4th step:
  for(int ieq=0;ieq<NEQ;ieq++)
    xx[ieq] = x[ieq] + h *k3[ieq];
  tt=t+h ;
  f(tt,xx,k4);
  //Update:
  t += h;
  for(int ieq=0;ieq<NEQ;ieq++)
    x[ieq] += h6*(k1[ieq]+2.0*(k2[ieq]+k3[ieq])+k4[ieq]);
  //Clean up memory:
  delete[] k1;
  delete[] k2;
  delete[] k3;
  delete[] k4;
  delete[] xx;
}//RKSTEP()
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
