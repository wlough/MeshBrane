//========================================================
//Program to solve Damped Linear Oscillator
//using 4th order Runge-Kutta Method
//Output is written in file dlo.dat
//========================================================
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;
//--------------------------------------------------------
const int P = 110000;
double T[P], X1[P], X2[P];
double omega_0,omega,gam,a_0,omega_02,omega2;
//--------------------------------------------------------
double
f1(const double& t  , const double& x1, const double& x2);
double
f2(const double& t  , const double& x1, const double& x2);
void
RK(const double& Ti , const double& Tf, const double& X10,
   const double& X20, const int   & Nt);
void
RKSTEP(double& t, double& x1, double& x2,
       const double& dt);
//--------------------------------------------------------
int main(){
  double Ti,Tf,X10,X20;
  double Energy;
  int    Nt;
  int     i;
  string buf;

  //Input:
  cout << "Runge-Kutta Method for DLO Integration\n";
  cout << "Enter omega_0, omega, gamma, a_0:\n";
  cin  >> omega_0>> omega>> gam>> a_0;getline(cin,buf);
  omega_02 = omega_0*omega_0;
  omega2   = omega  *omega;
  cout << "omega_0= " << omega_0
       << "  omega= " << omega         << endl;
  cout << "gamma=   " << gamma
       << "  a_0=   " << a_0           << endl;
  cout << "Enter Nt,Ti,TF,X10,X20:"    << endl;
  cin  >> Nt >> Ti >> Tf >> X10 >> X20;getline(cin,buf);
  cout << "Nt = "               << Nt  << endl;
  cout << "Time: Initial Ti = " << Ti
       << " Final Tf = "        << Tf  << endl;
  cout << "           X1(Ti)= " << X10
       << " X2(Ti)= "           << X20 << endl;
  if(Nt >= P){cerr << "Error! Nt >= P\n";exit(1);}
  //Calculate:
  RK(Ti,Tf,X10,X20,Nt);
  //Output:
  ofstream myfile("dlo.dat");
  myfile.precision(17);
  myfile << "# Damped Linear Oscillator - dlo\n";
  myfile << "# omega_0= " << omega_0 << " omega= " << omega
         << "    gamma= " << gam     << "   a_0= " << a_0 << endl;
  for(i=0;i<Nt;i++){
    Energy = 0.5*X2[i]*X2[i]+0.5*omega_02*X1[i]*X1[i];
    myfile << T [i]  << " "
           << X1[i]  << " "
           << X2[i]  << " "
           << Energy << '\n';
  }
  myfile.close();
}//main()
//========================================================
//The functions f1,f2(t,x1,x2) provided by the user
//========================================================
double
f1(const double& t, const double& x1, const double& x2){
  return x2;
}
//--------------------------------------------------------
double
f2(const double& t, const double& x1, const double& x2){
  double a;
  a = a_0*cos(omega*t);
  return -omega_02*x1-gam*x2+a; 
}
//========================================================
//RK(Ti,Tf,X10,X20,Nt) is the driver 
//for the Runge-Kutta integration routine RKSTEP
//Input: Initial and final times Ti,Tf
//       Initial values at t=Ti  X10,X20
//       Number of steps of integration: Nt-1
//Output: values in arrays T[Nt],X1[Nt],X2[Nt] where
//T[0]    = Ti X1[0]   = X10 X2[0] = X20
//             X1[k-1] = X1(at t=T(k))
//             X2[k-1] = X2(at t=T(k))
//T[Nt-1] = Tf
//========================================================
void
RK(const double& Ti , const double& Tf, const double& X10,
   const double& X20, const int   & Nt){
  double dt;
  double TS,X1S,X2S; //time and X1,X2 at given step
  int     i;
  //Initialize variables:
  dt      = (Tf-Ti)/(Nt-1);
  T [0]   = Ti;
  X1[0]   = X10;
  X2[0]   = X20;
  TS      = Ti;
  X1S     = X10;
  X2S     = X20;
  //Make RK steps: The arguments of RKSTEP are 
  //replaced with the new ones!
  for(i=1;i<Nt;i++){
    RKSTEP(TS,X1S,X2S,dt);
    T [i]  = TS;
    X1[i]  = X1S;
    X2[i]  = X2S;
  }
}//RK()
//========================================================
//Function RKSTEP(t,x1,x2,dt)
//Runge-Kutta Integration routine of ODE
//dx1/dt=f1(t,x1,x2) dx2/dt=f2(t,x1,x2)
//User must supply derivative functions:
//real function f1(t,x1,x2)
//real function f2(t,x1,x2) 
//Given initial point (t,x1,x2) the routine advances it
//by time dt.
//Input : Inital time t    and function values x1,x2
//Output: Final  time t+dt and function values x1,x2
//Careful!: values of t,x1,x2 are overwritten...
//========================================================
void
RKSTEP(double& t, double& x1, double& x2,
       const double& dt){
  double k11,k12,k13,k14,k21,k22,k23,k24;
  double h,h2,h6;
  
  h  =dt;    //h =dt, integration step
  h2 =0.5*h; //h2=h/2
  h6 =h/6.0; //h6=h/6
      
  k11=f1(t,x1,x2);
  k21=f2(t,x1,x2);
  k12=f1(t+h2,x1+h2*k11,x2+h2*k21);
  k22=f2(t+h2,x1+h2*k11,x2+h2*k21);
  k13=f1(t+h2,x1+h2*k12,x2+h2*k22);
  k23=f2(t+h2,x1+h2*k12,x2+h2*k22);
  k14=f1(t+h ,x1+h *k13,x2+h *k23);
  k24=f2(t+h ,x1+h *k13,x2+h *k23);

  t  =t+h;
  x1 =x1+h6*(k11+2.0*(k12+k13)+k14);
  x2 =x2+h6*(k21+2.0*(k22+k23)+k24);
  
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


