//========================================================
//Program that computes scattering cross-section of a central
//force on the plane. The user should first check that the
//parameters used, lead to a free state in the end.
//  **  X20 is the impact parameter b  **
//A 4 ODE system is solved using Runge-Kutta Method
//User must supply derivatives 
//dx1/dt=f1(t,x1,x2,x3,x4) dx2/dt=f2(t,x1,x2,x3,x4) 
//dx3/dt=f3(t,x1,x2,x3,x4) dx4/dt=f4(t,x1,x2,x3,x4) 
//as real(8) functions 
//Output is written in file scatter.dat
//========================================================
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;
//--------------------------------------------------------
const int P = 1010000;
double T[P], X1[P], X2[P], V1[P], V2[P];
double k1,k2;
//--------------------------------------------------------
double
f1(const double& t  , const double& x1, const double& x2,
   const double& v1 , const double& v2);
double
f2(const double& t  , const double& x1, const double& x2,
   const double& v1 , const double& v2);
double
f3(const double& t  , const double& x1, const double& x2,
   const double& v1 , const double& v2);
double
f4(const double& t  , const double& x1, const double& x2,
   const double& v1 , const double& v2);
double
energy
  (const double& t  , const double& x1, const double& x2,
   const double& v1 , const double& v2);
void
RK(const double& Ti , const double& Tf ,
   const double& X10, const double& X20, 
   const double& V10, const double& V20,
   const int   & Nt);
void
RKSTEP(double& t ,
       double& x1, double& x2,
       double& x3, double& x4,
       const       double& dt);
//--------------------------------------------------------
int main(){
  string buf;
  double Ti,Tf,X10,X20,V10,V20;
  double X20F,dX20; //max impact parameter and step
  int    Nt,i ;
  const int Nbins=20;
  int index;
  double angle,bins[Nbins],Npart;
  const double PI     =3.14159265358979324;
  const double rad2deg=180.0/PI;
  const double dangle =PI/Nbins;
  double R,density,dOmega,sigma,sigmatot;
  
  //Input:
  cout << "Runge-Kutta Method for 4-ODEs Integration\n";
  cout << "Enter coupling constants:\n";
  cin  >> k1 >> k2;getline(cin,buf);
  cout << "k1= " << k1 << " k2= " << k2 << endl;
  cout << "Enter Nt,Ti,Tf,X10,X20,V10,V20:\n";
  cin  >> Nt >> Ti >> Tf>> X10 >> X20 >> V10 >> V20;
  getline(cin,buf);
  cout << "Enter final impact parameter X20F and step dX20:\n";
  cin  >> X20F >> dX20;
  cout << "Nt = " << Nt << endl;
  cout << "Time: Initial Ti = " << Ti
       << " Final Tf= "         << Tf   << endl;
  cout << "           X1(Ti)= " << X10
       << " X2(Ti)="            << X20  << endl;
  cout << "           V1(Ti)= " << V10
       << " V2(Ti)="            << V20  << endl;
  cout << "Impact par X20F  ="  << X20F
       << " dX20  ="            << dX20 << endl;
  ofstream myfile("scatter.dat");
  myfile.precision(17);
  for(i=0;i<Nbins;i++) bins[i] = 0.0;
  //Calculate:
  Npart  = 0.0;
  X20    = X20 + dX20/2.0; //starts in middle of first interval
  while( X20 < X20F ){
    RK(Ti,Tf,X10,X20,V10,V20,Nt);
    //Take absolute value due to symmetry:
    angle = abs(atan2(V2[Nt-1],V1[Nt-1]));
    //Output: The final angle. Check if almost constant
    myfile << "@ " << X20 << " " <<  angle
           << "  " << abs(atan2(V2[Nt-51],V1[Nt-51]))
           << "  " << k1/(V10*V10)/tan(angle/2.0) << endl;
    //Update histogram:
    index       = int(angle/dangle);
    //Number of incoming particles per unit time
    //is proportional to radius of ring
    //of radius X20, the impact parameter:
    bins[index] += X20 ; // db is cancelled from density
    Npart       += X20 ; // <-- i.e. from here
    X20         += dX20;
  }//while( X20 < X20F )
  //Print scattering cross section:
  R         = X20;             //beam radius
  density   = Npart/(PI*R*R);  //beam flux density J
  sigmatot  = 0.0;             //total cross section
  for(i=0;i<Nbins;i++){
    angle    = (i+0.5)*dangle;
    dOmega   = 2.0*PI*sin(angle)*dangle; //d(Solid Angle)
    sigma    = bins[i]/(density*dOmega);
    if(sigma>0.0)
      myfile << "ds= " << angle
             << " "    << angle*rad2deg
             << " "    << sigma << endl;
    sigmatot += sigma*dOmega;
  }//for(i=0;i<Nbins;i++)
  myfile << "sigmatot= " << sigmatot << endl;
  myfile.close();
}//main()
//========================================================
//The velocity functions f1,f2(t,x1,x2,v1,v2)
//========================================================
double
f1(const double& t , const double& x1, const double& x2,
   const double& v1, const double& v2){
  return v1;
}
//--------------------------------------------------------
double
f2(const double& t , const double& x1, const double& x2,
   const double& v1, const double& v2){
  return v2;
}
//========================================================
//RK(Ti,Tf,X10,X20,V10,V20,Nt) is the driver 
//for the Runge-Kutta integration routine RKSTEP
//Input: Initial and final times Ti,Tf
//       Initial values at t=Ti  X10,X20,V10,V20
//       Number of steps of integration: Nt-1
//       Size of arrays T,X1,X2,V1,V2
//Output: real arrays T[Nt],X1[Nt],X2[Nt],
//                          V1[Nt],V2[Nt] where
//T[0] = Ti X1[0] = X10 X2[0] = X20 V1[0] = V10 V2[0] = V20
//          X1[k] = X1(at t=T[k]) X2[k] = X2(at t=T[k])
//          V1[k] = V1(at t=T[k]) V2[k] = V2(at t=T[k])
//T[Nt-1]= Tf
//========================================================
void
RK(const double& Ti , const double& Tf ,
   const double& X10, const double& X20, 
   const double& V10, const double& V20,
   const int   & Nt){

  double dt;
  double TS,X1S,X2S; //values of time and X1,X2 at given step
  double    V1S,V2S;
  int i;
  //Initialize:
  dt     = (Tf-Ti)/(Nt-1);
  T [0]  = Ti;
  X1[0]  = X10; X2[0] = X20;
  V1[0]  = V10; V2[0] = V20;
  TS     = Ti;
  X1S    = X10; X2S   = X20;
  V1S    = V10; V2S   = V20;
  //Make RK steps: The arguments of RKSTEP are 
  //replaced with the new ones
  for(i=1;i<Nt;i++){
    RKSTEP(TS,X1S,X2S,V1S,V2S,dt);
    T [i] = TS;
    X1[i] = X1S; X2[i] = X2S;
    V1[i] = V1S; V2[i] = V2S;
  }
}//RK()
//========================================================
//Subroutine RKSTEP(t,x1,x2,dt)
//Runge-Kutta Integration routine of ODE
//dx1/dt=f1(t,x1,x2,x3,x4) dx2/dt=f2(t,x1,x2,x3,x4)
//dx3/dt=f3(t,x1,x2,x3,x4) dx4/dt=f4(t,x1,x2,x3,x4)
//User must supply derivative functions:
//real function f1(t,x1,x2,x3,x4)
//real function f2(t,x1,x2,x3,x4) 
//real function f3(t,x1,x2,x3,x4)
//real function f4(t,x1,x2,x3,x4) 
//Given initial point (t,x1,x2) the routine advances it
//by time dt.
//Input : Inital time t    and function values x1,x2,x3,x4
//Output: Final  time t+dt and function values x1,x2,x3,x4
//Careful: values of t,x1,x2,x3,x4 are overwritten...
//========================================================
void
RKSTEP(double& t ,
       double& x1, double& x2,
       double& x3, double& x4,
       const       double& dt){
  double k11,k12,k13,k14,k21,k22,k23,k24;
  double k31,k32,k33,k34,k41,k42,k43,k44;
  double h,h2,h6;

  h =dt;     // h  = dt, integration step
  h2=0.5*h;  // h2 = h/2
  h6=h/6.0;  // h6 = h/6
  
  k11=f1(t,x1,x2,x3,x4);
  k21=f2(t,x1,x2,x3,x4);    
  k31=f3(t,x1,x2,x3,x4);
  k41=f4(t,x1,x2,x3,x4);

  k12=f1(t+h2,x1+h2*k11,x2+h2*k21,x3+h2*k31,x4+h2*k41);
  k22=f2(t+h2,x1+h2*k11,x2+h2*k21,x3+h2*k31,x4+h2*k41);
  k32=f3(t+h2,x1+h2*k11,x2+h2*k21,x3+h2*k31,x4+h2*k41);
  k42=f4(t+h2,x1+h2*k11,x2+h2*k21,x3+h2*k31,x4+h2*k41);

  k13=f1(t+h2,x1+h2*k12,x2+h2*k22,x3+h2*k32,x4+h2*k42);
  k23=f2(t+h2,x1+h2*k12,x2+h2*k22,x3+h2*k32,x4+h2*k42);
  k33=f3(t+h2,x1+h2*k12,x2+h2*k22,x3+h2*k32,x4+h2*k42);
  k43=f4(t+h2,x1+h2*k12,x2+h2*k22,x3+h2*k32,x4+h2*k42);

  k14=f1(t+h ,x1+h *k13,x2+h *k23,x3+h *k33,x4+h *k43);
  k24=f2(t+h ,x1+h *k13,x2+h *k23,x3+h *k33,x4+h *k43);
  k34=f3(t+h ,x1+h *k13,x2+h *k23,x3+h *k33,x4+h *k43);
  k44=f4(t+h ,x1+h *k13,x2+h *k23,x3+h *k33,x4+h *k43);
  
  t =t+h;
  x1=x1+h6*(k11+2.0*(k12+k13)+k14);
  x2=x2+h6*(k21+2.0*(k22+k23)+k24);
  x3=x3+h6*(k31+2.0*(k32+k33)+k34);
  x4=x4+h6*(k41+2.0*(k42+k43)+k44);

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
