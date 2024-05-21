//========================================================
//Program to solve a 6 ODE system using Runge-Kutta Method
//Output is written in file sr.dat
//========================================================
// Compile with the commands:
// gfortran -c rksuite/rksuite.f
// g++ sr.cpp sr_B.cpp rksuite.o -o rk3 -lgfortran
//--------------------------------------------------------
#include "sr.h"
double k1,k2,k3,k4;
extern "C" {
  void setup_(const int& NEQ,
              double& TSTART,double* YSTART,double& TEND,
              double& TOL   ,double* THRES ,
              const int& METHOD, const char& TASK,
              bool  & ERRASS,double& HSTART,double* WORK,
              const int& LENWRK,       bool& MESSAGE);
  void ut_( void f(double& t,double* Y, double* YP),
            double& TWANT, double& TGOT, double* YGOT,
            double* YPGOT, double* YMAX, double* WORK,
            int& UFLAG);
}
//--------------------------------------------------------
int main(){
  string buf;
  double T0,TF,X10,X20,X30,V10,V20,V30;
  double P10,P20,P30;
  double P1,P2,P3,V1,V2,V3;
  double t,dt,tstep;
  int    STEPS, i;
  // rksuite variables:
  double TOL,THRES[NEQ],WORK[LENWRK],HSTART;
  double Y[NEQ],YMAX[NEQ],YP[NEQ],YSTART[NEQ];
  bool   ERRASS, MESSAGE;
  int    UFLAG;
  const char TASK = 'U';
  //Input:
  cout << "Runge-Kutta Method for 6-ODEs Integration\n";
  cout << "Special Relativistic Particle:\n";
  cout << "Enter coupling constants k1,k2,k3,k4:\n";
  cin  >> k1 >> k2 >> k3 >> k4;getline(cin,buf);
  cout << "Enter STEPS,T0,TF,X10,X20,X30,V10,V20,V30:\n";
  cin  >> STEPS >> T0  >> TF
       >> X10   >> X20 >> X30
       >> V10   >> V20 >> V30;getline(cin,buf);
  momentum(V10,V20,V30,P10,P20,P30);
  cout << "No. Steps= " << STEPS << endl;
  cout << "Time: Initial T0 =" << T0
       << " Final TF="         << TF  << endl;
  cout << "           X1(T0)=" << X10
       << " X2(T0)="           << X20
       << " X3(T0)="           << X30 << endl;
  cout << "           V1(T0)=" << V10
       << " V2(T0)="           << V20
       << " V3(T0)="           << V30 << endl;
  cout << "           P1(T0)=" << P10
       << " P2(T0)="           << P20
       << " P3(T0)="           << P30 << endl;
  //Initial Conditions:
  dt        = (TF-T0)/STEPS;
  YSTART[0] = X10;
  YSTART[1] = X20;
  YSTART[2] = X30;
  YSTART[3] = P10;
  YSTART[4] = P20;
  YSTART[5] = P30;
  //Set control parameters:
  TOL = 5.0e-6;
  for( i = 0; i < NEQ; i++)
    THRES[i] = 1.0e-10;
  MESSAGE = true;
  ERRASS  = false;
  HSTART  = 0.0;
  //Initialization:
  setup_(NEQ,T0,YSTART,TF,TOL,THRES,METHOD,TASK,
         ERRASS,HSTART,WORK,LENWRK,MESSAGE);
  ofstream  myfile("sr.dat");
  myfile.precision(16);
  myfile << T0        << " "
         << YSTART[0] << " " << YSTART[1] << " "
         << YSTART[2] << " "
         << V1        << " " << V2        << " "
         << V3        << " "
         << energy(T0,YSTART)             << " "
         << YSTART[3] << " " << YSTART[4] << " "
         << YSTART[5] << '\n';
  //The calculation:
  for(i=1;i<=STEPS;i++){
    t = T0 + i*dt;
    ut_(f,t,tstep,Y,YP,YMAX,WORK,UFLAG);
    if(UFLAG > 2) break; //error: break the loop and exit
    velocity(Y[3],Y[4],Y[5],V1,V2,V3);
    myfile << tstep << " "
           << Y[0]  << " " << Y[1] << " "
           << Y[2]  << " " 
           << V1    << " " << V2   << " "
           << V3    << " "
           << energy(T0,Y)         << " "
           << Y[3]  << " " << Y[4] << " "
           << Y[5]  << " " << '\n';
  }
  myfile.close();
}// main()
//========================================================
//momentum -> velocity  transformation
//========================================================
void velocity(const double& p1,const double& p2,
              const double& p3,
                    double& v1,      double& v2,
                    double& v3){
  double psq;
  psq = p1*p1+p2*p2+p3*p3;
      
  v1  = p1/sqrt(1.0+psq);
  v2  = p2/sqrt(1.0+psq);
  v3  = p3/sqrt(1.0+psq);
}
//========================================================
//velocity -> momentum transformation
//========================================================
void momentum(const double& v1,const double& v2,
              const double& v3,
                    double& p1,      double& p2,
                    double& p3){
  double vsq;
  vsq = v1*v1+v2*v2+v3*v3;
  if(vsq >= 1.0){cerr << "momentum: vsq>=1\n";exit(1);}
  p1  = v1/sqrt(1.0-vsq);
  p2  = v2/sqrt(1.0-vsq);
  p3  = v3/sqrt(1.0-vsq);
  
    
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

