//=========================================================
//Program to integrate equations of motion for accelerations
//which are functions of x with the method of Euler, 
//Euler-Cromer and Euler-Verlet.
//The user sets initial conditions and the functions return
//X[t] and V[t]=dX[t]/dt in arrays
//T[0..Nt-1],X[0..Nt-1],V[0..Nt-1]
//The user provides number of times Nt and the final
//time Tfi. Initial time is assumed to be t_i=0 and the
//integration step h = Tfi/(Nt-1) 
//The user programs a real function accel(x) which gives the 
//acceleration  dV(t)/dt as function of X.
//NOTE: T[0] = 0 T[Nt-1] = Tfi
//=========================================================
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;
//--------------------------------------------------------
const int P = 110000;
double T[P], X[P], V[P];
//--------------------------------------------------------
void euler       (const double& Xin, const double& Vin,
                  const double& Tfi, const int   & Nt );
void euler_cromer(const double& Xin, const double& Vin,
                  const double& Tfi, const int   & Nt );
void euler_verlet(const double& Xin, const double& Vin,
                  const double& Tfi, const int   & Nt );
double accel     (const double& x  );
//--------------------------------------------------------
int main(){
  double Xin, Vin, Tfi;
  int    Nt , i;
  string buf;
  //The user provides initial conditions X_0,V_0
  //final time t_f and Nt:
  cout << "Enter X_0,V_0,t_f,Nt (t_i=0):\n";
  cin  >> Xin >> Vin >> Tfi >> Nt; getline(cin,buf);
  //This check is necessary in order to avoid  
  //memory access violations:
  if(Nt>=P){cerr << "Error: Nt>=P\n";exit(1);}
  //Xin= X[0], Vin=V[0], T[0]=0 and the routine gives
  //evolution in T[1..Nt-1], X[1..Nt-1], V[1..Nt-1]
  //which we print in a file
  euler(Xin,Vin,Tfi,Nt);
  ofstream myfile("euler.dat");
  myfile.precision(17);
  for(i=0;i<Nt;i++)
    //Each line in file has time, position, velocity:
    myfile << T[i] << " " << X[i] << " " << V[i] << endl;
  myfile.close();//we close the stream to be reused below
  //------------------------------------
  //We repeat everything for each method
  euler_cromer(Xin,Vin,Tfi,Nt);
  myfile.open("euler_cromer.dat");
  for(i=0;i<Nt;i++)
    myfile << T[i] << " " << X[i] << " " << V[i] << endl;
  myfile.close();
  //------------------------------------
  euler_verlet(Xin,Vin,Tfi,Nt);
  myfile.open("euler_verlet.dat");
  for(i=0;i<Nt;i++)
    myfile << T[i] << " " << X[i] << " " << V[i] << endl;
  myfile.close();
}//main()
//=========================================================
//Function which returns the value of acceleration at
//position x used in the integration functions
//euler, euler_cromer and euler_verlet
//=========================================================
double accel(const double& x){
  return -10.0 * sin(x);
}
//=========================================================
//Driver routine for integrating equations of motion 
//using the Euler method
//Input:
//Xin=X[0], Vin=V[0] -- initial condition at t=0,
//Tfi the final time and Nt the number of times
//Output:
//The arrays T[0..Nt-1], X[0..Nt-1], V[0..Nt-1] which 
//gives x(t_k)=X[k-1], dx/dt(t_k)=V[k-1], t_k=T[k-1] k=1..Nt
//where for k=1 we have the initial condition.
//=========================================================
void euler       (const double& Xin, const double& Vin,
                  const double& Tfi, const int   & Nt ){
  int    i;
  double h;
  //Initial conditions set here: 
  T[0] = 0.0;
  X[0] = Xin;
  V[0] = Vin;
  //h is the time step Dt
  h    = Tfi/(Nt-1);
  for(i=1;i<Nt;i++){
    T[i] = T[i-1]+h;       //time advances by Dt=h
    X[i] = X[i-1]+V[i-1]*h;//advancement and storage of 
    V[i] = V[i-1]+accel(X[i-1])*h;//position and velocity
  }
}//euler()
//=========================================================
//Driver routine for integrating equations of motion 
//using the Euler-Cromer method
//Input:
//Xin=X[0], Vin=V[0] -- initial condition at t=0,
//Tfi the final time and Nt the number of times
//Output:
//The arrays T[0..Nt-1], X[0..Nt-1], V[0..Nt-1] which 
//gives x(t_k)=X[k-1], dx/dt(t_k)=V[k-1], t_k=T[k-1] k=1..Nt
//where for k=1 we have the initial condition.
//=========================================================
void euler_cromer(const double& Xin, const double& Vin,
                  const double& Tfi, const int   & Nt ){
  int    i;
  double h;
  //Initial conditions set here: 
  T[0] = 0.0;
  X[0] = Xin;
  V[0] = Vin;
  //h is the time step Dt
  h    = Tfi/(Nt-1);
  for(i=1;i<Nt;i++){
    T[i] = T[i-1]+h;    
    V[i] = V[i-1]+accel(X[i-1])*h;   
    X[i] = X[i-1]+V[i]*h; //note difference from Euler
  }

}//euler_cromer()
//=========================================================
//Driver routine for integrating equations of motion 
//using the Euler-Verlet method
//Input:
//Xin=X[0], Vin=V[0] -- initial condition at t=0,
//Tfi the final time and Nt the number of times
//Output:
//The arrays T[0..Nt-1], X[0..Nt-1], V[0..Nt-1] which 
//gives x(t_k)=X[k-1], dx/dt(t_k)=V[k-1], t_k=T[k-1] k=1..Nt
//where for k=1 we have the initial condition.
//=========================================================
void euler_verlet(const double& Xin, const double& Vin,
                  const double& Tfi, const int   & Nt ){
  int    i;
  double h,h2,X0,o2h;
  //Initial conditions set here: 
  T[0]     = 0.0;
  X[0]     = Xin;
  V[0]     = Vin;
  h        = Tfi/(Nt-1); //time step
  h2       = h*h;        //time step squared
  o2h      = 0.5/h;      //  h/2
  //We have to initialize one more step:
  //X0 corresponds to 'X[-1]'
  X0       = X[0] - V[0] * h + accel(X[0]) * h2 / 2.0;
  T[1]     = h;
  X[1]     = 2.0*X[0] - X0   + accel(X[0]) * h2;
  //Now i starts from 2:
  for(i=2;i<Nt;i++){
    T[i]   = T[i-1] + h;
    X[i]   = 2.0*X[i-1] - X[i-2] + accel(X[i-1])*h2;
    V[i-1] = o2h * (X[i]- X[i-2]);
  }
  //we have one more step for the velocity:
  V[Nt-1]  = (X[Nt-1] - X[Nt-2])/h;
}//euler_verlet()
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

