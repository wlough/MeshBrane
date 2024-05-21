//========================================================
//Subroutine RKSTEP(t,x1,x2,dt)
//Runge-Kutta Integration routine of ODE
//dx1/dt=f1(t,x1,x2) dx2/dt=f2(t,x1,x2)
//User must supply derivative functions:
//real(8) function f1(t,x1,x2)
//real(8) Function f2(t,x1,x2) 
//Given initial point (t,x1,x2) the routine advnaces it
//by time dt.
//Input : Inital time t    and function values x1,x2
//Output: Final  time t+dt and function values x1,x2
//Careful: values of t,x1,x2 are overwritten...
//=======================================================
double
f1(const double&  x,const double& psi,const double& psip);
double
f2(const double&  x,const double& psi,const double& psip);
//-------------------------------------------------------
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
