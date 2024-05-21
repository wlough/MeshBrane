#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;
//--------------------------------------------------------
void
eline (double  xin,double  yin,
       double*   X,double*   Y,double*  Q, const int N);
void
efield(double   x0,double   y0,
       double*   X,double*   Y,double*  Q, const int N,
       double&  Ex,double&  Ey);
void
mdist (double   x0,double   y0,
       double*   X,double*   Y,            const int N,
       double&  rm,double&  rM);
//--------------------------------------------------------
int main(){
  string buf;
  const int     P = 20;  //max number of charges
  double        X[P], Y[P], Q[P];
  int           N;
  int           i,j,draw;
  double        x0,y0;
  //-------------  SET CHARGE DISTRIBUTION ----
  cout << "# Enter number of charges:"        << endl;
  cin  >> N;                         getline(cin,buf);
  cout << "# N= "        << N                 << endl;
  for(i=0;i<N;i++){
    cout << "# Charge: " << i+1               << endl;
    cout << "# Position and charge: (X,Y,Q):" << endl;
    cin  >> X[i] >> Y[i] >> Q[i];    getline(cin,buf);
    cout <<         "# (X,Y)= "
         << X[i] << " "
         << Y[i] << " Q= "
         << Q[i]                              << endl;
  }
  //-------------  DRAWING LINES  -------------
  cout << "# How many lines to draw?\n";
  cin  >> draw;                      getline(cin,buf);
  for(i=1;i<=draw;i++){
    cout << "# Initial point (x0,y0):\n";
    cin  >> x0 >> y0;                getline(cin,buf);
    eline(x0,y0,X,Y,Q,N);
  }
}//main()
//--------------------------------------------------------
void
eline (double  xin,double  yin,
       double*   X,double*   Y,double*  Q, const int N){
  const double step    =0.01;
  const double max_dist=20.0;
  int          i, direction;
  double       x0,y0;
  double       rmin,rmax,r,dx,dy,dl;
  double       Ex,Ey,E;

  cout.precision(17);
  for(direction=-1;direction<=1;direction+=2){
    dl = direction * step;
    x0 = xin;
    y0 = yin;
    dx = 0.0;
    dy = 0.0;
    mdist(x0,y0,X,Y,N,rmin,rmax);
    while(rmin > (2.0*step) && rmax < max_dist){
      cout << x0 << " " << y0 << '\n';
      //We evaluate the E-field at the midpoint: 
      //This reduces systematic errors 
      efield(x0+0.5*dx,y0+0.5*dy,X,Y,Q,N,Ex,Ey);
      E  = sqrt(Ex*Ex+Ey*Ey);
      if( E <= 1.0e-10 ) break;
      dx = dl*Ex/E;
      dy = dl*Ey/E;
      x0 = x0 + dx;
      y0 = y0 + dy;
      mdist(x0,y0,X,Y,N,rmin,rmax);
    }//while(rmin > (2.0*step) && rmax < max_dist)
  }//for(direction=-1;direction<=1;direction+=2)
}//eline()
//--------------------------------------------------------
void
efield(double   x0,double   y0,
       double*   X,double*   Y,double*  Q, const int N,
       double&  Ex,double&  Ey){
  int    i;
  double r3,xi,yi;
  Ex  = 0.0;
  Ey  = 0.0;
  for(i=0;i<N;i++){
    xi = x0-X[i];
    yi = y0-Y[i];
    r3 = pow(xi*xi+yi*yi,-1.5);
    Ex = Ex + Q[i]*xi*r3;
    Ey = Ey + Q[i]*yi*r3;
  }
}//efield()
//--------------------------------------------------------
void
mdist (double     x0,double     y0,
       double*     X,double*     Y,       const int N,
       double&  rmin,double&  rmax){
  int    i;
  double r;
  rmax = 0.0;
  rmin = 1000.0;
  for(i=0;i<N;i++){
    r = sqrt((x0-X[i])*(x0-X[i]) + (y0-Y[i])*(y0-Y[i]));
    if(r > rmax) rmax = r;
    if(r < rmin) rmin = r;
  }
}//mdist()
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
