#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;
//---------------------------------------------
const int NEQ    = 6;
const int LENWRK = 32*NEQ;
const int METHOD = 2;
extern double k1,k2,k3,k4;
//---------------------------------------------
double energy(const double& t,       double* Y);
void   f(double& t, double* Y,       double* YP);
void velocity(const double& p1,const double& p2,
              const double& p3,
                    double& v1,      double& v2,
                    double& v3);
void momentum(const double& v1,const double& v2,
              const double& v3,
                    double& p1,      double& p2,
                    double& p3);
//---------------------------------------------
