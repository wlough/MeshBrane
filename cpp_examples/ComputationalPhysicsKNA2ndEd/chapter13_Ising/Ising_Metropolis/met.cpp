//============== met.cpp     ==================
#include "include.h"

void met(){
  int i,k,acc;
  int nn,snn,dE;

  acc=0;
  for(k=0;k<N;k++){
    i=N*drandom(mxmx);//pick a random site i
    //Sum of neighboring spins:
    if((nn=i+XNN)>= N) nn -= N; snn  = s[nn];
    if((nn=i-XNN)<  0) nn += N; snn += s[nn];
    if((nn=i+YNN)>= N) nn -= N; snn += s[nn];
    if((nn=i-YNN)<  0) nn += N; snn += s[nn];
    //change in energy/2
    dE = snn*s[i];
    //flip
    if(dE <=0 )                    {s[i] = -s[i];acc++;} //accept
    else if(drandom(mxmx)<prob[dE]){s[i] = -s[i];acc++;} //accept
  }//sweep ends
  acceptance += acc;
}
// Code from the book by:
// M.E.J. Newman and G.T. Barkema, Monte Carlo  Methods in Statistical Physics, Clarendon Press, Oxford (2002).
