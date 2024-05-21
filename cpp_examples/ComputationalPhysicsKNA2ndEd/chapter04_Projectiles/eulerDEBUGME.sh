#!/bin/bash
# 
# Script to run euler methods for several values of stepseu
#


X0=0.2
V0=0.0
tf=6.0
maxsteps=100001


cdir=`pwd`
prog=`basename $0`
#-------------------------------------------------------------------------------------------
# If you want newlines, you have to echo "$USAGE" and not echo $USAGE
USAGE="\
Usage: $prog [-n] [-x <X_0>] [-v <V_0>] [-t <t_f>] <steps>
 -n: Do not plot the simple pendulum solution
 Run euler for different values of number of steps (<$maxsteps)"
#-------------------------------------------------------------------------------------------
while getopts :T:hx:v:t:n opt_char 
do
 case $opt_char in
 x)
  X0=$OPTARG
  ;;
 v)
  V0=$OPTARG
  ;;
 t)
  tf=$OPTARG
  ;;
 n)
  exact_sol=$OPTARG
  ;;
 h)
  echo "$USAGE"
  exit 1
  ;;
 \?)
  echo "$OPTARG is not a valid option"
  echo "$USAGE"
  exit 1
  ;;
 esac
done
#-------------------------------------------------------------------------------------------
acalc     () { awk  -v OFMT="%.17g" "BEGIN{print $* }"      ; } # Works as function: pi2=`acalc "2 * $pi"`
extension () { echo     "$*" | awk -F'.'   'NF>1{print $NF}'; } # extension file.tar.gz = gz
filename  () { basename "$*"              ".${f##*.}"       ; } # filename  file.tar.gz = file.tar
#-------------------------------------------------------------------------------------------
# ~/lib/bash/hello.sh
# f=/path/to/fn.tar.gz;b=$(basename $f);d=$(dirname $f);gz=$(extension $f);fn=$(filename "$f"); fn=$(basename $f .$(extension $f));
#                                                       gz=${f##*.};                            fn=$(basename $f .${f##*.})       ;fnpure=$(basename $f .${f#*.});
#-------------------------------------------------------------------------------------------
# a=({1..7});echo "na= ${#a[@]} a[0]= ${a[0]} a[2]= ${a[2]}  a= ${a[@]}"
#-------------------------------------------------------------------------------------------
# for i in "${a[@]}";do echo $i;done
# for i in {1..10};do echo $i;done;
# for (( i=1 ; i<=10 ; i++ ));do echo $i;done;
# i=1;while true;do i=$((i+1));echo $i;if(( $i == 5 ));then break;fi;done
#-------------------------------------------------------------------------------------------
# if [         ! -e $f  ];then echo "$f does not exist";fi    (CAREFUL: spaces in if [ .. ];)
# if [[ -f $f && -x $f ]];then echo "$f is executable ";fi
# grep string file > /dev/null;
# if [ $? -eq 0         ];then echo found;else echo not found;fi
# if (( $X < $Y ));then echo "X<Y";fi
# if [ -d $f ];then echo "$f directory";elif [ -x $f ];then echo "$f executable";else echo bummer;fi
#-------------------------------------------------------------------------------------------
if (( $# < 1 ));then
 steps=(100 500 1000 2000 5000 7000 9999)
else
 steps=($@)
fi

for p in euler;do
 if [ -f $p.cpp ];then
  g++ -O2 $p.cpp -o $p
 else
  echo "${prog}: $p.cpp not found. Exiting"
  exit 1
 fi
done



# cleaning old data files
if [ -f euler.all        ];then \rm -f euler.all       ;fi
if [ -f euler_cromer.all ];then \rm -f euler_cromer.all;fi
if [ -f euler_verlet.all ];then \rm -f euler_verlet.all;fi
# preparing gnuplot plot commands for STEPS plots
 pltx_e="plot "
pltx_ec="plot "
pltx_ev="plot "
 pltv_e="plot "
pltv_ec="plot "
pltv_ev="plot "
# Now running for each value of STEPS:
let scnt=0 # count index in $steps in the loopt
for s in ${steps[@]};do
 if (( $s >= $maxsteps ));then
  echo "${prog}: steps= $s is >= $maxsteps. Skipping...."
  continue
 fi
 echo -n "Working on X0= $X0 V0= $V0 tf= $tf  STEPS= $s ...."
 # Running the program:
 ./euler <<EOF > /dev/null
$X0 $V0 $tf $s
EOF


# Making the plots:
 gnuplot -persist << EOF
set dummy t  #t is the independent variable in functions
set style data lines
omega2  = 10  #This is g/l in program
X0      = $X0 #Amplitute of oscillation
V0      = $V0 #Velocity at t0
omega   = sqrt(omega2)
x(t)    = X0 * cos(omega * t) +(V0/omega)*sin(omega*t)
v(t)    = V0 * cos(omega * t) -(omega*X0)*sin(omega*t)

set term x11 1
set title "Positions. STEPS= $s"
set ylabel "{/Symbol q}"
set xlabel "t"
plot "euler.dat" using 1:2  title "Euler", \
     "euler_cromer.dat" using 1:2  title "Euler-Cromer", \
     "euler_verlet.dat" using 1:2  title "Euler-Verlet", \
     x(t) title "{/Symbol q}_s(t)"
set term postscript enhanced color eps
set out "eulerX0${X0}V0${V0}_xs$s.eps"
set title ""
replot
set out
set term x11

set term x11 2
set title "Velocities. STEPS= $s"
set ylabel "{/Symbol w}"
set xlabel "t"
plot "euler.dat" using 1:3  title "Euler", \
     "euler_cromer.dat" using 1:3  title "Euler-Cromer", \
     "euler_verlet.dat" using 1:3  title "Euler-Verlet", \
     v(t) title "v_s(t)"
set term postscript enhanced color eps
set out "eulerX0${X0}V0${V0}_vs$s.eps"
set title ""
replot
set out
set term x11
EOF

# Preparing
 echo -n " Processing data....." 
 for f in euler euler_cromer euler_verlet;do
  cat $f.dat >> $f.all
  echo "\n"  >> $f.all
 done

  pltx_e= "$pltx_e   'euler.all'        using 1:2 index $scnt title '$s',"
 pltx_ec= "$pltx_ec  'euler_cromer.all' using 1:2 index $scnt title '$s',"
 pltx_ev= "$pltx_ev  'euler_verlet.all' using 1:2 index $scnt title '$s',"
  pltv_e= "$pltv_e   'euler.all'        using 1:3 index $scnt title '$s',"
 pltv_ec= "$pltv_ec  'euler_cromer.all' using 1:3 index $scnt title '$s',"
 pltv_ev= "$pltv_ev  'euler_verlet.all' using 1:3 index $scnt title '$s',"

 let scnt++
 echo "Done!"   

done # for s in ${steps[@]}
#--------------------------------------------------------------------------
# If we want the simple pendulum solution we add it as the last function
# to plo
if (( $exact_sol == 1 )); then
  pltx_e="$pltx_e   x(t) title '{/Symbol q}_s(t)'"
 pltx_ec="$pltx_ec  x(t) title '{/Symbol q}_s(t)'"
 pltx_ev="$pltx_ev  x(t) title '{/Symbol q}_s(t)'"
  pltv_e="$pltv_e   v(t) title 'v_s(t)'"
 pltv_ec="$pltv_ec  v(t) title 'v_s(t)'"
 pltv_ev="$pltv_ev  v(t) title 'v_s(t)'"
else
# else we plot something not plotable
  pltx_e="$pltx_e   1/0 notit"
 pltx_ec="$pltx_ec  1/0 notit"
 pltx_ev="$pltx_ev  1/0 notit"
  pltv_e="$pltv_e   1/0 notit"
 pltv_ec="$pltv_ec  1/0 notit"
 pltv_ev="$pltv_ev  1/0 notit"
fi


gnuplot -persist<<EOF
set dummy t  #t is the independent variable in functions
set style data lines
omega2  = 10  #This is g/l in program
X0      = $X0 #Amplitute of oscillation
V0      = $V0 #Velocity at t0
omega   = sqrt(omega2)
x(t)    = X0 * cos(omega * t) +(V0/omega)*sin(omega*t)
v(t)    = V0 * cos(omega * t) -(omega*X0)*sin(omega*t)

set term x11 1
set title "Positions: Euler Method"
set ylabel "{/Symbol q}"
set xlabel "t"
$pltx_e
set term postscript enhanced color eps
set out "euler.STEPS_X0${X0}V0${V0}_x.eps"
set title ""
replot
set out
set term x11

set term x11 2
set title "Velocities: Euler Method"
set ylabel "{/Symbol w}"
set xlabel "t"
$pltv_e
set term postscript enhanced color eps
set out "euler.STEPS_X0${X0}V0${V0}_v.eps"
set title ""
replot
set out
set term x11

set term x11 3
set title "Positions: Euler-Cromer Method"
set ylabel "{/Symbol q}"
set xlabel "t"
$pltx_ec
set term postscript enhanced color eps
set out "euler_cromer.STEPS_X0${X0}V0${V0}_x.eps"
set title ""
replot
set out
set term x11

set term x11 4
set title "Velocities: Euler-Cromer Method"
set ylabel "{/Symbol w}"
set xlabel "t"
$pltv_ec
set term postscript enhanced color eps
set out "euler_cromer.STEPS_X0${X0}V0${V0}_v.eps"
set title ""
replot
set out
set term x11

set term x11 5
set title "Positions: Euler-Verlet Method"
set ylabel "{/Symbol q}"
set xlabel "t"
$pltx_ev
set term postscript enhanced color eps
set out "euler_verlet.STEPS_X0${X0}V0${V0}_x.eps"
set title ""
replot
set out
set term x11

set term x11 6
set title "Velocities: Euler-Verlet Method"
set ylabel "{/Symbol w}"
set xlabel "t"
$pltv_ev
set term postscript enhanced color eps
set out "euler_verlet.STEPS_X0${X0}V0${V0}_v.eps"
set title ""
replot
set out
set term x11

EOF
#  ---------------------------------------------------------------------
#  Copyright by Konstantinos N. Anagnostopoulos (2004-2014)
#  Physics Dept., National Technical University,
#  konstant@mail.ntua.gr, www.physics.ntua.gr/~konstant
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, version 3 of the License.
#  
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#  
#  You should have received a copy of the GNU General Public Liense along
#  with this program.  If not, see <http://www.gnu.org/licenses/>.
#  -----------------------------------------------------------------------

