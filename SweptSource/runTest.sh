#!/usr/bin/zsh

make

BINPTH=./bin
RSLPTH=./Results
RSLSUFFX=_Timing.txt
NTS=5e5

EQS=(Heat KS Euler)
DTS=(0.001 0.005 1.0e-7)
PREC=(Single Double)
SCH=(Classic Shared Alternative)

for neq in $(seq 3)
do 
    eqn=$EQS[$neq]
    execs=$BINPTH/$eqn
    tdt=$DTS[$neq]
    tf=$($tdt*$NTS) 
    for nprec in $(seq 2)
    do 
        pr=$PREC[$nprec]
        execs+=$pr+"Out"
        for nsch in $(seq 3)
        do
            sch=$SCH[$nsch]
            rsltp=$RSLPTH/$eqn_$pr_$sch$RSLSUFFX
            for dv in $(seq 10)
            do
                dvi=$($(dv+10)**2)
                for tpb in $(seq 5)
                do 
                    tpbb=$($(tpb+5)**2)    
                    $execs $dvi $tpb $tdt $tf $(2 * $tf) $sch _ $rsltp
                done
            done  
        done
    done
done

