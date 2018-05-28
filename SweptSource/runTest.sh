#!/usr/bin/zsh

make

BINPTH=./bin
RSLPTH=./Results
RSLSUFFX=_Timing.txt
NTS=5e4

EQS=(Heat KS Euler)
DTS=(0.001 0.005 1.0e-7)
PREC=(Single Double)
SCH=(Classic Shared Alternative)

for neq in $(seq 3)
do 
    eqn=$EQS[$neq]
    execs=$BINPTH/$eqn
    tdt=$DTS[$neq]
    tf=$(( $tdt*$NTS ))
    for nprec in 2
    do 
        pr=$PREC[$nprec]
        execs+=${pr}Out
        for nsch in $(seq 3)
        do
            sch=$SCH[$nsch]
            nsarg=$(($nsch - 1))
            rsltp=${RSLPTH}/${eqn}_${pr}_${sch}${RSLSUFFX}
            for dv in $(seq 10)
            do
                dvi=$((2**($dv+10)))
                for tpb in $(seq 5)
                do 
                    tpbb=$((2**($tpb+5)))
                    echo $execs $dvi $tpbb $tdt $tf $((2 * $tf)) $nsarg $RSLPTH/temp.dat $rsltp
                    $execs $dvi $tpbb $tdt $tf $((2 * $tf)) $nsarg $RSLPTH/temp.dat $rsltp
                done
            done  
        done
    done
done

