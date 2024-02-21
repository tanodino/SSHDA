for i in 25 50 100 200
do
    for j in 0 1 2 3 4
    do
        python SSHIDA2.py $1 $2 $3 $i $j > log_HIDA_${1}_${2}_${3}_${i}_${j}
    done
done

#sh launch_HIDA.sh TreeSatAI AERIAL MS
#sh launch_HIDA.sh EuroSAT_OPT_SAR MS SAR
#sh launch_HIDA.sh RESISC45_EuroSAT EURO RESISC45
#python SSHIDA.py EuroSAT_OPT_SAR MS SAR 100 0
