for i in 1 2 3 4 5 6
do
    for j in 0 1 2 3 4
    do
        python abla.py EuroSAT_OPT_SAR MS SAR 50 $j $i > log_our_${1}_${2}_${3}_50_${j}_abla_${i}
    done
done

#sh launch.sh TreeSatAI AERIAL MS
#sh launch.sh EuroSAT_OPT_SAR MS SAR