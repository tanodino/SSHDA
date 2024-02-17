#for i in 0 1 2 3 4 5 6
for i in 8 9
do
    for j in 0 1 2 3 4
    do
        python abla.py EuroSAT_OPT_SAR MS SAR 50 $j $i > log_our_EuroSAT_OPT_SAR_MS_SAR_50_${j}_abla_${i}
    done
done

#sh launch.sh TreeSatAI AERIAL MS
#sh launch.sh EuroSAT_OPT_SAR MS SAR
