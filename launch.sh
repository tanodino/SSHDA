for i in 25 50 100 200
do
    for j in 0 1 2 3 4
    do
        python main.py $1 $2 $3 $i $j > log_our_${1}_${2}_${3}_${i}_${j}
    done
done

#sh launch.sh TreeSatAI AERIAL MS
#sh launch.sh EuroSAT_OPT_SAR MS SAR