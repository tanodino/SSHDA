for i in 50 100 150 200 250 300 350 400
do
    for j in 0 1 2 3 4
    do
        python main.py $1 $2 $3 $i $j > log_our_${1}_${2}_${3}_${i}_${j}
    done
done

#sh launch.sh TreeSatAI AERIAL MS
#sh launch.sh EuroSAT_OPT_SAR MS SAR