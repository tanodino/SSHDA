for i in 50 100 150 200 250 300 350 400
do
    for j in 0 1 2 3 4
    do
        python fixmatch.py $1 $2 $i $j > log_${1}_${2}_${i}_${j}
    done
done