for i in 50 100 150 200 250 300 350 400
do
    for j in 0 1 2 3 4
    do
        echo "python fixmatch.py $1 $2 $i $j > log_$1_$2_$i_$j"
    done
done