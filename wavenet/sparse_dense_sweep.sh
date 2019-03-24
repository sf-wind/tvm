for m in 1 2 4 8 16
do
  for bs_r in 1 2 4 8 16
  do
    for bs_c in 1 2 4 8 16
    do
      python3 sparse_test.py --m $m --bs_r $bs_r --bs_c $bs_c --num_threads 1
      sleep 1
    done
  done
done
