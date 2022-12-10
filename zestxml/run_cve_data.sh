extra_args="${@:3}"
./run.sh $1 $2 -bilinear_classifier_cost 5 -bs_count 120 -bs_direct_wt 0.8 -bs_alpha 0.02 -shortyK 150 ${extra_args}
