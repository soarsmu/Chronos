while getopts d:l:m:i: flag
do
    case "${flag}" in
        d) data=${OPTARG};;
        l) label=${OPTARG};;
        m) para=${OPTARG};;
        i) top=${OPTARG};;
    esac
done
python prepare_data.py $data $label
cd zero_shot_dataset
cp -r zestxml /workspace/Chronos/zestxml/GZXML-Datasets
cd ..
cd zestxml
./run_cve_data.sh zestxml train
./run_cve_data.sh zestxml predict

if [[ $top -gt 0 ]]
then
  python rerank_cached.py  --dataset zestxml --output_file output.txt --updated_by_cache  --para $para --update_position $top --replacement_by_versions
else
  python F1_metrics.py zestxml
fi

# python rerank_cached_exp.py zestxml
cd ..