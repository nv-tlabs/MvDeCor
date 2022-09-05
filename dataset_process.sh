mkdir logs_selfsup-proj
mkdir logs_selfsup-proj/logs
mkdir logs_selfsup-proj/results
mkdir logs_selfsup-proj/configs
mkdir logs_selfsup-proj/scripts
mkdir logs_selfsup-proj/models
mkdir logs_selfsup-proj/tensorboard

mkdir -p partnet/partnet_dataset/
cd partnet/partnet_dataset/

wget http://download.cs.stanford.edu/orion/partnet_dataset/data_v0.zip
wget http://download.cs.stanford.edu/orion/partnet_dataset/sem_seg_h5.zip

unzip data_v0.zip
unzip sem_seg_h5.zip
cd ../../

# choose your categories from: Bed Bottle Chair Clock Dishwasher Display Door Earphone Faucet Knife Lamp Microwave Refrigerator StorageFurniture Table TrashCan Vase
declare -a categories=(Faucet)
declare -a ids=("train" "test")
fewshot=10
for cat in "${categories[@]}"
do
    for id in "${ids[@]}"
    do
        echo $cat $id
        python rendering_partnet.py $cat $id 0 10000 "partnet/partnet_dataset/sem_seg_h5/" "partnet/partnet_dataset/data_v0/"
    done
done

