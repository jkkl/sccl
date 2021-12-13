/opt/conda/bin/pip install -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple sentence_transformers==0.4.1.2 tensorboardX
/opt/conda/bin/python3.8 /mnt/user/wangyuanzhuo/sccl/top_intention_response_cluster.py \
    --result_path ./restest/intention_multiturn/ \
    --cluster_result_path saved_result/intention_multi_random \
    --num_classes 135 \
    --data_path ./datasamples/intention_multi \
    --dataname intention_next_turn.txt.sort.2.randomchit \
    --dataset intention_next_turn.txt.sort.2.randomchit \
    --bert chinese \
    --alpha 1 \
    --lr 1e-05 \
    --lr_scale 100 \
    --batch_size 10 \
    --temperature 0.5 \
    --base_temperature 0.07 \
    --max_iter 10 \
    --print_freq 250 \
    --seed 0 \
    --gpuid 0 \
    --is_use_cl 0 \
    --is_use_simcse 0 






