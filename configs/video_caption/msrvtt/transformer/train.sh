CUDA_VISIBLE_DEVICES=0 python3 train_net.py --config-file configs/video_caption/msrvtt/transformer/transformer.yaml --num-gpus 1 OUTPUT_DIR ./experiments/vtt-transformer-f50 DATALOADER.TRAIN_BATCH_SIZE 256 DATALOADER.MAX_FEAT_NUM 50 DECODE_STRATEGY.BEAM_SIZE 5 MODEL.BERT.NUM_GENERATION_LAYERS 3 MODEL.BERT.NUM_HIDDEN_LAYERS 3 DATALOADER.NUM_WORKERS 4


CUDA_VISIBLE_DEVICES=0 python3 train_net.py --config-file configs/video_caption/msrvtt/transformer/transformer.yaml --num-gpus 1 OUTPUT_DIR ./experiments/vtt-transformer-f25-2-2-warmup DATALOADER.TRAIN_BATCH_SIZE 256 DECODE_STRATEGY.BEAM_SIZE 5 DATALOADER.NUM_WORKERS 4 DATALOADER.MAX_FEAT_NUM 25
