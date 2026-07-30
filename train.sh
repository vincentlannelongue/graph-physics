python -m graphphysics.train \
            --project_name=AX \
            --training_parameters_path=training_config/BAX_grad.json \
            --num_epochs=20 \
            --init_lr=0.001 \
            --batch_size=1 \
            --warmup=500 \
            --num_workers=0 \
            --prefetch_factor=0 \
            --model_save_name=BAX_grad_1 \
            --no_edge_feature \
            --use_previous_data=true \
            --previous_data_start=4 \
            --previous_data_end=7 \
            --seed=1 \
            # --model_path=checkpoints/.ckpt \