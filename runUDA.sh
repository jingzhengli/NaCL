#!/usr/bin/env bash

        # 'amazon': 'Office31',
        # 'dslr': 'Office31',
        # 'webcam': 'Office31',
        # 'c': 'image-clef',
        # 'i': 'image-clef',
        # 'p': 'image-clef',
        # 'Art': 'Office-Home',
        # 'Clipart': 'Office-Home',
        # 'Product': 'Office-Home',
        # 'Real_World': 'Office-Home',
        # 'train': 'visda-2017',
        # 'validation': 'visda-2017'

# Office31
# python main.py --network resnet50 --src i --tgt p --contrast_dim 128 --module contrastive_loss --cw 1 --batch_size 32 --max_key_size 20
# python main.py --network resnet50 --src p --tgt i --contrast_dim 128 --module contrastive_loss --cw 1 --batch_size 32 --max_key_size 20
# python main.py --network resnet50 --src i --tgt c --contrast_dim 128 --module contrastive_loss --cw 1 --batch_size 32 --max_key_size 20
# python main.py --network resnet50 --src c --tgt i --contrast_dim 128 --module contrastive_loss --cw 1 --batch_size 32 --max_key_size 20
# python main.py --network resnet50 --src c --tgt p --contrast_dim 128 --module contrastive_loss --cw 1 --batch_size 32 --max_key_size 20

python main.py --network resnet50 --src Art --tgt Clipart --contrast_dim 128 --module contrastive_loss --cw 1 --batch_size 32 --max_key_size 20 --max_iterations 5000 --kcc 3
python main.py --network resnet50 --src Art --tgt Product --contrast_dim 128 --module contrastive_loss --cw 1 --batch_size 32 --max_key_size 20 --max_iterations 5000 --kcc 3
python main.py --network resnet50 --src Art --tgt Real_World --contrast_dim 128 --module contrastive_loss --cw 1 --batch_size 32 --max_key_size 20 --max_iterations 5000 --kcc 3
python main.py --network resnet50 --src Clipart --tgt Product --contrast_dim 128 --module contrastive_loss --cw 1 --batch_size 32 --max_key_size 20 --max_iterations 5000 --kcc 3
python main.py --network resnet50 --src Clipart --tgt Art --contrast_dim 128 --module contrastive_loss --cw 1 --batch_size 32 --max_key_size 20 --max_iterations 5000 --kcc 3
python main.py --network resnet50 --src Clipart --tgt Real_World --contrast_dim 128 --module contrastive_loss --cw 1 --batch_size 32 --max_key_size 20 --max_iterations 5000 --kcc 3
python main.py --network resnet50 --src Real_World --tgt Product --contrast_dim 128 --module contrastive_loss --cw 1 --batch_size 32 --max_key_size 20 --max_iterations 5000 --kcc 3
python main.py --network resnet50 --src Real_World --tgt Clipart --contrast_dim 128 --module contrastive_loss --cw 1 --batch_size 32 --max_key_size 20 --max_iterations 5000 --kcc 3
python main.py --network resnet50 --src Real_World --tgt Art --contrast_dim 128 --module contrastive_loss --cw 1 --batch_size 32 --max_key_size 20 --max_iterations 5000 --kcc 3
python main.py --network resnet50 --src Product --tgt Clipart --contrast_dim 128 --module contrastive_loss --cw 1 --batch_size 32 --max_key_size 20 --max_iterations 5000 --kcc 3
python main.py --network resnet50 --src Product --tgt Art --contrast_dim 128 --module domain_loss --cw 1 --batch_size 32 --max_key_size 20 --max_iterations 5000 --kcc 3
python main.py --network resnet50 --src Product --tgt Real_World --contrast_dim 128 --module domain_loss --cw 1 --batch_size 32 --max_key_size 20 --max_iterations 5000 --kcc 3


python main.py --network resnet101 --src train --tgt validation --contrast_dim 128 --module domain_loss --cw 1 --batch_size 48 --max_key_size 100 --max_iterations 20000 --kcc 3


python main.py --network resnet50 --src amazon --tgt webcam --contrast_dim 128 --module domain_loss --cw 1 --batch_size 32 --max_key_size 20 --kcc 3 --max_iterations 2000
python main.py --network resnet50 --src amazon --tgt dslr --contrast_dim 128 --module domain_loss --cw 1 --batch_size 32 --max_key_size 20 --kcc 3 --max_iterations 2000 
python main.py --network resnet50 --src dslr --tgt amazon --contrast_dim 128 --module domain_loss --cw 1 --batch_size 32 --max_key_size 20 --kcc 3 --max_iterations 2000 
python main.py --network resnet50 --src dslr --tgt webcam --contrast_dim 128 --module domain_loss --cw 1 --batch_size 32 --max_key_size 20 --kcc 3 --max_iterations 2000 
python main.py --network resnet50 --src webcam --tgt amazon --contrast_dim 128 --module domain_loss --cw 1 --batch_size 32 --max_key_size 20 --kcc 3 --max_iterations 2000 
python main.py --network resnet50 --src webcam --tgt dslr --contrast_dim 128 --module domain_loss --cw 1 --batch_size 32 --max_key_size 20 --kcc 3 --max_iterations 2000 


