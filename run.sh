#!/bin/bash
set -e

# fineweb_edu
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py config/preln.py
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py config/periln.py
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py config/keel.py
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py config/predyt.py
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py config/geonorm.py --out_prefix_method=geonorm_lns
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py config/projspheretnorm.py --out_prefix_method=projspheretnorm_lns
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py config/cayspheretnorm.py --out_prefix_method=cayspheretnorm_lns
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py config/pspheretnorm.py --out_prefix_method=pspheretnorm_lns

# owt
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py config/preln.py
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py config/periln.py
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py config/keel.py
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py config/predyt.py
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py config/geonorm.py --out_prefix_method=geonorm_lns
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py config/projspheretnorm.py --out_prefix_method=projspheretnorm_lns
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py config/cayspheretnorm.py --out_prefix_method=cayspheretnorm_lns
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py config/pspheretnorm.py --out_prefix_method=pspheretnorm_lns