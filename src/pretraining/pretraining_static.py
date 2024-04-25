import copy
import sys
import os

sys.path.append(os.path.abspath('..'))

import random
import numpy as np
import torch
from tqdm import tqdm
from datasets.zones import Zones
from evaluate.evaluation import SegmentationEvaluator, write_segmentation_performance
from datasets.dataset import *
import wandb


def write_log_args(args, save_path):
    f = open(os.path.join(save_path, 'arguments'), 'w')
    for key, value in vars(args).items():
        f.write(f'{key} = {value}\n')
    f.close()


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


if __name__ == "__main__":

    """ Import args dict """
    print("-"*20, "Initialization", "-"*20)

    from arguments import args
    args.dynamic_weather = False
    args.learning = 'pretraining'
    args.pretraining = '_'
    args.name = 'static' + '_' + args.supervision
    args.dataset = os.path.join(args.dataset, 'static_weather')

    """ Prepare world-related variables """

    world = Zones(args.dataset)
    envs = sorted(world.ZONES, key=lambda x: world.ZONES[x]['name'])
    n_envs = len(envs)

    """ Prepare output folders and save args dict """

    os.makedirs(os.path.join(args.save_folder, args.name), exist_ok=True)
    #write_log_args(args, save_path=os.path.join(args.save_folder, args.name))

    """ Set determinism for reproducibility """

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    """ Training-related initialization """

    from models.tinynet import TinyNet
    device = get_default_device()

    # Get the weights for the class imbalance and update the criterion
    # weights = compute_weights(labels).to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    """ Benchmark for each environment """

    #wandb.login()
    #wandb.init(project='', entity='', config=args)  # To complete with specific project and entity names
    #wandb.run.name = args.name

    n_envs = 1 if args.cell_type == 'common' else n_envs

    for env_idx in range(n_envs):
        zone_name = world.ZONES[envs[env_idx]]['name'] if n_envs > 1 else 'common'
        if zone_name not in args.zones_to_consider and args.zones_to_consider[0] != 'all':
            print("not in list, skipping...")
            continue

        print('-'*30)
        if n_envs > 1:
            print(f"Environment '{world.ZONES[envs[env_idx]]['name']}' ({env_idx+1}/{n_envs})")
            args.env_name_key = f'{zone_name}'
        else:
            print(f"All environments considered as one")
        print('-'*30)

        env_name = world.ZONES[envs[env_idx]]['name'] if n_envs > 1 else 'common'
        save_path_env = os.path.join(args.save_folder, args.name, env_name)
        os.makedirs(save_path_env, exist_ok=True)

        # Dataset instantiation

        pretraining_set = OfflineDataset(args, (envs[env_idx], -1, -1), args.start, args.stop)
        train_loader, val_loader = pretraining_set.get_train_val()

        # Definition of the optimizer

        model = TinyNet(num_classes=args.num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        n_images = (len(pretraining_set.train_indices) + len(pretraining_set.val_indices))
        scale_epoch = 78822 / n_images
        args.n_epochs = max(args.n_epochs, int(args.n_epochs * scale_epoch))

        if args.scheduler == 'constant':
            lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=args.n_epochs)
        elif args.scheduler == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-5)

        print(f'train : {len(train_loader)*args.batchsize} | validation : {len(val_loader)*args.batchsize}   (total : {len(train_loader)*args.batchsize + len(val_loader)*args.batchsize})')

        evaluator = SegmentationEvaluator(n_classes=args.num_classes, args=args)
        best_metric = -1


        for epoch in range(args.n_epochs):
            print(f"Epoch {epoch + 1}/{args.n_epochs}")

            # Training step
            model.train()
            epoch_loss = 0
            step = 0
            for i, (images, pseudos, masks, _, _) in enumerate(tqdm(train_loader, desc="Training", unit="batch")):
                if args.supervision == 'teacher':
                    targets = pseudos
                elif args.supervision == 'carla':
                    targets = masks
                torch.cuda.empty_cache()
                images = images.to(device)
                targets = targets.to(device)
                outputs = model.forward(images)
                optimizer.zero_grad()
                loss = criterion(outputs, targets)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                images = images.to("cpu")
                targets = targets.to("cpu")
                step += 1

            lr_scheduler.step()
            epoch_loss /= step
            print(f'Epoch loss : {epoch_loss}')

            # Validation step
            if args.validation_fraction > 0:

                score_st, score_sc = evaluator.validation(model, val_loader, keyword='Val', env_name=world.ZONES[envs[env_idx]]['name'] if n_envs>1 else 'common')


                if args.supervision == 'teacher':
                    score = score_st
                elif args.supervision == 'carla':
                    score = score_sc
                print(f'Validation score : {score} (mIoU student - {args.supervision})')

                if score >= best_metric:
                    best_metric = score
                    torch.save(model.state_dict(), os.path.join(save_path_env, f'pretrained.pth'))

            else:
                torch.save(model.state_dict(), os.path.join(save_path_env, f'pretrained.pth'))  # save model at each epoch if no validation



