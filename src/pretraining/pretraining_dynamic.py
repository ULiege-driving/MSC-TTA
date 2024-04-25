import copy
import sys
import os

sys.path.append(os.path.abspath('..'))

import random
import numpy as np
import torch
from tqdm import tqdm
from datasets.zones_dynamic import Zones
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
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


if __name__ == "__main__":

    """ Import args dict """
    print("-"*20, "Initialization", "-"*20)

    from arguments import args

    args.dynamic_weather = True
    args.learning = 'pretraining'
    args.pretraining = '_'
    args.name = 'dynamic' + '_' + args.supervision
    args.dataset = os.path.join(args.dataset, 'dynamic_weather')

    """ Prepare world-related variables """

    world = Zones(args.dataset, dynamic_weather=True)
    zones = sorted(world.ZONES, key=lambda x: world.ZONES[x]['name'])
    weathers = sorted(world.WEATHERS, key=lambda x: world.WEATHERS[x]['name'])
    periods = sorted(world.PERIODS, key=lambda x: world.PERIODS[x]['name'])
    n_zones = len(zones)
    n_weathers = len(weathers)
    n_periods = len(periods)




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
    #wandb.init(project='', entity='', config=args)
    #wandb.run.name = args.name

    if args.cell_type == 'common':
        n_zones = 1
        n_weathers = 1
        n_periods = 1
    if args.cell_type == 'spatial':
        n_weathers = 1
        n_periods = 1
    if args.cell_type == 'weather':
        n_zones = 1
        n_periods = 1
    if args.cell_type == 'daylight':
        n_zones = 1
        n_weathers = 1
    if args.cell_type == 'weatherdaylight':
        n_zones = 1

    n_envs = n_zones * n_weathers * n_periods

    for zone_idx in range(n_zones):
        for weather_idx in range(n_weathers):
            for period_idx in range(n_periods):

                zone_name = world.ZONES[zones[zone_idx]]['name'] if n_zones > 1 else 'common'
                weather_name = world.WEATHERS[weathers[weather_idx]]['name'] if n_weathers > 1 else 'common'
                period_name = world.PERIODS[periods[period_idx]]['name'] if n_periods > 1 else 'common'
                print('-'*50)
                if args.cell_type == 'specific':
                    print(f"Zone '{zone_name}' ({zone_idx+1}/{n_zones})")
                    print(f"Weather '{weather_name}' ({weather_idx + 1}/{n_weathers})")
                    print(f"Period '{period_name}' ({period_idx + 1}/{n_periods})")
                    args.env_name_key = f'{zone_name}_{weather_name}_{period_name}'
                    save_path_env = os.path.join(args.save_folder, args.name, zone_name, weather_name, period_name)
                    if (zone_name not in args.zones_to_consider and args.zones_to_consider[0] != 'all') or (weather_name not in args.weathers_to_consider and args.weathers_to_consider[0] != 'all') or (period_name not in args.periods_to_consider and args.periods_to_consider[0] != 'all'):
                        print("not in list, skipping...")
                        continue
                elif args.cell_type == 'spatial':
                    print(f"Zone '{zone_name}' ({zone_idx+1}/{n_zones})")
                    save_path_env = os.path.join(args.save_folder, args.name, zone_name, 'common')
                    args.env_name_key = f'{zone_name}'
                    if zone_name not in args.zones_to_consider and args.zones_to_consider[0] != 'all':
                        print("not in list, skipping...")
                        continue
                elif args.cell_type == 'weather':
                    print(f"Weather '{zone_name}' ({weather_idx+1}/{n_weathers})")
                    save_path_env = os.path.join(args.save_folder, args.name, 'weather', weather_name)
                    args.env_name_key = f'{weather_name}'
                    if weather_name not in args.weathers_to_consider and args.weathers_to_consider[0] != 'all':
                        print("not in list, skipping...")
                        continue
                elif args.cell_type == 'daylight':
                    print(f"Period '{period_name}' ({period_idx + 1}/{n_periods})")
                    save_path_env = os.path.join(args.save_folder, args.name, 'period', period_name)
                    args.env_name_key = f'{period_name}'
                    if period_name not in args.periods_to_consider and args.periods_to_consider[0] != 'all':
                        print("not in list, skipping...")
                        continue
                elif args.cell_type == 'weatherdaylight':
                    print(f"Weather '{zone_name}' ({weather_idx + 1}/{n_weathers})")
                    print(f"Period '{period_name}' ({period_idx + 1}/{n_periods})")
                    save_path_env = os.path.join(args.save_folder, args.name, 'weatherperiod', weather_name, period_name)
                    args.env_name_key = f'{weather_name}_{period_name}'
                    if (weather_name not in args.weathers_to_consider and args.weathers_to_consider[0] != 'all') or (period_name not in args.periods_to_consider and args.periods_to_consider[0] != 'all'):
                        print("not in list, skipping...")
                        continue
                else:
                    print(f"All environments considered as one")
                    save_path_env = os.path.join(args.save_folder, args.name, 'common')
                    args.env_name_key = 'common'
                print('-'*50)

                os.makedirs(save_path_env, exist_ok=True)

                env_id = (zones[zone_idx], weathers[weather_idx], periods[period_idx])
                # dataset instantiation
                pretraining_set = OfflineDataset(args, env_id, args.start, args.stop)
                train_loader, val_loader = pretraining_set.get_train_val()
                if train_loader is None:
                    continue

                # Definition of the optimizer

                model = TinyNet(num_classes=args.num_classes).to(device)
                scaler = torch.cuda.amp.GradScaler()
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

                n_images = (len(pretraining_set.train_indices) + len(pretraining_set.val_indices))
                scale_epoch = 283658 / n_images
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
                        with torch.cuda.amp.autocast():
                            images = images.to(device)
                            targets = targets.to(device)
                            outputs = model.forward(images)
                            loss = criterion(outputs, targets)
                        epoch_loss += loss.item()
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        images = images.to("cpu")
                        targets = targets.to("cpu")
                        step += 1

                    lr_scheduler.step()
                    epoch_loss /= step
                    print(f'Epoch loss : {epoch_loss}')

                    # Validation step
                    if args.validation_fraction > 0:

                        score_st, score_sc = evaluator.validation(model, val_loader, keyword='Val', env_name=f'{zone_name}_{weather_name}_{period_name}')

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



