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

import wandb


def write_log_args(args, save_path):
    f = open(os.path.join(save_path, 'arguments'), 'w')
    for key, value in vars(args).items():
        f.write(f'{key} = {value}\n')
    f.close()


def get_weights(args, zone_name, weather_name, period_name, device):
    if args.cell_type == 'standalone':
        weights_path = os.path.join(args.pretraining_path, 'dynamic_common', 'common', 'pretrained.pth')
    elif args.cell_type == 'spatial':
        weights_path = os.path.join(args.pretraining_path, 'dynamic_spatial', zone_name, 'common', 'pretrained.pth')
    elif args.cell_type == 'weather':
        weights_path = os.path.join(args.pretraining_path, 'dynamic_weather', 'weather', weather_name, 'pretrained.pth')
    elif args.cell_type == 'daylight':
        weights_path = os.path.join(args.pretraining_path, 'dynamic_daylight', 'daylight', period_name, 'pretrained.pth')
    elif args.cell_type == 'weatherdaylight':
        weights_path = os.path.join(args.pretraining_path, 'dynamic_weatherdaylight', 'weatherdaylight', weather_name, period_name, 'pretrained.pth')
    elif args.cell_type == 'specific':
        weights_path = os.path.join(args.pretraining_path, 'dynamic_specific', zone_name, weather_name, period_name,
                                    'pretrained.pth')

    elif args.cell_type == 'common':
        weights_path = os.path.join(args.pretraining_path, 'dynamic_common', 'common', 'pretrained.pth')

    return torch.load(weights_path, map_location=device)


if __name__ == "__main__":

    """ Import args dict """
    print("-" * 20, "Initialization", "-" * 20)

    from arguments_dynamic import args
    args.learning = 'offline'
    args.dynamic_weather = True
    args.name = 'offline' + '_' + 'dynamic'
    args.onlinedataset = 'fifo'
    args.dataset = os.path.join(args.dataset, 'dynamic_weather')


    """ Prepare world-related variables """
    world = Zones(args.dataset, dynamic_weather=True)
    zones = sorted(world.ZONES, key=lambda x: world.ZONES[x]['name'])
    weathers = sorted(world.WEATHERS, key=lambda x: world.WEATHERS[x]['name'])
    periods = sorted(world.PERIODS, key=lambda x: world.PERIODS[x]['name'])
    n_zones = len(zones)
    n_weathers = len(weathers)
    n_periods = len(periods)

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

    """ Prepare output folders """

    os.makedirs(args.save_folder, exist_ok=True)


    """ Set determinism for reproducibility """

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    """ Training-related initialization """

    from models.tinynet import TinyNet

    device = args.device

    # Get the weights for the class imbalance and update the criterion
    # weights = compute_weights(labels).to(device)


    dataset = None
    if args.onlinedataset == "fifo":
        from datasets.fifo import FifoDataset

    """ Benchmark for each environment """
    """
    wandb.login()
    wandb.init(project='', entity='', config=args)
    wandb.run.name = args.name
    fuse = True
    """

    for zone_idx in range(n_zones):
        for weather_idx in range(n_weathers):
            for period_idx in range(n_periods):

                zone_name = world.ZONES[zones[zone_idx]]['name'] if n_zones > 1 else 'no_env'
                weather_name = world.WEATHERS[weathers[weather_idx]]['name'] if n_weathers > 1 else 'no_env'
                period_name = world.PERIODS[periods[period_idx]]['name'] if n_periods > 1 else 'no_env'
                print('-' * 50)
                if args.cell_type == 'specific':
                    print(f"Zone '{zone_name}' ({zone_idx + 1}/{n_zones})")
                    print(f"Weather '{weather_name}' ({weather_idx + 1}/{n_weathers})")
                    print(f"Period '{period_name}' ({period_idx + 1}/{n_periods})")
                    args.env_name_key = f'{zone_name}/{weather_name}/{period_name}'
                    # save_path_env = os.path.join(args.save_folder, args.name, zone_name, weather_name, period_name)
                    if (zone_name not in args.zones_to_consider and args.zones_to_consider[0] != 'all') or (
                            weather_name not in args.weathers_to_consider and args.weathers_to_consider[
                        0] != 'all') or (
                            period_name not in args.periods_to_consider and args.periods_to_consider[0] != 'all'):
                        print("not in list, skipping...")
                        continue
                elif args.cell_type == 'spatial':
                    print(f"Zone '{zone_name}' ({zone_idx + 1}/{n_zones})")
                    # args.save_path_env = os.path.join(args.save_folder, args.name, zone_name, 'no_env')
                    args.env_name_key = f'{zone_name}'
                    if zone_name not in args.zones_to_consider and args.zones_to_consider[0] != 'all':
                        print("not in list, skipping...")
                        continue
                elif args.cell_type == 'weather':
                    print(f"Weather '{weather_name}' ({weather_idx+1}/{n_weathers})")
                    #args.save_path_env = os.path.join(args.save_folder, args.name, zone_name, 'no_env')
                    args.env_name_key = f'{weather_name}'
                    if weather_name not in args.weathers_to_consider and args.weathers_to_consider[0] != 'all':
                        print("not in list, skipping...")
                        continue
                elif args.cell_type == 'daylight':
                    print(f"Period '{period_name}' ({period_idx+1}/{n_periods})")
                    #args.save_path_env = os.path.join(args.save_folder, args.name, zone_name, 'no_env')
                    args.env_name_key = f'{period_name}'
                    if period_name not in args.periods_to_consider and args.periods_to_consider[0] != 'all':
                        print("not in list, skipping...")
                        continue
                elif args.cell_type == 'weatherdaylight':
                    print(f"Weather '{weather_name}' ({weather_idx + 1}/{n_weathers})")
                    print(f"Period '{period_name}' ({period_idx + 1}/{n_periods})")
                    # args.save_path_env = os.path.join(args.save_folder, args.name, zone_name, 'no_env')
                    args.env_name_key = f'{weather_name}_{period_name}'
                    if (weather_name not in args.weathers_to_consider and args.weathers_to_consider[0] != 'all') or (
                            period_name not in args.periods_to_consider and args.periods_to_consider[0] != 'all'):
                        print("not in list, skipping...")
                        continue
                elif args.cell_type == 'common':
                    print(f"All environment considered as one")
                    # args.save_path_env = os.path.join(args.save_folder, args.name, 'no_env')
                    args.env_name_key = 'common'
                print('-' * 50)

                if fuse:
                    sf = os.path.join(args.save_folder)
                    subfolder = f'{args.cell_type}-{args.learning}-{args.supervision}/{args.env_name_key}'
                    sp = os.path.join(sf, subfolder)
                    if os.path.exists(sp):
                        if os.listdir(sp):
                            exit()
                    fuse = False

                # dataset instantiation
                env_id = (zones[zone_idx], weathers[weather_idx], periods[period_idx])
                dataset = FifoDataset(args, env_id, args.start, args.stop)
                if not dataset.timestamps:
                    print('No frame in range, skipping ...')
                    continue

                evaluator_now = SegmentationEvaluator(n_classes=args.num_classes, args=args, keyword='now')

                # Definition of the optimizer

                model = TinyNet(num_classes=args.num_classes).to(device)
                state_dict = get_weights(args, zone_name, weather_name, period_name, device=device)
                model.load_state_dict(state_dict)

                first_timestamp = dataset.first_timestamp
                last_timestamp = dataset.last_timestamp

                evaluate_span = 30

                print(f'FIRST - LAST : {first_timestamp} - {last_timestamp}')

                for time_idx in range(first_timestamp, last_timestamp + 1, 30):


                    ############################################
                    # Evaluate model in [t, t + evaluate_span] #
                    ############################################
                    print(f"Evaluate model in [{time_idx}, {time_idx + evaluate_span}]")

                    dataloader = dataset.create_testset(time_idx, time_idx + evaluate_span)
                    evaluator_now.evaluate(args, model, dataloader, keyword='now', save=args.save_CM, time_idx=time_idx)

                    dataset.step()

                st, sc = evaluator_now.compute_accumulated_miou()
                print("Immediate mIoU (accumulated):")
                print("student-teacher: {:.3f}   |   student-carla: {:.3f}".format(st, sc))
                evaluator_now._save_CMS()
