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
import time

import wandb


def write_log_args(args, save_path):
    f = open(os.path.join(save_path, 'arguments'), 'w')
    for key, value in vars(args).items():
        f.write(f'{key} = {value}\n')
    f.close()


def get_weights(args, zone_name, weather_name, period_name, device):
    if args.cell_type == 'standalone':
        weights_path = os.path.join(args.pretraining_path, 'dynamic_common', 'common', 'pretrained.pth')
    elif args.pretraining == 'spatial':
        weights_path = os.path.join(args.pretraining_path, 'dynamic_spatial', zone_name, 'common', 'pretrained.pth')
    elif args.pretraining == 'weather':
        weights_path = os.path.join(args.pretraining_path, 'dynamic_weather', 'weather', weather_name, 'pretrained.pth')
    elif args.pretraining == 'daylight':
        weights_path = os.path.join(args.pretraining_path, 'dynamic_daylight', 'daylight', period_name, 'pretrained.pth')
    elif args.pretraining == 'weatherdaylight':
        weights_path = os.path.join(args.pretraining_path, 'dynamic_weatherdaylight', 'weatherdaylight', weather_name,
                                    period_name, 'pretrained.pth')
    elif args.pretraining == 'specific':
        weights_path = os.path.join(args.pretraining_path, 'dynamic_specific', zone_name, weather_name, period_name,
                                    'pretrained.pth')

    elif args.pretraining == 'general':
        weights_path = os.path.join(args.pretraining_path, 'dynamic_common', 'common', 'pretrained.pth')

    return torch.load(weights_path, map_location=device)


if __name__ == "__main__":

    """ Import args dict """
    print("-" * 20, "Initialization", "-" * 20)

    from arguments_dynamic import args

    args.learning = 'online'
    args.dynamic_weather = True
    args.name = 'online' + '_' + 'dynamic'
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

    if args.cell_type == 'standalone':
        if args.pretraining == 'specific' or args.pretraining in ['spatial', 'weather', 'daylight', 'weatherdaylight']:
            print('Not implemented yet, skipping this run')
            exit()
        vehicle_tag_list = [d for d in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, d))]
        vehicle_tag_list.sort()
        if args.vehicle_idx is not None:
            vehicle_tag_list = [vehicle_tag_list[args.vehicle_idx]]

        n_envs = len(vehicle_tag_list)
        n_zones = n_envs
        n_weathers = 1
        n_periods = 1


    training_step = args.training_step  # time (in [s]) between two training step
    training_delay = args.training_delay  # time (in [s]) that takes the model to train 1 epoch (simulate training time)

    evaluate_span = args.evaluate_span  # time window (in [s]) on which to test the model
    forward_jump = args.forward_jump  # time (in [s]) to add from current time to get the beginning of the forward test window
    backward_jump = args.backward_jump  # time (in [s]) to remove from current time to get the beginning of the backward test window

    """ Prepare output folder """

    os.makedirs(args.save_folder, exist_ok=True)

    """ Set determinism for reproducibility """

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    """ Training-related initialization """

    from models.tinynet import TinyNet

    device = args.device

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    network_trainer = None

    from training.classic import ClassicNetworkTrainer
    Trainer = ClassicNetworkTrainer

    dataset = None
    from datasets.fifo import FifoDataset
    Dataset = FifoDataset

    """ Benchmark for each environment """
    """
    wandb.login()
    wandb.init(project='', entity='', config=args)
    wandb.run.name = args.name
    """

    fuse = True
    for zone_idx in range(n_zones):
        for weather_idx in range(n_weathers):
            for period_idx in range(n_periods):

                zone_name = world.ZONES[zones[zone_idx]]['name'] if n_zones > 1 else 'common'
                weather_name = world.WEATHERS[weathers[weather_idx]]['name'] if n_weathers > 1 else 'common'
                period_name = world.PERIODS[periods[period_idx]]['name'] if n_periods > 1 else 'common'
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
                    # args.save_path_env = os.path.join(args.save_folder, args.name, zone_name, 'common')
                    args.env_name_key = f'{zone_name}'
                    if zone_name not in args.zones_to_consider and args.zones_to_consider[0] != 'all':
                        print("not in list, skipping...")
                        continue
                elif args.cell_type == 'weather':
                    print(f"Weather '{weather_name}' ({weather_idx + 1}/{n_weathers})")
                    # args.save_path_env = os.path.join(args.save_folder, args.name, zone_name, 'common')
                    args.env_name_key = f'{weather_name}'
                    if weather_name not in args.weathers_to_consider and args.weathers_to_consider[0] != 'all':
                        print("not in list, skipping...")
                        continue
                elif args.cell_type == 'daylight':
                    print(f"Period '{period_name}' ({period_idx + 1}/{n_periods})")
                    # args.save_path_env = os.path.join(args.save_folder, args.name, zone_name, 'common')
                    args.env_name_key = f'{period_name}'
                    if period_name not in args.periods_to_consider and args.periods_to_consider[0] != 'all':
                        print("not in list, skipping...")
                        continue
                elif args.cell_type == 'weatherdaylight':
                    print(f"Weather '{weather_name}' ({weather_idx + 1}/{n_weathers})")
                    print(f"Period '{period_name}' ({period_idx + 1}/{n_periods})")
                    # args.save_path_env = os.path.join(args.save_folder, args.name, zone_name, 'common')
                    args.env_name_key = f'{weather_name}_{period_name}'
                    if (weather_name not in args.weathers_to_consider and args.weathers_to_consider[0] != 'all') or (
                            period_name not in args.periods_to_consider and args.periods_to_consider[0] != 'all'):
                        print("not in list, skipping...")
                        continue
                elif args.cell_type == 'common':
                    print(f"All environments considered as one")
                    # args.save_path_env = os.path.join(args.save_folder, args.name, 'common')
                    args.env_name_key = 'common'
                elif args.cell_type == 'standalone':
                    print(f"Vehicle '{vehicle_tag_list[zone_idx]}' ({zone_idx + 1}/{n_envs})")
                    args.env_name_key = vehicle_tag_list[zone_idx]
                print('-' * 50)

                if fuse:
                    sf = os.path.join(args.save_folder)
                    sup = args.teacher_type if args.supervision == 'teacher' else args.supervision
                    subfolder = f'{args.cell_type}-{args.learning}-{args.pretraining}-{sup}/{args.env_name_key}'
                    sp = os.path.join(sf, subfolder)
                    if os.path.exists(sp):
                        if os.listdir(sp):
                            exit()
                    fuse = False

                # Definition of optimization objects

                model = TinyNet(num_classes=args.num_classes).to(
                    device)
                if not args.pretraining == "scratch":
                    state_dict = get_weights(args, zone_name, weather_name, period_name, device=device)
                    model.load_state_dict(state_dict)

                scaler = torch.cuda.amp.GradScaler()
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                trainer = Trainer(model, optimizer, scaler, criterion, device, args)

                kwargs = {'model': model, 'optimizer': optimizer, 'scaler': scaler, 'criterion': criterion, 'Trainer': Trainer}

                # dataset instantiation
                env_id = (zones[zone_idx], weathers[weather_idx], periods[period_idx])
                dataset = Dataset(args, vehicle_tag_list[zone_idx] if args.cell_type == 'standalone' else env_id,
                                  args.start, args.stop, **kwargs)

                evaluator_now = SegmentationEvaluator(n_classes=args.num_classes, args=args, keyword='now')
                # evaluator_before = SegmentationEvaluator(n_classes=args.num_classes, args=args, keyword='backward')
                evaluator_after = SegmentationEvaluator(n_classes=args.num_classes, args=args, keyword='forward')

                first_timestamp = dataset.first_timestamp
                last_timestamp = dataset.last_timestamp
                if first_timestamp is None or last_timestamp is None:
                    print('empty...')
                    continue

                print(f'FIRST - LAST : {first_timestamp} - {last_timestamp}')

                for time_idx in range(first_timestamp, last_timestamp + 1, training_step):
                    #########################################
                    # Gather data in [t - training_step, t] #
                    #########################################
                    print(f"Gather data in [{time_idx - training_step}, {time_idx}]")
                    dataloader = dataset.create_trainset()  # return None if not ready?

                    ###############################
                    # Train model on current data #
                    ###############################
                    # print(f"Training on current data")

                    old_model = copy.deepcopy(model)

                    if dataset.is_different:
                        trainer.train_epoch(dataloader)  # does not train if None input

                    ############################################
                    # Evaluate model in [t, t + evaluate_span] #
                    ############################################
                    print(f"NOW : Evaluate model in [{time_idx}, {time_idx + evaluate_span}]")

                    # use old model in [t, t + training_delay]
                    if training_delay > 0:
                        dataloader = dataset.create_testset(time_idx, time_idx + training_delay)
                        evaluator_now.evaluate(args, old_model, dataloader, keyword='now', save=args.save_CM,
                                               time_idx=time_idx)

                    # use new model in [t + training_delay, t + evaluate_span]
                    dataloader = dataset.create_testset(time_idx + training_delay, time_idx + evaluate_span)
                    evaluator_now.evaluate(args, model, dataloader, keyword='now', save=args.save_CM, time_idx=time_idx)

                    #######################################################################
                    # Evaluate model in [t + training_delay - 1 - backward_jump, t - 1] #
                    #######################################################################
                    '''
                    print(
                        f"BACKWARD : Evaluate model in [{time_idx - backward_jump - 1}, {time_idx - backward_jump - 1 + evaluate_span}]")

                    dataloader = dataset.create_testset(time_idx - backward_jump - 1,
                                                        time_idx - backward_jump - 1 + evaluate_span)
                    evaluator_before.evaluate(args, model, dataloader, keyword='backward', save=args.save_CM,
                                              time_idx=time_idx)
                    '''
                    #################################################################################
                    # Evaluate model in [t + training_step + 1, t + training_step + forward_jump] #
                    #################################################################################
                    print(
                        f"FORWARD : Evaluate model in [{time_idx + forward_jump}, {time_idx + forward_jump + evaluate_span}]")

                    dataloader = dataset.create_testset(time_idx + forward_jump,
                                                        time_idx + forward_jump + evaluate_span)
                    evaluator_after.evaluate(args, model, dataloader, keyword='forward', save=args.save_CM,
                                             time_idx=time_idx)

                    dataset.step()

                    st, sc = evaluator_now.compute_accumulated_miou()
                    print("Immediate mIoU (accumulated):")
                    print("student-teacher: {:.3f}   |   student-carla: {:.3f}".format(st, sc))

                    st, sc = evaluator_after.compute_accumulated_miou()
                    print("Future mIoU (accumulated):")
                    print("student-teacher: {:.3f}   |   student-carla: {:.3f}".format(st, sc))

                    print("-" * 20)

                evaluator_now._save_CMS()
                # evaluator_before._save_CMS()
                evaluator_after._save_CMS()
