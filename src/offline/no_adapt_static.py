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

import wandb


def write_log_args(args, save_path):
    f = open(os.path.join(save_path, 'arguments'), 'w')
    for key, value in vars(args).items():
        f.write(f'{key} = {value}\n')
    f.close()


def get_weights(args, env_name, device):
    if args.cell_type == 'standalone':
        weights_path = os.path.join(args.pretraining_path, 'common', 'pretrained.pth')
    elif args.cell_type == 'specific':
        weights_path = os.path.join(args.pretraining_path, 'static_spatial', env_name, 'pretrained.pth')

    elif args.cell_type == 'common':
        weights_path = os.path.join(args.pretraining_path, 'common', 'pretrained.pth')

    return torch.load(weights_path, map_location=device)


if __name__ == "__main__":

    """ Import args dict """
    print("-" * 20, "Initialization", "-" * 20)

    from arguments_static import args

    args.learning = 'offline'
    args.dynamic_weather = False
    args.name = 'offline' + '_' + 'static'
    args.onlinedataset = 'fifo'
    args.dataset = os.path.join(args.dataset, 'static_weather')


    """ Prepare world-related variables """
    if args.cell_type == 'specific':
        world = Zones(args.dataset)
        envs = sorted(world.ZONES, key=lambda x: world.ZONES[x]['name'])
        n_envs = len(envs)
    elif args.cell_type == 'common':
        world = Zones(args.dataset)
        envs = sorted(world.ZONES, key=lambda x: world.ZONES[x]['name'])
        n_envs = 1

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


    dataset = None
    from datasets.fifo import FifoDataset

    """ Benchmark for each environment """
    """
    wandb.login()
    wandb.init(project='', entity='', config=args)
    wandb.run.name = args.name
    """

    fuse = True
    for env_idx in range(n_envs):

        print('-' * 30)
        if args.cell_type == 'specific':
            print(f"Environment '{world.ZONES[envs[env_idx]]['name']}' ({env_idx + 1}/{n_envs})")
            args.env_name_key = world.ZONES[envs[env_idx]]['name']

            if (not args.envs_to_consider[0] == 'all') and (
                    world.ZONES[envs[env_idx]]['name'] not in args.envs_to_consider):
                print("not in list, skipping...")
                continue
        elif args.cell_type == 'common':
            print(f"All environments considered as one")
            args.env_name_key = 'common'
        print('-' * 30)

        # dataset instantiation
        # print('creating cached dataset ...')
        dataset = FifoDataset(args, (envs[env_idx], -1, -1) if args.cell_type == 'specific' else -1, args.start,
                              args.stop)
        if not dataset.timestamps:
            print('No frame in range, skipping ...')
            continue
        if fuse:
            sf = os.path.join(args.save_folder)
            subfolder = f'{args.cell_type}-{args.learning}-{args.supervision}/{args.env_name_key}'
            sp = os.path.join(sf, subfolder)
            if os.path.exists(sp):
                if os.listdir(sp):
                    exit()
            fuse = False

        evaluator_now = SegmentationEvaluator(n_classes=args.num_classes, args=args, keyword='now')

        # Definition of the optimizer

        model = TinyNet(num_classes=args.num_classes).to(device)
        state_dict = get_weights(args,
                                 env_name=None if args.cell_type == 'standalone' else world.ZONES[envs[env_idx]]['name'],
                                 device=device)
        model.load_state_dict(state_dict)

        first_timestamp = dataset.first_timestamp
        last_timestamp = dataset.last_timestamp

        evaluate_span = 30

        print(f'FIRST - LAST : {first_timestamp} - {last_timestamp}')

        for time_idx in range(args.start, last_timestamp + 1, 30):
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
