import os
import json
from glob import glob
from datasets.zones_dynamic import Zones


def get_all_env_frames(root_folder, start_index, end_index, env_id, args, cached=False):
    image_paths = []
    pseudo_paths = []
    semantic_mask_paths = []
    timestamps = []

    # pseudo_folder = 'pseudo' if args.teacher_type == 'segformer' else f'pseudo_{args.teacher_type}'
    pseudo_folder = 'pseudo'

    gt_keyword = 'semantic_masks'

    timestamp_dilatation = 1


    if args.dynamic_weather:
        zone_id, weather_id, period_id = env_id
    else:
        zone_id = env_id[0]
        weather_id = -1
        period_id = -1

    Z = Zones(root_folder, dynamic_weather=args.dynamic_weather)

    ext = 'npz' if cached else 'png'

    # iterate over all subfolders in the root folder
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # load the info json file
        with open(os.path.join(subfolder_path, f'{subfolder}.json')) as json_file:
            info = json.load(json_file)

        # check if the timestamp is in the given range

        seq_start = info['timestamp']
        seq_end = seq_start + info['seq_length']

        check_start = max(seq_start, start_index)
        check_end = min(seq_end, end_index)

        if check_end - check_start < 0:
            continue

        # load the gnss json file
        with open(os.path.join(subfolder_path, 'gnss.json')) as json_file:
            gnss = json.load(json_file)

        # iterate over all frames in the gnss file
        for frame, frame_info in gnss.items():
            if not check_start <= int(frame) + seq_start <= check_end:
                continue

            id = Z.get_zone(subfolder, frame)

            if args.dynamic_weather:
                z_id = int(id[0])
                w_id = int(id[1])
                p_id = int(id[2])
            else:
                z_id = int(id)
                w_id = -1
                p_id = -1
            # check if the coordinates correspond to the given env_id
            if (zone_id in (z_id, -1)) and (weather_id in (w_id, -1)) and (period_id in (p_id, -1)):
                img_path = glob(os.path.join(subfolder_path, 'images', '*', f'{int(frame):06d}.png'))
                pgt_path = glob(
                    os.path.join(subfolder_path, pseudo_folder, '*', f'{int(frame):06d}.{ext}'))  # soft label to implement
                gt_path = glob(os.path.join(subfolder_path, gt_keyword, '*', f'{int(frame):06d}.{ext}'))

                if img_path:
                    image_paths.append(img_path[0])
                    pseudo_paths.append(pgt_path[0])
                    semantic_mask_paths.append(gt_path[0])
                    timestamps.append((int(frame)+seq_start) * timestamp_dilatation)

    return image_paths[::args.student_subset], pseudo_paths[::args.student_subset], semantic_mask_paths[::args.student_subset], timestamps[::args.student_subset], None, None


def get_all_vehicle_frames(root_folder, start_index, end_index, vehicle_folder, args, cached=False):
    image_paths = []
    pseudo_paths = []
    semantic_mask_paths = []
    timestamps = []

    #pseudo_folder = 'pseudo' if args.teacher_type == 'segformer' else f'pseudo_{args.teacher_type}'
    pseudo_folder = 'pseudo'

    gt_keyword = 'semantic_masks'
    timestamp_dilatation = 1

    ext = 'npz' if cached else 'png'

    # iterate over all subfolders in the root folder
    subfolder_path = os.path.join(root_folder, vehicle_folder)

    # load the info json file
    with open(os.path.join(subfolder_path, f'{vehicle_folder}.json')) as json_file:
        info = json.load(json_file)

    # check if the timestamp is in the given range

    seq_start = info['timestamp']
    seq_end = seq_start + info['seq_length']
    check_start = max(seq_start, start_index)
    check_end = min(seq_end, end_index)

    # load the gnss json file
    with open(os.path.join(subfolder_path, 'gnss.json')) as json_file:
        gnss = json.load(json_file)

    # iterate over all frames in the gnss file
    for frame, frame_info in gnss.items():
        if not check_start <= int(frame) + seq_start <= check_end:
            continue

        img_path = glob(os.path.join(subfolder_path, 'images', '*', f'{int(frame):06d}.png'))
        pgt_path = glob(
            os.path.join(subfolder_path, pseudo_folder, '*', f'{int(frame):06d}.{ext}'))  # soft label to implement
        gt_path = glob(os.path.join(subfolder_path, gt_keyword, '*', f'{int(frame):06d}.{ext}'))

        if img_path:
            image_paths.append(img_path[0])
            pseudo_paths.append(pgt_path[0])
            semantic_mask_paths.append(gt_path[0])
            timestamps.append((int(frame) + seq_start) * timestamp_dilatation)

    return image_paths[::args.student_subset], pseudo_paths[::args.student_subset], semantic_mask_paths[::args.student_subset], timestamps[::args.student_subset], check_start, check_end


def get_subsets_length(start, length, image_paths, pseudo_paths, semantic_mask_paths, timestamps):
    # Initialize the subsets
    subset_image_paths = []
    subset_pseudo_paths = []
    subset_semantic_mask_paths = []
    subset_timestamps = []

    # Iterate over the four lists simultaneously
    for i, timestamp in enumerate(timestamps):
        if timestamp > start:

            # Once you found it, take the next n items from each list
            end = min(i + length, len(timestamps))
            subset_image_paths = image_paths[i:end]
            subset_pseudo_paths = pseudo_paths[i:end]
            subset_semantic_mask_paths = semantic_mask_paths[i:end]
            subset_timestamps = timestamps[i:end]
            break

    return subset_image_paths, subset_pseudo_paths, subset_semantic_mask_paths, subset_timestamps

def cached_get_subsets_length(start, length, timestamps):
    subset_timestamps = []

    # Iterate over the timestamps
    for i, timestamp in enumerate(timestamps):
        if timestamp > start:
            # Once you find the starting timestamp, take the next n timestamps
            end = min(i + length, len(timestamps))
            subset_timestamps = timestamps[i:end]
            break

    return subset_timestamps


def get_subsets_period(start, stop, image_paths, pseudo_paths, semantic_mask_paths, timestamps):
    # Initialize the subsets
    subset_image_paths = []
    subset_pseudo_paths = []
    subset_semantic_mask_paths = []
    subset_timestamps = []

    # Iterate over the four lists simultaneously
    for img_path, pseudo_path, mask_path, timestamp in zip(image_paths, pseudo_paths, semantic_mask_paths, timestamps):
        # Check if the timestamp is within the specified range
        if start <= timestamp <= stop:
            # If it is, add the paths to the subsets
            subset_image_paths.append(img_path)
            subset_pseudo_paths.append(pseudo_path)
            subset_semantic_mask_paths.append(mask_path)
            subset_timestamps.append(timestamp)

    return subset_image_paths, subset_pseudo_paths, subset_semantic_mask_paths, subset_timestamps

def cached_get_subsets_period(start, stop, timestamps):
    # Initialize the subsets
    subset_timestamps = []

    # Iterate over the four lists simultaneously
    for timestamp in timestamps:
        # Check if the timestamp is within the specified range
        if start <= timestamp <= stop:
            # If it is, add the paths to the subsets

            subset_timestamps.append(timestamp)

    return subset_timestamps



