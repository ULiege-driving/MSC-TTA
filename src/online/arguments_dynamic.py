import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--name", help="wandb run name")
parser.add_argument("--save_folder", help="Path to save confusion matrix and others", required=True)

parser.add_argument("--dataset", help="Path to dataset root", required=True)
parser.add_argument("--ego_mask_path", help="Path to ego vehicle mask", default='ego_mask.png')
parser.add_argument("--num_classes", help="Number of classes", type=int, default=19)
parser.add_argument("--seed", help="experiment's seed", type=int, default=0)
parser.add_argument("--start", help="Only select a subset of the dataset (starting point)", type=int, default=60*60*2)
parser.add_argument("--stop", help="Only select a subset of the dataset (end point)", type=int, default=int(1e10))
parser.add_argument("--numworkers", help="number of workers for the dataloader", type=int, default=8)
parser.add_argument("--batchsize", help="Batch size", type=int, default=8)
parser.add_argument("--lr", help="Learning rate", type=float, default=0.0001)
parser.add_argument("--device", help="Device to use (use cuda:x or cpu)", default="cuda:0")

parser.add_argument("--zones_to_consider", nargs='+', type=str, choices=['community buildings', 'countryside', 'forest', 'high density residential', 'highway', 'low density residential', 'rural farmland', 'all'], default=['all'])
parser.add_argument("--weathers_to_consider", nargs='+', type=str, choices=['clear', 'foggy', 'rainy', 'all'], default=['all'])
parser.add_argument("--periods_to_consider", nargs='+', type=str, choices=['day', 'night', 'all'], default=['all'])

parser.add_argument("--vehicle_idx", type=int, default=None, help="specify if a single vehicle should be considered for parallel execution")

parser.add_argument("--save_CM", action='store_true', default=False)

parser.add_argument("--pretraining", help="Specify the weight initialization", default="scratch", choices=['scratch', 'general', 'specific', 'spatial', 'weather', 'daylight', 'weatherdaylight'])
parser.add_argument("--pretraining_path", help="Path to pretraining weights folder", default=None)

parser.add_argument("--cell_type", help="Describe environment partition", type=str, choices=['common', 'specific', 'standalone', 'spatial', 'weather', 'daylight', 'weatherdaylight'], required=True)
parser.add_argument("--supervision", help="Select Carla or teacher for supervision during learning", type=str, choices=['teacher', 'carla'], required=True)
parser.add_argument("--teacher_type", help="precise teacher model", type=str, choices=['segformer', 'mask2former'], default='segformer')

# Type of training
parser.add_argument("--trainer", help="Trainer type", default='')


parser.add_argument("--baseline", choices=[None, "TENT", "SAR"], default=None)


parser.add_argument("--C", type=int, default=150)
parser.add_argument("--k", type=int, default=100)


parser.add_argument("--buffer_size", help="image bank size", default=100,
                    type=int)
parser.add_argument("--student_subset", help="only upload one out of X images from each student", default=3,
                    type=int)
parser.add_argument("--training_step", help="time (in [s]) between two training step", default=30,
                    type=int)
parser.add_argument("--training_delay", help="time (in [s]) that takes the model to train 1 epoch (simulate training time)", default=0,
                    type=int)
parser.add_argument("--evaluate_span", help="time window (in [s]) on which to test the model", default=30,
                    type=int)
parser.add_argument("--forward_jump", help="time (in [s]) to add from current time to get the beginning of the forward test window", default=300,
                    type=int)
parser.add_argument("--backward_jump", help="time (in [s]) to remove from current time to get the beginning of the backward test window", default=330,
                    type=int)



args = parser.parse_args()
