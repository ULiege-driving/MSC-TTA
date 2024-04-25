import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--name", help="wandb run name", required=False)
parser.add_argument("--save_folder", help="Path to save confusion matrix and others", required=True)

parser.add_argument("--dataset", help="Path to dataset root", required=True)
parser.add_argument("--ego_mask_path", help="Path to ego vehicle mask", default='ego_mask.png')
parser.add_argument("--num_classes", help="Number of classes", type=int, default=19)
parser.add_argument("--seed", help="experiment's seed", type=int, default=0)
parser.add_argument("--start", help="Only select a subset of the dataset (starting point)", type=int, default=60*60*2)
parser.add_argument("--stop", help="Only select a subset of the dataset (end point)", type=int, default=int(1e10))
parser.add_argument("--numworkers", help="number of workers for the dataloader", type=int, default=12)
parser.add_argument("--batchsize", help="Batch size", type=int, default=25)
parser.add_argument("--device", help="Device to use (use cuda:x or cpu)", default="cuda:0")
parser.add_argument("--envs_to_consider", nargs='+', type=str, choices=['community buildings', 'countryside', 'forest', 'high density residential', 'highway', 'low density residential', 'rural farmland', 'all'], default='all')
parser.add_argument("--save_CM", action='store_true', default=False)

#parser.add_argument("--pretraining", help="Specify the weight initialization", default="general", choices=['general', 'spatial'])
parser.add_argument("--pretraining_path", help="Path to pretraining weights folder", default=None)

parser.add_argument("--cell_type", help="Describe environment partition", type=str, choices=['common', 'spatial'], required=True)
parser.add_argument("--supervision", help="Select Carla or teacher for supervision during learning", type=str, choices=['teacher', 'carla'], required=True)


parser.add_argument("--buffer_size", help="image bank size", default=1,
                    type=int)
parser.add_argument("--student_subset", help="only upload one out of X images from each student", default=3,
                    type=int)
parser.add_argument("--training_step", help="time (in [s]) between two training step", default=30,
                    type=int)
parser.add_argument("--teacher_type", help="precise teacher model", type=str, choices=['segformer', 'mask2former'], default='segformer')

args = parser.parse_args()
