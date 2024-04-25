import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--name", help="wandb run name, and also subfolder for saving")
parser.add_argument("--save_folder", help="Path to save model weights", required=True)

parser.add_argument("--dataset", help="Path to dataset root", type=str, required=True)
#parser.add_argument("--dynamic_weather", action='store_true', help='set to True if dynamic weather dataset is in use', default=False)
parser.add_argument("--ego_mask_path", help="Path to ego vehicle mask", default='ego_mask.png')
parser.add_argument("--num_classes", help="Number of classes", type=int, default=19)
parser.add_argument("--seed", help="experiment's seed", type=int, default=0)
parser.add_argument("--start", help="Only select a subset of the dataset (starting point)", type=int, default=0)
parser.add_argument("--stop", help="Only select a subset of the dataset (end point)", type=int, default=60*60*2)  # correspond to 2 first hours
parser.add_argument("--numworkers", help="number of workers for the dataloader", type=int, default=12)
parser.add_argument("--batchsize", help="Batch size", type=int, default=25)
parser.add_argument("--lr", help="Learning rate", type=float, default=0.0001)
parser.add_argument("--n_epochs", help="Number of training epochs", type=int, default=3)

parser.add_argument("--scheduler", help="Learning rate scheduler type", type=str, choices=['constant', 'cosine'], default='constant')
parser.add_argument("--validation_fraction", help="Float in [0,1] that indicate the proportion of images to keep for "
                                                  "a validation set", type=float, default=0.1)
parser.add_argument("--cell_type", help="Describe environment division", type=str, choices=['common', 'specific', 'spatial', 'weather', 'daylight', 'weatherdaylight'])
parser.add_argument("--supervision", help="Select Carla or teacher for supervision during learning", type=str, choices=['teacher', 'carla'])

parser.add_argument("--zones_to_consider", nargs='+', type=str, choices=['community buildings', 'countryside', 'forest', 'high density residential', 'highway', 'low density residential', 'rural farmland', 'all'], default=['all'])
parser.add_argument("--weathers_to_consider", nargs='+', type=str, choices=['clear', 'foggy', 'rainy', 'all'], default=['all'])
parser.add_argument("--periods_to_consider", nargs='+', type=str, choices=['day', 'night', 'all'], default=['all'])


args = parser.parse_args()
