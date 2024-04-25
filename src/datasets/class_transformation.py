import numpy as np
from torchvision.transforms import functional as F
from datasets.classes import *
import torch
from PIL import Image
import torchvision.transforms as transforms



class ProjectCarlaToCommonLabels:
    def __init__(self, carla_labels=carla_labels, common_labels=common_labels):
        # Create mappings
        self.carla_rgb_to_id = {label.color: label.id for label in carla_labels}
        self.common_id_to_rgb = {label.id: label.color for label in common_labels}

        self.mapping = {}

        # Direct mappings from Carla to common_labels based on name
        for common_label in common_labels:
            for carla_label in carla_labels:
                if carla_label.name == common_label.name:
                    self.mapping[carla_label.id] = common_label.id

        # Custom mappings from the given specification
        other_id = next(label.id for label in carla_labels if label.name == "other")
        water_id = next(label.id for label in carla_labels if label.name == "water")
        road_line_id = next(label.id for label in carla_labels if label.name == "road line")

        # static_id = next(label.id for label in common_labels if label.name == "static")
        road_id = next(label.id for label in common_labels if label.name == "road")
        unlabeled_id = next(label.id for label in common_labels if label.name == "unlabeled")

        self.mapping[other_id] = unlabeled_id
        self.mapping[water_id] = unlabeled_id
        self.mapping[road_line_id] = road_id

    def __call__(self, mask):
        """
        Args:
            mask (PIL Image): Carla RGB mask image to be transformed.

        Returns:
            PIL Image: RGB mask image transformed to common labels.
        """
        mask = np.array(mask)[:, :, :3]
        h, w, _ = mask.shape
        mapped_mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Convert entire RGB mask to Carla ID mask
        id_mask = np.zeros((h, w), dtype=np.uint8)
        for rgb, label_id in self.carla_rgb_to_id.items():
            id_mask[np.where((mask == np.array(rgb)).all(axis=2))] = label_id

        # Map Carla ID to common ID, and then to RGB
        for carla_id, common_id in self.mapping.items():
            rgb_value = self.common_id_to_rgb[common_id]
            mapped_mask[np.where(id_mask == carla_id)] = rgb_value

        return F.to_pil_image(mapped_mask)


class ProjectCityScapeToCommonLabels:
    def __init__(self, cityscape_labels=cityscape_labels, common_labels=common_labels):
        # Create mappings
        self.cityscape_rgb_to_id = {label.color: label.id for label in cityscape_labels}
        self.common_id_to_rgb = {label.id: label.color for label in common_labels}

        self.mapping = {}

        # Direct mappings from CityScape to common_labels based on name
        for common_label in common_labels:
            for cityscape_label in cityscape_labels:
                if cityscape_label.name == common_label.name:
                    self.mapping[cityscape_label.id] = common_label.id

        # Custom mappings from the given specification
        custom_mapping_labels = [
            'ego vehicle', 'rectification border', 'out of roi', 'tunnel', 'caravan', 'trailer',
            'train'
        ]

        unlabeled_id = next(label.id for label in common_labels if label.name == "unlabeled")
        for custom_label in custom_mapping_labels:
            cityscape_id = next(label.id for label in cityscape_labels if label.name == custom_label)
            self.mapping[cityscape_id] = unlabeled_id

    def __call__(self, mask):
        """
        Args:
            mask (PIL Image): CityScape RGB mask image to be transformed.

        Returns:
            PIL Image: RGB mask image transformed to common labels.
        """
        mask = np.array(mask)[:, :, :3]
        h, w, _ = mask.shape
        mapped_mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Convert entire RGB mask to CityScape ID mask
        id_mask = np.zeros((h, w), dtype=np.uint8)
        for rgb, label_id in self.cityscape_rgb_to_id.items():
            id_mask[np.where((mask == np.array(rgb)).all(axis=2))] = label_id

        # Map CityScape ID to common ID, and then to RGB
        for cityscape_id, common_id in self.mapping.items():
            rgb_value = self.common_id_to_rgb[common_id]
            mapped_mask[np.where(id_mask == cityscape_id)] = rgb_value

        return F.to_pil_image(mapped_mask)


class ProjectCommonLabelsToCItyscapes20:
    def __init__(self, mapping=mapping_cc):
        # Create mappings

        self.mapping = mapping

    def __call__(self, mask):
        """
        Args:
            mask (PIL Image): CityScape RGB mask image to be transformed.

        Returns:
            PIL Image: RGB mask image transformed to common labels.
        """
        h, w = mask.shape
        mapped_mask = np.zeros((h, w), dtype=np.int32)
        mapped_mask[mask==255] = -1
        for cls in self.mapping.keys():
            mapped_mask[mask==cls] = self.mapping[cls]

        return mapped_mask


class RGBToID:
    def __init__(self, labels=common_labels):
        self.rgb_to_id = {label.color: label.id for label in labels}

    def __call__(self, mask):
        mask = np.array(mask)[:, :, :3]
        id_mask = np.zeros(mask.shape[:2], dtype=np.int64)
        for rgb, label_id in self.rgb_to_id.items():
            id_mask[np.where((mask == np.array(rgb)).all(axis=2))] = label_id
        return id_mask


class OneHotEncode:
    def __init__(self, num_classes=25):
        self.num_classes = num_classes

    def __call__(self, mask):
        # Convert to tensor
        mask_tensor = torch.tensor(mask)
        # One-hot encode
        one_hot = torch.nn.functional.one_hot(mask_tensor, self.num_classes).permute(2, 0, 1).float()
        return one_hot


class UpdateMaskBasedOnTrainId:
    def __init__(self, labels=common_labels):
        # Map each label's ID to its trainId
        self.id_to_trainId = {label.id: label.trainId for label in labels}

    def __call__(self, mask):
        """
        Args:
            mask (numpy ndarray or torch.Tensor): 2D tensor/array representing the mask of IDs.

        Returns:
            Updated mask where IDs are replaced by -1 if the corresponding trainId is 255.
        """
        # Convert the mask to numpy if it's a tensor
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        # Iterate over the unique values in the mask (which should be more efficient than iterating over all pixels)
        for unique_id in np.unique(mask):
            if self.id_to_trainId[unique_id] == 255:
                mask[mask == unique_id] = -1

        return mask


class UpdateMaskBasedOnTrainId_Houyon:
    def __init__(self, labels=cityscape_labels_raw, mapping=mapping_inv):
        # Map each label's ID to its trainId

        self.id_to_trainId = {cls: labels[mapping[cls]].trainId for cls in mapping.keys()}

    def __call__(self, mask):
        """
        Args:
            mask (numpy ndarray or torch.Tensor): 2D tensor/array representing the mask of IDs.

        Returns:
            Updated mask where IDs are replaced by -1 if the corresponding trainId is 255.
        """
        # Convert the mask to numpy if it's a tensor
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        mask = mask.astype(np.int32)

        # Iterate over the unique values in the mask (which should be more efficient than iterating over all pixels)
        for unique_id in np.unique(mask):
            if self.id_to_trainId[unique_id] == 255:
                mask[mask == unique_id] = -1

        return mask


class UpdateMaskBasedOnBinaryImage:
    def __init__(self, binary_img_path, COMMON_SIZE):
        binary_img = Image.open(binary_img_path)
        '''
        T = transforms.Resize(COMMON_SIZE, interpolation=transforms.InterpolationMode.NEAREST)
        binary_img = T(binary_img)
        '''
        self.binary_img = np.array(binary_img)[:, :, 0]

    def __call__(self, mask):
        """
        Args:
            mask (numpy ndarray or torch.Tensor): 2D tensor/array representing the mask of IDs.
            binary_img_path (str): Path to the binary PNG image.

        Returns:
            Updated mask where IDs are replaced by -1 if the corresponding pixel in the binary image is white.
        """
        # Load the binary image

        # Convert the mask to numpy if it's a tensor
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        white_pixels = self.binary_img == 255

        # Update the mask
        mask[white_pixels] = -1

        return mask
