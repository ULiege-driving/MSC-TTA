import sys
import os

sys.path.append(os.path.abspath('..'))
import math
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
import wandb
from PIL import Image
import torchvision.transforms as transforms
import warnings
from filelock import FileLock
from tqdm import tqdm
import time
import zipfile



class SegmentationEvaluator:
    def __init__(self, n_classes=19, args=None, keyword=None):
        self.n_classes = n_classes
        self.confusion_matrix_st = np.zeros((n_classes, n_classes))
        self.confusion_matrix_sc = np.zeros((n_classes, n_classes))
        self.valid_zone = None
        self.args = args
        self.cm_cache = {}

        self.save_folder = os.path.join(self.args.save_folder)
        self.subfolder = f'{args.cell_type}-{args.learning}-{args.pretraining}-{args.supervision}/{self.args.env_name_key}'
        self.save_path = os.path.join(self.save_folder, self.subfolder)
        self.keyword = keyword


        if self.args.ego_mask_path is not None:
            binary_img = Image.open(args.ego_mask_path)
            '''
            T = transforms.Resize(COMMON_SIZE, interpolation=transforms.InterpolationMode.NEAREST)
            binary_img = T(binary_img)
            '''
            binary_img = np.array(binary_img) == 0
            self.valid_zone = binary_img.astype(float)[:, :, 0]

        self.global_counter = 0  # for wandb !

    def update(self, CM_ST, CM_SC, weight=1.):
        self.confusion_matrix_st += weight * CM_ST
        self.confusion_matrix_sc += weight * CM_SC

    def compute_balanced_matrix(self, CM):
        return CM / np.expand_dims(CM.sum(axis=1), 1).repeat(self.n_classes, 1)

    def compute_accuracy(self, CM, balanced=False):

        mat = self.compute_balanced_matrix(CM) if balanced else CM

        return mat.diagonal().sum() / mat.sum()

    def compute_iou_c(self, CM, cls, balanced=False):

        mat = self.compute_balanced_matrix(CM) if balanced else CM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return mat[cls, cls] / (mat[cls, :].sum() + mat[:, cls].sum() - mat[cls, cls])

    def compute_miou(self, CM, balanced=False):

        miou = 0.
        remove = 0.
        for i in range(1, self.n_classes):
            tmp_miou = self.compute_iou_c(CM, i, balanced)
            if not math.isnan(tmp_miou):
                miou += tmp_miou
            else:
                remove += 1

        if self.n_classes == remove:
            return math.nan

        return miou / (self.n_classes - remove)

    def _save_CMS(self):
        if not self.args.save_CM:
            return

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        file_path = os.path.join(self.save_path, f'{self.keyword}.npz')
        lock_path = f"{file_path}.lock"

        with FileLock(lock_path):

            if os.path.exists(file_path):
                # Load existing file and its data
                data = dict(np.load(file_path, allow_pickle=True))
                data.update(self.cm_cache)
            else:
                # If file does not exist, create data with the new CM
                data = self.cm_cache
            # Save data
            np.savez(file_path, **data)






    def save_performance(self, outputs, pseudos, masks, img_paths, keyword=None, time_idx=None, args=None):
        to_divide = 0
        acc_st = 0
        miou_st = 0
        acc_sc = 0
        miou_sc = 0
        acc_tc = 0
        miou_tc = 0

        for output, pseudo, mask, img_path in zip(outputs, pseudos, masks, img_paths):
            # print(output.shape, pseudo.shape, mask.shape)
            output = output.argmax(axis=0).astype(np.uint8)

            pseudo = pseudo.astype(np.uint8)
            mask = mask.astype(np.uint8)

            CM_student_teacher = confusion_matrix(pseudo.flatten(), output.flatten(), labels=np.arange(self.n_classes),
                                                  sample_weight=self.valid_zone.flatten() if self.valid_zone is not None else None)
            CM_student_carla = confusion_matrix(mask.flatten(), output.flatten(), labels=np.arange(self.n_classes),
                                                sample_weight=self.valid_zone.flatten() if self.valid_zone is not None else None)

            CM_teacher_carla = confusion_matrix(mask.flatten(), pseudo.flatten(), labels=np.arange(self.n_classes),
                                                sample_weight=self.valid_zone.flatten() if self.valid_zone is not None else None)

            self.update(CM_student_teacher, CM_student_carla, weight=1.)

            dirs = img_path.split('/')
            index = dirs.index('images')
            # save_folder = '/'.join(dirs[:index-3])

            key_prefix = f'{dirs[index-1]}-{dirs[index+1]}-{dirs[index + 2][:-4]}'


            self.cm_cache[f'{key_prefix}-{time_idx}-CM_student_teacher'] = CM_student_teacher
            self.cm_cache[f'{key_prefix}-{time_idx}-CM_student_carla'] = CM_student_carla
            self.cm_cache[f'{key_prefix}-{time_idx}-CM_teacher_carla'] = CM_teacher_carla

            acc_st += self.compute_accuracy(CM_student_teacher)
            miou_st += self.compute_miou(CM_student_teacher)

            acc_sc += self.compute_accuracy(CM_student_carla)
            miou_sc += self.compute_miou(CM_student_carla)

            acc_tc += self.compute_accuracy(CM_teacher_carla)
            miou_tc += self.compute_miou(CM_teacher_carla)

            to_divide += 1

        return acc_st, miou_st, acc_sc, miou_sc, acc_tc, miou_tc, to_divide

    def compute_performance(self, outputs, pseudos, masks, img_paths):
        to_divide = 0
        acc_st = 0
        miou_st = 0
        acc_sc = 0
        miou_sc = 0
        acc_tc = 0
        miou_tc = 0


        labels = np.arange(0, self.n_classes)

        for output, pseudo, mask, img_path in zip(outputs, pseudos, masks, img_paths):
            # print(output.shape, pseudo.shape, mask.shape)
            output = output.argmax(axis=0).astype(np.uint8)
            #output -= 1  # correct annotation shift (last channel is background)
            pseudo = pseudo.astype(np.uint8)
            mask = mask.astype(np.uint8)

            CM_student_teacher = confusion_matrix(pseudo.flatten(), output.flatten(), labels=labels,
                                                  sample_weight=self.valid_zone.flatten() if self.valid_zone is not None else None)
            CM_student_carla = confusion_matrix(mask.flatten(), output.flatten(), labels=labels,
                                                sample_weight=self.valid_zone.flatten() if self.valid_zone is not None else None)
            CM_teacher_carla = confusion_matrix(mask.flatten(), pseudo.flatten(), labels=labels,
                                                sample_weight=self.valid_zone.flatten() if self.valid_zone is not None else None)

            self.update(CM_student_teacher, CM_student_carla, weight=1.)

            acc_st += self.compute_accuracy(CM_student_teacher)
            miou_st += self.compute_miou(CM_student_teacher)

            acc_sc += self.compute_accuracy(CM_student_carla)
            miou_sc += self.compute_miou(CM_student_carla)

            acc_tc += self.compute_accuracy(CM_teacher_carla)
            miou_tc += self.compute_miou(CM_teacher_carla)

            to_divide += 1

        return acc_st, miou_st, acc_sc, miou_sc, acc_tc, miou_tc, to_divide

    def compute_accumulated_miou(self):
        miou_st = self.compute_miou(self.confusion_matrix_st)
        miou_sc = self.compute_miou(self.confusion_matrix_sc)
        return miou_st, miou_sc
    def evaluate(self, args, model: torch.nn.Module, dataloader: DataLoader, keyword='oups', time_idx=None,
                 save=True):
        self.keyword = keyword
        if dataloader is None:
            return

        model.eval()
        device = next(model.parameters()).device

        to_divide = 0
        acc_st = 0
        miou_st = 0
        acc_sc = 0
        miou_sc = 0
        acc_tc = 0
        miou_tc = 0

        with torch.no_grad():
            for i, (images, pseudos, masks, _, img_paths) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing batches"):
                images = images.to(device)
                # print(images.shape, pseudos.shape, masks.shape, img_paths)
                outputs = model.forward(images)
                if save:
                    _acc_st, _miou_st, _acc_sc, _miou_sc, _acc_tc, _miou_tc, _to_divide = self.save_performance(
                        outputs.cpu().detach().numpy(), pseudos.cpu().numpy(), masks.cpu().numpy(), img_paths,
                        keyword=keyword, time_idx=time_idx, args=args)

                else:
                    _acc_st, _miou_st, _acc_sc, _miou_sc, _acc_tc, _miou_tc, _to_divide = self.compute_performance(
                        outputs.cpu().detach().numpy(), pseudos.cpu().numpy(), masks.cpu().numpy(), img_paths)

                acc_st += _acc_st
                miou_st += _miou_st
                acc_sc += _acc_sc
                miou_sc += _miou_sc
                acc_tc += _acc_tc
                miou_tc += _miou_tc
                to_divide += _to_divide

        acc_st /= to_divide
        miou_st /= to_divide
        acc_sc /= to_divide
        miou_sc /= to_divide
        acc_tc /= to_divide
        miou_tc /= to_divide



        """
        wandb.log(
            {f'accuracy student - teacher /env : {args.env_name_key} - {keyword}': acc_st,
             f'mIoU student - teacher /env : {args.env_name_key} - {keyword}': miou_st,
             f'step student - teacher /env : {args.env_name_key} - {keyword}': self.global_counter},
        )
        wandb.log(
            {f'accuracy student - carla /env : {args.env_name_key} - {keyword}': acc_sc,
             f'mIoU student - carla /env : {args.env_name_key} - {keyword}': miou_sc,
             f'step student - carla /env : {args.env_name_key} - {keyword}': self.global_counter},
        )
        wandb.log(
            {f'accuracy teacher - carla /env : {args.env_name_key} - {keyword}': acc_tc,
             f'mIoU teacher - carla /env : {args.env_name_key} - {keyword}': miou_tc,
             f'step teacher - carla /env : {args.env_name_key} - {keyword}': self.global_counter},
        )
        """
        self.global_counter += 1

    def validation(self, model: torch.nn.Module, dataloader: DataLoader, keyword='oups', env_name='oups', save=False,
                   step=None):
        if dataloader is None:
            return None

        model.eval()
        device = next(model.parameters()).device

        to_divide = 0
        acc_st = 0
        miou_st = 0
        acc_sc = 0
        miou_sc = 0
        acc_tc = 0
        miou_tc = 0

        with torch.no_grad():
            torch.cuda.empty_cache()
            for i, (images, pseudos, masks, _, img_paths) in enumerate(dataloader):
                with torch.cuda.amp.autocast():
                    images = images.to(device)
                    # print(images.shape, pseudos.shape, masks.shape, img_paths)
                    outputs = model.forward(images)
                #images = images.to("cpu")
                outputs = outputs.to("cpu")
                if save:
                    _acc_st, _miou_st, _acc_sc, _miou_sc, _acc_tc, _miou_tc, _to_divide = self.save_performance(
                        outputs.cpu().detach().numpy(), pseudos.cpu().numpy(), masks.cpu().numpy(), img_paths,
                        keyword=keyword, env_name=env_name)
                else:
                    _acc_st, _miou_st, _acc_sc, _miou_sc, _acc_tc, _miou_tc, _to_divide = self.compute_performance(
                        outputs.cpu().detach().numpy(), pseudos.cpu().numpy(), masks.cpu().numpy(), img_paths,
                        )

                acc_st += _acc_st
                miou_st += _miou_st
                acc_sc += _acc_sc
                miou_sc += _miou_sc
                acc_tc += _acc_tc
                miou_tc += _miou_tc
                to_divide += _to_divide

        acc_st /= to_divide
        miou_st /= to_divide
        acc_sc /= to_divide
        miou_sc /= to_divide
        acc_tc /= to_divide
        miou_tc /= to_divide
        """
        wandb.log(
            {f'accuracy student - teacher /env : {env_name} - {keyword}': acc_st,
             f'mIoU student - teacher /env : {env_name} - {keyword}': miou_st
             },
            step=step
        )
        wandb.log(
            {f'accuracy student - carla /env : {env_name} - {keyword}': acc_sc,
             f'mIoU student - carla /env : {env_name} - {keyword}': miou_sc
             },
            step=step
        )
        wandb.log(
            {f'accuracy teacher - carla /env : {env_name} - {keyword}': acc_tc,
             f'mIoU teacher - carla /env : {env_name} - {keyword}': miou_tc
             },
            step=step
        )
        """
        return miou_st, miou_sc


def write_segmentation_performance(path, evaluator_now, evaluator_before, evaluator_after, init=False):
    with open(path, "a") as f:
        if init:
            f.write("accuracy;miou;accuracy_bwt;miou_fwt;accuracy_fwt;miou_fwt")
            for i in range(evaluator_now.n_classes):
                f.write(f';iou_{i}')
        else:
            f.write(
                str(evaluator_now.compute_accuracy()) + ";" +
                str(evaluator_now.compute_miou()) + ";" +
                str(evaluator_before.compute_accuracy()) + ";" +
                str(evaluator_before.compute_miou()) + ";" +
                str(evaluator_after.compute_accuracy()) + ";" +
                str(evaluator_after.compute_miou())
            )

            for i in range(evaluator_now.n_classes):
                iou = evaluator_now.compute_iou_c(i)
                if math.isnan(iou):
                    f.write(";")
                else:
                    f.write(";" + str(iou))
        f.write("\n")


