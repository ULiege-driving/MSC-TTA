import zipfile
import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import copy


def LoadJsonFromZip(zippedFile, JsonPath):
    with zipfile.ZipFile(zippedFile, "r") as z:
        # print(zipfile.ZipFile.namelist(z))
        with z.open(JsonPath) as f:
            data = f.read()
            d = json.loads(data.decode("utf-8"))
    return d


def LoadNpzFromZip(zippedFile, NpzPath):
    with zipfile.ZipFile(zippedFile, "r") as z:
        # print(zipfile.ZipFile.namelist(z))
        with z.open(NpzPath) as f:
            data = np.load(f)['arr_0']
        # print(data)
    return data


def LoadImageFromZip(zippedFile, ImagePath):
    with zipfile.ZipFile(zippedFile, "r") as z:
        with z.open(ImagePath) as f:
            img = Image.open(f)
            # img.show()
            img_copy = copy.copy(img)
            img_copy = np.array(img_copy, np.uint8)
    return img_copy


def WriteImageInZip(ImagePath, zippedFile, DestinationPath):
    with zipfile.ZipFile(zippedFile, 'a') as z:
        z.write(ImagePath, DestinationPath)


# Mimic os.walk() function for zipfiles
def zipwalk(zippedFile):
    # Initialize database
    dlistdb = {}

    # Walk through zip file information list
    with zipfile.ZipFile(zippedFile, "r") as z:
        for info in z.infolist():
            if info.is_dir():
                zpath = os.path.dirname(os.path.dirname(info.filename).rstrip('/'))
                zfile = os.path.basename(os.path.dirname(info.filename).rstrip('/'))
                if zpath in dlistdb:
                    dlistdb[zpath][0].append(zfile)
                else:
                    dlistdb[zpath] = [[zfile], []]
            else:
                zpath = os.path.dirname(info.filename)
                zfile = os.path.basename(info.filename)
                if zpath in dlistdb:
                    dlistdb[zpath][1].append(zfile)
                else:
                    dlistdb[zpath] = [[], [zfile]]

    # Convert to os.walk() output format
    dlist = []
    for key in dlistdb.keys():
        dlist.append((key, dlistdb[key][0], dlistdb[key][1]))

    return iter(dlist)


if __name__ == "__main__":

    # Examples of argument to give
    zips_path = ''
    seq_name = '2023-08-23_17-10-09'
    subfolder = '001'
    file_name = '000001'
    image_name = file_name + '.png'
    npz_name = file_name + '.npz'

    if zipfile.is_zipfile(os.path.join(zips_path, seq_name + '.zip')):
        metadata = LoadJsonFromZip(os.path.join(zips_path, seq_name + '.zip'),
                                   os.path.join(seq_name, seq_name + '.json'))
        gnss = LoadJsonFromZip(os.path.join(zips_path, seq_name + '.zip'), os.path.join(seq_name, 'gnss.json'))
        weather = LoadJsonFromZip(os.path.join(zips_path, seq_name + '.zip'), os.path.join(seq_name, 'weather.json'))

        RGB_image = LoadImageFromZip(os.path.join(zips_path, seq_name + '.zip'),
                                     os.path.join(seq_name, 'images', subfolder, image_name))
        semantic_mask = LoadImageFromZip(os.path.join(zips_path, seq_name + '.zip'),
                                         os.path.join(seq_name, 'semantic_masks', subfolder, image_name))
        npz_file = LoadNpzFromZip(os.path.join(zips_path, seq_name + '.zip'),
                                  os.path.join(seq_name, 'pseudo', subfolder, npz_name))
# print(metadata)
# plt.imshow(RGB_image)
# plt.show()
