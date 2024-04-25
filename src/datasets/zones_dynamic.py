"""
Zones class definition.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import zipfile

#from read_from_zip import LoadJsonFromZip


class Zones:
    """A class that allows you to get the zone in which a vehicle is located according to its x and y coordinates,
    by providing an image of the map in which each zone of the map is colored"""

    def __init__(self, dataset_path, dynamic_weather=False):
        """Loads the image of the map in which each zone is colored and generate an numpy array where each pixel is
        equal to the label of the zone (from the RGB color to the label)."""

        self.dataset_path = dataset_path

        self.ZONES = {
            0: {'name': 'forest', 'color': [85, 91, 25]},
            1: {'name': 'countryside', 'color': [111, 163, 27]},
            2: {'name': 'rural farmland', 'color': [237, 197, 0]},
            3: {'name': 'highway', 'color': [105, 110, 106]},
            4: {'name': 'low density residential', 'color': [13, 213, 148]},
            5: {'name': 'community buildings', 'color': [0, 147, 230]},
            6: {'name': 'high density residential', 'color': [213, 42, 0]},
        }

        self.dynamic_weather = dynamic_weather

        if dynamic_weather:
            self.WEATHERS = {
                0: {'name': 'clear', 'parameters': {'cloudiness': 5, 'precipitation': 0, 'precipitation_deposits': 0,
                                                    'wind_intensity': 10, 'fog_density': 2}},
                1: {'name': 'rainy', 'parameters': {'cloudiness': 50, 'precipitation': 90, 'precipitation_deposits': 80,
                                                    'wind_intensity': 100, 'fog_density': 7}},
                2: {'name': 'foggy', 'parameters': {'cloudiness': 50, 'precipitation': 0, 'precipitation_deposits': 0,
                                                    'wind_intensity': 10, 'fog_density': 70}},
            }

            self.PERIODS = {
                0: {'name': 'night', 'sun_altitude_angle': {'min': -90, 'max': 5}},
                1: {'name': 'day', 'sun_altitude_angle': {'min': 5, 'max': 90}}
            }

        # We load the image of the map in which each zone is colored
        try:
            parent_folder = os.path.dirname(dataset_path)
            img = cv2.imread(os.path.join(parent_folder, 'Town12.png'))
            red = img[:, :, 2]  # cv2 uses BGR channel order
            green = img[:, :, 1]
            blue = img[:, :, 0]
        except:
            img = cv2.imread(os.path.join(dataset_path, 'Town12.png'))
            red = img[:, :, 2]  # cv2 uses BGR channel order
            green = img[:, :, 1]
            blue = img[:, :, 0]

        # We create a numpy array with the same height and width as the image
        height, width = img.shape[0], img.shape[1]
        zones_img = np.zeros([height, width])

        # We generate the numpy array where each pixel is equal to the label of the zone
        for key, value in self.ZONES.items():
            mask_red, mask_green, mask_blue = (red == value['color'][0]), (green == value['color'][1]), (
                        blue == value['color'][2])
            mask = (mask_red & mask_green & mask_blue)
            zones_img[mask] = key
        self.zones_img = zones_img

        # We calculate the homography matrix
        pts_src = np.array([[-5678.9, -2900.3], [4809.0, -2879.6], [-5678.5, 7597.0],
                            [4804.2, 7602.6]])  # Four corners of the map in CARLA
        pts_dst = np.array(
            [[0, 0], [7127, 0], [0, 7127], [7127, 7127]])  # Four corners of the image of the map containning the zones
        self._h_matrix, status = cv2.findHomography(pts_src, pts_dst)

        # We load the gnns.json file for each sequence
        seq_paths = []
        seq_paths.extend([f.path for f in os.scandir(dataset_path) if f.is_dir()])
        # seq_paths.extend([f.path for f in os.scandir(dataset_path) if zipfile.is_zipfile(f.path)])

        # seq_paths = [ f.path for f in os.scandir(dataset_path) if f.is_dir() ]
        # seq_paths = [ f for f in os.listdir('./') if zipfile.is_zipfile(f) ]
        self.gnss = {}
        for seq_path in seq_paths:
            if zipfile.is_zipfile(seq_path):
                seq_name = os.path.basename(seq_path)[:-4]
                self.gnss[seq_name] = LoadJsonFromZip(seq_path, os.path.join(seq_name, 'gnss.json'))
            else:
                seq_name = os.path.basename(seq_path)
                with open(os.path.join(self.dataset_path, seq_name, 'gnss.json'), "r") as fp:
                    self.gnss[seq_name] = json.load(fp)

        if dynamic_weather:
            # We load the weather.json file for each sequence
            self.weather = {}
            for seq_path in seq_paths:
                if zipfile.is_zipfile(seq_path):
                    seq_name = os.path.basename(seq_path)[:-4]
                    self.weather[seq_name] = LoadJsonFromZip(seq_path, os.path.join(seq_name, 'weather.json'))
                else:
                    seq_name = os.path.basename(seq_path)
                    with open(os.path.join(self.dataset_path, seq_name, 'weather.json'), "r") as fp:
                        self.weather[seq_name] = json.load(fp)

    def get_zone(self, seq_name, frame_id):
        """Given a frame in a sequence, returns the zone in which the vehicle is.
        If dynamic_weather is false, the returned value is a string containg 1 digit corresponding the the zone label.
        If dynamic_weather is true, the retruned value is a string containing 3 digits: the left most digit corresponds to the zone label, the middle digit corresponds to te weather label, and the right most digit corresponds to the period label. """

        # Gets the x and y coordinates of the vehicle given the seq_name and the frame_id
        x = self.gnss[seq_name][str(frame_id)]["x"]
        y = self.gnss[seq_name][str(frame_id)]["y"]

        # Converts the x and y coordinates of the vehicle in CARLA to a pixel position, w and h, in the image of the map using homography propreties
        w, h = self._convert_coordinates(x, y)

        # Gets the pixel value of the image of the map at the coordinates (w,h)
        zone = self.zones_img[h, w]
        zone = str(int(zone))

        if self.dynamic_weather:
            weather_id = self._get_weather(seq_name, frame_id)
            zone = str(zone) + str(weather_id)

        return zone

    def _get_weather(self, seq_name, frame_id):
        """Given a frame in a sequence, returns the weather in which the vehicle is"""

        # Gets the weather parameter given the seq_name and the frame_id
        cloudiness = self.weather[seq_name][str(frame_id)]["cloudiness"]
        precipitation = self.weather[seq_name][str(frame_id)]["precipitation"]
        precipitation_deposits = self.weather[seq_name][str(frame_id)]["precipitation_deposits"]
        wind_intensity = self.weather[seq_name][str(frame_id)]["wind_intensity"]
        fog_density = self.weather[seq_name][str(frame_id)]["fog_density"]
        sun_altitude_angle = self.weather[seq_name][str(frame_id)]["sun_altitude_angle"]

        # Gets the weather label to which it corresponds
        min_L1_norm = float('inf')
        weather_id = None

        # Iterate through each predefined weather to find the closest match
        for key, weather in self.WEATHERS.items():
            predefined_weather = weather["parameters"]
            L1_norm = (abs(precipitation - predefined_weather["precipitation"]) +
                       abs(fog_density - predefined_weather["fog_density"]))

            # Check if this predefined weather is closer than what we have found so far
            if L1_norm < min_L1_norm:
                min_L1_norm = L1_norm
                weather_id = key

        # If no weather_id was found that means all were above the minimum or no WEATHERS are defined
        if weather_id is None:
            weather_id = 3  # or some default value or behavior

        period_id = 0
        if sun_altitude_angle >= self.PERIODS[1]["sun_altitude_angle"]["min"]:
            period_id = 1

        weather_id = str(weather_id) + str(period_id)

        return weather_id

    def _convert_coordinates(self, x, y):
        """Converts the x and y coordinates of the vehicle in CARLA to a pixel position, w and h, in the image of the map using homography propreties"""
        p_src = np.array([x, y, 1]).reshape((3, 1))
        p_dst = np.matmul(self._h_matrix, p_src)
        w = p_dst[0] / p_dst[2]
        h = p_dst[1] / p_dst[2]
        w = int(round(w[0]))
        h = int(round(h[0]))
        return w, h

    # !!! Function to remove before publication !!!
    def check_map(self):
        """Checks the validity of the image of the map (i.e., verifies that each pixel corresponds to the label of a zone."""
        nb_pixels = np.zeros(len(self.ZONES.keys()))
        classes = []
        for key, value in self.ZONES.items():
            nb_pixels[key] = (self.zones_img == key).sum()
            classes.append(value["name"])

        # Plot the histogram
        plt.bar(classes, nb_pixels)
        plt.xticks(classes, rotation=45, rotation_mode='anchor', ha='right')
        plt.xlabel('Classes')
        plt.ylabel('Number of pixels')
        plt.title(Zones)
        plt.show()
        # plt.savefig('Nb_pixels_per_zone.png', bbox_inches = 'tight')
        # plt.close()
        return nb_pixels.sum() == (self.zones_img.shape[0] * self.zones_img.shape[1]), nb_pixels