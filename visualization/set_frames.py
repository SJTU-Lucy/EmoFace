import json
import logging
import os
import traceback
import fnmatch
from collections import defaultdict

import maya.cmds as cmds
import pymel.core as pm


class IOHelper:
    @staticmethod
    def make_dirs(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def read_anim_txt(file_path, config_names):
        data = []
        with open(file_path, "r") as fp:
            for line in fp:
                record = [float(item) for item in line.split(",")]
                if len(record) != len(config_names):
                    raise ValueError("Data size(%d) is inconsistency with config names size(%d)!"
                                     % (len(record), len(config_names)))
                data.append(record)
        return data

    def get_param_names(file_path):
        return [line.strip() for line in open(file_path, "r")] if file_path else []

    @staticmethod
    def parse_path_pattern(input_path_pattern):
        end = min([input_path_pattern.find(c) if input_path_pattern.find(c) != -1 else len(
            input_path_pattern) for c in ("*", "[", "?")])
        top_dir = input_path_pattern[:end].rsplit("/", 1)[0]

        files = []
        for root, _, filenames in os.walk(top_dir):
            full_paths = [os.path.join(root, filename) for filename in filenames]
            files.extend(fnmatch.filter(full_paths, input_path_pattern))

        return top_dir, sorted(list(set(files)))


class CTRDataLoader:
    def __init__(self, config_names_file_path):
        self._config_names = []
        if config_names_file_path:
            self._config_names = IOHelper.get_param_names(config_names_file_path)
        self.total_frames = 0

    def read_anim_data(self, data_file_path):
        records = IOHelper.read_anim_txt(data_file_path, self._config_names)
        anim_data = defaultdict(dict)
        for idx, name in enumerate(self._config_names):
            ctr_name, attr_name = name.split("-")
            anim_data[ctr_name][attr_name] = [record[idx] for record in records]
        return anim_data

    def load_anim_data(self, anim_data):
        total_frames = -1
        # empty_frames = 10
        for obj, attrs in anim_data.items():
            for attr, values in attrs.items():
                total_frames = len(values)
                for i in range(total_frames):
                    cmds.setKeyframe(obj, at=attr, time=self.total_frames + i, v=values[i])
                # for i in range(empty_frames):
                #     cmds.setKeyframe(obj, at=attr, time=self.total_frames + i, v=0)
        self.total_frames += total_frames
        # self.total_frames += empty_frames
        logging.info("Animation with the Controllers has been reenacted!, Animation Size is %d" % total_frames)
        return self.total_frames

    @staticmethod
    def clear_anim_data():
        tl_keys = cmds.ls(type="animCurveTL")
        if tl_keys:
            cmds.cutKey(tl_keys, s=True)

        tu_keys = cmds.ls(type="animCurveTU")
        if tu_keys:
            cmds.cutKey(tu_keys, s=True)  # delete key command

        if tl_keys or tu_keys:
            logging.info("Clear CTR animation data success!")
        else:
            logging.info("No CTR animation data here, clear success!")


class AnimProcessor:
    def __init__(self):
        self._anim_fps = 60
        self._input_path_pattern = "YOUR Controller Rig File Path"
        self._config_file_path = "metahuman_attr_names.txt"
        self._data_loader = CTRDataLoader(self._config_file_path)
        cmds.currentUnit(time="%dfps" % self._anim_fps)

    def export(self):
        self._data_loader.clear_anim_data()
        file_path = self._input_path_pattern
        anim_data = self._data_loader.read_anim_data(file_path)
        total_frames = self._data_loader.load_anim_data(anim_data)
        logging.info("Process " + file_path + " success!" + " Total Frame = " + str(total_frames))

if __name__ == '__main__':
    logging.basicConfig(format="%(levelname)s: %(message)s")
    AnimProcessor().export()
    logging.info("Process all anim data success!")

