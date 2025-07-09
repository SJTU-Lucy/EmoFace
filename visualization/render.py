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


class CtrAttrsManipulator:
    @staticmethod
    def set_neutral_face(attr_names):
        for name in attr_names:
            attr = name.replace("-", ".")
            try:
                cmds.setAttr(attr, 0)
            except ValueError:
                logging.info("Attr %s  set value error because it is locked " % attr)


class CTRDataLoader:
    def __init__(self, config_names_file_path):
        self._config_names = []
        if config_names_file_path:
            self._config_names = IOHelper.get_param_names(config_names_file_path)

    def read_anim_data(self, data_file_path):
        records = IOHelper.read_anim_txt(data_file_path, self._config_names)
        anim_data = defaultdict(dict)
        for idx, name in enumerate(self._config_names):
            ctr_name, attr_name = name.split("-")
            anim_data[ctr_name][attr_name] = [record[idx] for record in records]
        return anim_data

    def load_anim_data(self, anim_data):
        total_frames = -1
        for obj, attrs in anim_data.items():
            if obj == "timeinfo":
                continue

            for attr, values in attrs.items():
                total_frames = len(values)
                for i in range(total_frames):
                    cmds.setKeyframe(obj, at=attr, time=i, v=values[i])
        logging.info("Animation with the Controllers has been reenacted!, Animation Size is %d" % total_frames)
        return total_frames

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

    @staticmethod
    def load_audio(audio_file_path):
        audio_node = cmds.sound(file=audio_file_path)
        cmds.timeControl('timeControl1', edit=True, sound=audio_node)
        return audio_node

    @staticmethod
    def clear_audio(audio_node):
        if audio_node:
            cmds.delete(audio_node)

    def set_neutral_face(self):
        CtrAttrsManipulator.set_neutral_face(self._config_names)


class VideoExporter:
    def __init__(self):
        cmds.setAttr("frontShape.orthographicWidth", 30)
        cmds.setAttr("front.translateX", 0)
        cmds.setAttr("front.translateY", -2000)
        cmds.setAttr("front.translateZ", 149)

    def run(self, start_frame, end_frame, out_dir, out_filename):
        logging.info("[INFO]---export video start---")
        IOHelper.make_dirs(out_dir)
        file_path = os.path.join(out_dir, out_filename)
        # TODO more detailed configuration should be set here
        cmds.playblast(p=100, format='qt', st=start_frame, et=end_frame, f=file_path, widthHeight=[800, 800],
                       compression="H.264", fo=True, v=False, cc=True, offScreen=True)

        logging.info("---export video over---")


class AnimProcessor:
    def __init__(self):
        self._anim_fps = 60
        self._anim_type = "METAHUMAN-CTR"
        self._input_path_pattern = "YOUR Controller Rig File Path"
        self._config_file_path = "metahuman_attr_names.txt"
        self._mesh_type, self._file_type = self._anim_type.rsplit("-", 1)
        self._out_dir = "Render Output Path"
        self._export_video = true
        cmds.currentUnit(time="%dfps" % self._anim_fps)
        self._data_loader = CTRDataLoader(self._config_file_path)
        self._video_exporter = VideoExporter()

    def run(self):
        self.export()

    def export(self):
        top_dir, files = IOHelper.parse_path_pattern(input_path_pattern=self._input_path_pattern)
        counter = 0
        for file_path in files:
            out_video_dir = os.path.dirname(file_path.replace(top_dir, os.path.join(self._out_dir, "video")))
            out_filename = os.path.basename(file_path).split(".")[0] + ".mov"
            self._data_loader.clear_anim_data()
            self._data_loader.set_neutral_face()
            anim_data = self._data_loader.read_anim_data(file_path)
            total_frames = self._data_loader.load_anim_data(anim_data)
            counter += 1

            if self._video_exporter:
                self._video_exporter.run(0, total_frames - 1, out_video_dir, out_filename)

            logging.info("Process " + file_path + " success!")
            
            if counter == 100:
                break


if __name__ == '__main__':
    logging.basicConfig(format="%(levelname)s: %(message)s")
    AnimProcessor().export()
    logging.info("Process all anim data success!")

