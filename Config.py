import yaml
import os

class Config:
    def __init__(self):
        with open('./settings.yaml') as file:
            try:
                config = yaml.load(file, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)
        self.config = config
        self.save_dir = None

    @property
    def fx(self):
        return self.config["Camera.fx"]

    @property
    def fy(self):
        return self.config["Camera.fy"]

    @property
    def cx(self):
        return self.config["Camera.cx"]

    @property
    def cy(self):
        return self.config["Camera.cy"]

    @property
    def bf(self):
        return self.config["Camera.bf"]

    @property
    def width(self):
        return self.config["Camera.width"]

    @property
    def height(self):
        return self.config["Camera.height"]

    @property
    def dir(self):
        if self.save_dir:
            return self.save_dir
        save_dir = self.config["Dir.save"]
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        dir_list = [int(o) for o in os.listdir(save_dir)
                    if os.path.isdir(os.path.join(save_dir, o)) and o.isdigit()]
        if len(dir_list) == 0:
            o = "0"
        else:
            o = 0
            while o in dir_list:
                o += 1
            o = str(o)
        self.save_dir = os.path.join(save_dir, o)
        os.mkdir(self.save_dir)
        return self.save_dir

    @property
    def bow(self):
        return self.config["Dir.bow"]

    @property
    def scene(self):
        return self.config["Dir.scene"]

    @property
    def n_features(self):
        return self.config["ORBextractor.nFeatures"]

    @property
    def scale_factor(self):
        return self.config["ORBextractor.scaleFactor"]

    @property
    def n_levels(self):
        return self.config["ORBextractor.nLevels"]

    @property
    def th_fast(self):
        return self.config["ORBextractor.thFAST"]
