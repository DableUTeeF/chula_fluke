
main_path = '/media/palm/BiggerData/mmdetection/configs'

class LinearSchedules:
    @staticmethod
    def set_1x(cfg):
        cfg.lr_config.step = [8, 11]
        cfg.runner.max_epochs = 12
        return cfg

    @staticmethod
    def set_2x(cfg):
        cfg.lr_config.step = [16, 22]
        cfg.runner.max_epochs = 24
        return cfg

    @staticmethod
    def set_3x(cfg):
        cfg.lr_config.step = [27, 33]
        cfg.runner.max_epochs = 36
        return cfg

    @staticmethod
    def set_20e(cfg):
        cfg.lr_config.step = [16, 19]
        cfg.runner.max_epochs = 20
        return cfg
