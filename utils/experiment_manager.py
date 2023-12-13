import argparse
from fvcore.common.config import CfgNode as _CfgNode
from pathlib import Path


class CfgNode(_CfgNode):
    """
    The same as `fvcore.common.config.CfgNode`, but different in:

    1. Use unsafe yaml loading by default.
      Note that this may lead to arbitrary code execution: you must not
      load a config file from untrusted sources before manually inspecting
      the content of the file.
    2. Support config versioning.
      When attempting to merge an old config, it will convert the old config automatically.

    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Always allow merging new configs
        self.__dict__[CfgNode.NEW_ALLOWED] = True
        super(CfgNode, self).__init__(init_dict, key_list, True)

    # Note that the default value of allow_unsafe is changed to True
    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = True) -> None:
        loaded_cfg = _CfgNode.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)

        # defaults.py needs to import CfgNode
        self.merge_from_other_cfg(loaded_cfg)


def new_config():
    '''
    Creates a new config based on the default config file
    :return:
    '''

    C = CfgNode()

    C.CONFIG_DIR = 'config/'

    C.PATHS = CfgNode()
    C.TRAINER = CfgNode()
    C.MODEL = CfgNode()
    C.DATALOADER = CfgNode()
    C.DATASET = CfgNode()
    C.AUGMENTATIONS = CfgNode()

    return C.clone()


def setup_cfg(args, config_name: str = None):
    cfg = new_config()
    config_name = args.config_file if config_name is None else config_name
    cfg.merge_from_file(f'configs/{config_name}.yaml')
    cfg.merge_from_list(args.opts)
    cfg.NAME = config_name
    cfg.RUN_NUM = args.run
    cfg.PATHS.ROOT = str(Path.cwd())
    assert (Path(args.output_dir).exists())
    cfg.PATHS.OUTPUT = args.output_dir
    assert (Path(args.dataset_dir).exists())
    cfg.PATHS.DATASET = args.dataset_dir
    return cfg


