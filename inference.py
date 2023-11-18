import torch
import numpy as np
from tqdm import tqdm
from utils import datasets, experiment_manager, networks, parsers, helpers
from pathlib import Path

FONTSIZE = 16


def inference_quantitative(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg, device)
    net.eval()

    ds = datasets.GridPopulationDataset(cfg, run_type, no_augmentations=True)

    data = []
    for index in tqdm(range(len(ds))):
        item = ds.__getitem__(index)
        x = item['x'].to(device)

        with torch.no_grad():
            y_hat = net(x.unsqueeze(0))

        y_hat = y_hat.squeeze().cpu().item()
        y = item['y'].item()

        metadata = ds.samples[index]

        cell_data = {
            'hrsl_pop': y,
            'pop_m1': metadata['pop_m1'],
            'pop_m2': metadata['pop_m2'],
            'pred_pop': y_hat,
        }
        data.append(cell_data)

    out_file = Path(cfg.PATHS.OUTPUT) / 'inference' / f'{cfg.NAME}_quantitative.json'
    helpers.write_json(out_file, data)


def inference_qualitative(cfg: experiment_manager.CfgNode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg, device)
    net.eval()

    arr = np.empty((90, 212), dtype=np.float32)

    for run_type in ['train', 'val', 'test']:
        print(run_type)
        ds = datasets.GridPopulationDataset(cfg, run_type, no_augmentations=True)

        for index in tqdm(range(len(ds))):
            item = ds.__getitem__(index)
            x = item['x'].to(device)

            with torch.no_grad():
                y_hat = net(x.unsqueeze(0))

            y_hat = y_hat.squeeze().cpu().item()
            i = item['i']
            j = item['j']

            arr[i, j] = y_hat

    out_file = Path(cfg.PATHS.OUTPUT) / 'inference' / f'{cfg.NAME}_qualitative.tif'
    helpers.write_tif(out_file, arr, ds.get_geotransform(100), ds.crs)




if __name__ == '__main__':
    args = parsers.inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    # inference_quantitative(cfg)
    inference_qualitative(cfg)