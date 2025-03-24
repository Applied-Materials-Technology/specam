import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm


def load_data_emissivity(excel_path, curve_key, **kwargs):
    excel_data = pd.read_excel(excel_path, None)

    emissivity_curves = {}
    for curve_name, data in excel_data.items():
        curve_i = int(curve_name.split()[1])

        temp = data["Temperature K"][0]
        assert np.all(data["Temperature K"] == temp)

        key = (curve_i, int(temp))
        emissivity_curves[key] = data[["Wavelength", "Emissivity"]].to_numpy()

    curve = emissivity_curves[curve_key]
    return interp1d(curve[:, 0], curve[:, 1], **kwargs)

def load_telops_data(data_path, num_filters, save_metadata=False):
    frames_idx = [[] for _ in range(num_filters)]
    frames = [[] for _ in range(num_filters)]
    if save_metadata:
        frames_meta = [[] for _ in range(num_filters)]

    for file_path in tqdm(data_path.iterdir()):
        if file_path.is_dir():
            continue
        
        frame_idx = int(file_path.stem.split('_')[-1])
        frame_meta = {}

        with file_path.open() as f:
            while True:
                line = f.readline().strip()
                if line == 'Data':
                    break
                line = line.split(' ', 1)
                frame_meta[line[0]] = line[1]

            frame = pd.read_table(
                f, sep=',', header=None, dtype=np.float32,
                na_values=['LDS', 'UDS', 'BIT', 'BEC', 'UCR', 'OCR', 'BAD', 'SAT']
            ).to_numpy()[:, :-1]
        
        filter_idx = int(frame_meta['Filtno'])
        frames_idx[filter_idx].append(frame_idx)
        frames[filter_idx].append(frame)
        if save_metadata:
            frames_meta[filter_idx].append(frame_meta)

    # sort by frame number
    for i in range(num_filters):
        sort_idx = np.argsort(frames_idx[i])
        frames_idx[i] = np.array(frames_idx[i])[sort_idx]
        frames[i] = np.array(frames[i])[sort_idx]
        if save_metadata:
            frames_meta[i] = [frames_meta[i][j] for j in sort_idx]

    rtn_val = (frames_idx, frames)
    if save_metadata:
        rtn_val += (frames_meta, )
    return rtn_val