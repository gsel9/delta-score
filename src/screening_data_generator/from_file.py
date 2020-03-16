from typing import Union, List
from math import floor

import numpy as np
import scipy.sparse as sp
import h5py



def matlab_to_ndarray(path_to_file: str, keys: Union[str, List]) -> Union[List[np.ndarray], np.ndarray]:
    """Load contents from matlab file into numpy arrays.

    Args:
        path_to_file: Location of file on disk.
        keys: Names of target fields in the matlab file. Can be multiple.

    Returns:
        The contents of the matlab file as one array per key.
    """
    if isinstance(keys, str):
        return _load_matlab_file(path_to_file, keys).toarray()
        
    files = [_load_matlab_file(path_to_file, key).toarray() for key in keys]

    return files


def _load_matlab_file(path_to_file: str, keys: str):
    """Read screening data from .mat file.

    Args:
        path_to_file: Location of file on disk.
        keys: Names of target fields in the matlab file. Can be multiple.

    Returns:
        The contents of the matlab file as a sparse matrix.

    Note:
        * Expects '-v7.3' formatted matlab files.
    """

    db = h5py.File(path_to_file, 'r')
    ds = db[keys]

    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir   = np.asarray(ds['ir'])
            jc   = np.asarray(ds['jc'])
            out  = sp.csc_matrix((data, ir, jc)).astype(np.float32)

    # NOTE: Transpose in dense matrix case because of different row and column 
    # ordering between python and matlab.
    except AttributeError:
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out


if __name__ == "__main__":
    df = matlab_to_ndarray("/Users/sela/phd/data/real/data_matrix_3m.mat", "X")
    print(type(df))
    
