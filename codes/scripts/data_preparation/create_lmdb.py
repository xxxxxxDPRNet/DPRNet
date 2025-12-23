from lmdb_util import make_lmdb_from_imgs
from os import path as osp
import os

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)
def create_lmdb():

    folder_path = '/data/01_data/TM/hdrplus_4k/train/input'
    lmdb_path = '/data/01_data/TM/hdrplus_4k/hdrplus_train_source.lmdb'
    img_path_list, keys = prepare_keys_tif(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/data/01_data/TM/hdrplus_4k/train/gt'
    lmdb_path = '/data/01_data/TM/hdrplus_4k/hdrplus_train_target.lmdb'
    img_path_list, keys = prepare_keys_jpg(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/data/01_data/TM/hdrplus_4k/test/input'
    lmdb_path = '/data/01_data/TM/hdrplus_4k/hdrplus_test_source.lmdb'
    img_path_list, keys = prepare_keys_tif(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/data/01_data/TM/hdrplus_4k/test/gt'
    lmdb_path = '/data/01_data/TM/hdrplus_4k/hdrplus_test_target.lmdb'
    img_path_list, keys = prepare_keys_jpg(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def prepare_keys_tif(folder_path):

    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix='tif', recursive=False)))
    keys = [img_path.split('.tif')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys

def prepare_keys_jpg(folder_path):

    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix='jpg', recursive=False)))
    keys = [img_path.split('.jpg')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys

if __name__ == '__main__':

    create_lmdb()

