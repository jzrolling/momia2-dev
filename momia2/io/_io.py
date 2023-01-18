import PIL as pl
import numpy as np
import nd2reader as nd2
import tifffile as tf
import os,glob,pickle as pk

__all__ = ['sort2folder',
           'softread_nd2file',
           'softread_tiffile',
           'softread_file',
           'get_slice_by_index',
           'get_slice_by_index',
           'load_softread']

def sort2folder(src_folder):
    """
    sort JOBS .nd2 files to well folders
    :param src_folder:
    :return:
    """
    files = sorted(glob.glob(src_folder + '*.nd2'))
    for f in files:
        header = f.split('/')[-1]
        well = header.split('_')[1][4:]
        subfolder = '{}{}/'.format(src_folder, well)
        if not os.path.isdir(subfolder):
            os.mkdir(subfolder)
        os.rename(f, '{}{}'.format(subfolder, header))
        

def ImageJinfo2dict(ImageJinfo):
    """
    Extract metadata from imageJ processsed .tif files
    :param ImageJinfo: imagej converted metadata
    :return: dictionary of metadata
    """
    if "\n" in ImageJinfo:
        ImageJinfo = ImageJinfo.split("\n")
    ImageJdict = {}
    for info in ImageJinfo:
        info = info.split(" = ")
        if len(info) == 2:
            key = info[0]
            val = info[1]
            if key.startswith(" "):
                key = key[1:]
            if val.startswith(" "):
                val = val[1:]
            ImageJdict[key] = val
    return ImageJdict


def softread_nd2file(nd2file, header=None):
    """
    open .nd2 image file
    :param nd2file: .nd2 file path
    :param header: in case you want to use a different file header
    :return: ND2reader obj with sorted metadata
    """
    import nd2reader as nd2
    # convert .nd2 file to standard OMEGA input
    # img order: v->t->c
    if not isinstance(nd2file, str):
        raise ValueError('Illegal input!')
    if not nd2file.endswith('.nd2'):
        raise ValueError('Input {} is not a .nd2 file.'.format(nd2file.split('/')[-1]))
    if header is None:
        header = nd2file.split('/')[-1].split('.')[0]
    sorted_data = {}
    img = nd2.ND2Reader(nd2file)
    sizes = img.sizes
    metadata = img.metadata
    shape = (sizes['x'], sizes['y'])
    channels = metadata['channels']
    n_channels = len(channels)
    n_fields = sizes['v'] if 'v' in sizes else 1
    n_time = sizes['t'] if 't' in sizes else 1
    n_zcoords = sizes['z'] if 'z' in sizes else 1
    sorted_data['format'] = 'nd2'
    sorted_data['header'] = header
    sorted_data['metadata'] = metadata
    sorted_data['pixel_microns'] = metadata['pixel_microns']
    sorted_data['ND2_object'] = img
    sorted_data['channels'] = channels
    sorted_data['n_fields'] = n_fields
    sorted_data['n_channels'] = n_channels
    sorted_data['n_timepoints'] = n_time
    sorted_data['n_zpositions'] = n_zcoords
    sorted_data['shape'] = shape
    return sorted_data


def softread_tiffile(tif, header=None, channels=None, pixel_microns=0.065):
    """
    open .tif file
    :param tif: .tif file path
    :param header: in case you want to use a different file header
    :param channels: microscopy channel names
    :param pixel_microns: micron length per pixel
    :return: tifffile obj obj with sorted metadata
    """
    import tifffile as tf
    # convert .tif file to standard OMEGA input
    # only supports single view, single/multi- channel tif data
    if not isinstance(tif, str):
        raise ValueError('Illegal input!')
    if not tif.endswith('.tif'):
        raise ValueError('Input {} is not a .tif file.'.format(tif.split('/')[-1]))
    if header is None:
        header = tif.split('/')[-1].split('.')[0]
    sorted_data = {}
    img = tf.TiffFile(tif)
    series = img.series[0]
    if len(series.shape) == 2:
        x, y = series.shape
        n_channels = 1
    elif len(series.shape) == 3:
        n_channels, x, y = series.shape
    else:
        raise ValueError('Hyperstack not supported!')
    if 'Info' in img.imagej_metadata:
        metadata = ImageJinfo2dict(img.imagej_metadata['Info'])
    else:
        metadata = {}
    shape = (x, y)
    if channels is not None:
        if len(channels) != n_channels:
            raise ValueError('{} channels found, {} channel names specified!'.format(n_channels,
                                                                                     len(channels)))
    else:
        channels = ['C{}'.format(i + 1) for i in range(n_channels)]

    sorted_data['format'] = 'tif'
    sorted_data['header'] = header
    sorted_data['metadata'] = metadata
    sorted_data['pixel_microns'] = float(metadata['dCalibration']) if 'dCalibration' in metadata else float(
        pixel_microns)
    sorted_data['TiffFile_object'] = img
    sorted_data['channels'] = channels
    sorted_data['n_fields'] = 1
    sorted_data['n_channels'] = 1
    sorted_data['n_timepoints'] = 1
    sorted_data['n_zpositions'] = 1
    sorted_data['shape'] = shape
    return sorted_data


def softread_file(img_file):
    """
    open image file, only support .nd2 and .tif for now
    :param img_file: image file path
    :return: image file handle with sorted metadata
    """
    if img_file.endswith('.nd2'):
        data_dict = softread_nd2file(img_file)
    elif img_file.endswith('.tif'):
        data_dict = softread_tiffile(img_file)
    else:
        raise ValueError('Data format .{} not supported!'.format(img_file.split('.')[-1]))
    return data_dict


def get_slice_by_index(data_dict, channel=0, position=0, time=0, zposition=0):
    sliced_img = np.array([])
    if data_dict['format'] == 'tif':
        sliced_img = data_dict['TiffFile_object'].pages[channel].asarray()
    elif data_dict['format'] == 'nd2':
        sliced_img = np.array(data_dict['ND2_object'].get_frame_2D(c=channel, v=position, t=time, z=zposition))
    return sliced_img


def load_softread(data_dict):
    header = data_dict['header']
    if data_dict['format'] == 'tif':
        reader_obj = data_dict.pop('TiffFile_object', None)
        sorted_data = data_dict.copy()
        img_stack_id = '{}_Time{}_Position{}_Z{}'.format(header, 0, 0, 0)
        sorted_data['volumes'] = {}
        sorted_data['volumes'][img_stack_id] = {c: reader_obj.pages[i].asarray() \
                                                for i, c in enumerate(sorted_data['channels'])}

    elif data_dict['format'] == 'nd2':
        reader_obj = data_dict.pop('ND2_object', None)
        sorted_data = data_dict.copy()
        sorted_data['volumes'] = {}
        for t in range(sorted_data['n_timepoints']):
            for v in range(sorted_data['n_fields']):
                for z in range(sorted_data['n_zpositions']):
                    img_stack_id = '{}_Time{}_Position{}_Z{}'.format(header, t, v, z)
                    sorted_data['volumes'][img_stack_id] = {c: np.array(reader_obj.get_frame_2D(t=t, v=v, c=i, z=z))\
                                                            for i, c in enumerate(sorted_data['channels'])}
    return sorted_data