import numpy as np
import array
import struct
import collections

dm_header = collections.namedtuple('dmheader', ('version', 'root_length', 'little_endian'))
dm_tag = collections.namedtuple('dm_tag', ('type', 'name', 'offset', 'data_offset', 'tag_data_type', 'byte_length'))
dm_dir = collections.namedtuple('dm_tag', ('type', 'name', 'offset', 'data_offset', 'num_tags', 'byte_length'))
root_dir = collections.namedtuple('root_dir', ('name', 'offset', 'is_sorted', 'is_closed', 'num_tags'))
dmDataType = collections.namedtuple('dmDataTypes', ('num_bytes', 'signed', 'type_format'))
dir_obj = collections.namedtuple('dir_obj',
                                 ('name', 'tag', 'named_subdirs', 'unnamed_subdirs', 'named_tags', 'unnamed_tags'))

dmDataTypeDict = {2: dmDataType(2, True, 'h'),  # 2byte signed integer
                  3: dmDataType(4, True, 'i'),  # 4byte signed integer
                  4: dmDataType(2, False, 'H'),  # 2byte unsigned integer
                  5: dmDataType(4, False, 'I'),  # 4byte unsigned integer
                  6: dmDataType(4, False, 'f'),  # 4byte float
                  7: dmDataType(8, False, 'd'),  # 8byte float
                  8: dmDataType(1, False, '?'),
                  9: dmDataType(1, False, 'c'),
                  10: dmDataType(1, True, 'b'),
                  11: dmDataType(8, True, 'q'),
                  12: dmDataType(8, True, 'Q')
                  }


def _get_version(dmfile):
    dmfile.seek(0)
    # Read file version
    version = struct.unpack_from('>I', dmfile.read(4))[0]
    dmfile.seek(0)
    return version


def _get_endian(dmfile):
    header = read_header(dmfile)
    endian = header.little_endian
    dmfile.seek(0)

    if isinstance(endian, str):
        if endian == 'little':
            return '<'  # Little Endian
        else:
            return '>'  # Big Endian
    else:
        if endian == 1:
            return '<'  # Little Endian
        else:
            return '>'  # Big Endian


def read_header(dmfile):
    """
    Read file header of DM file. DM3 and DM4 file have three items: version, root_length, byte_order.
    : param
        dmfile: file handle to dm4 file
    """
    # sets the file's current position at the offset 0, i.e. the begining of the file
    dmfile.seek(0)
    # read dm version
    version = struct.unpack_from('>I', dmfile.read(4))[0]
    if version == 3:
        # read root length
        root_length = struct.unpack_from('>L', dmfile.read(4))[0]
    elif version == 4:
        # read root length
        root_length = struct.unpack_from('>Q', dmfile.read(8))[0]
    # read byte order
    byte_order = struct.unpack_from('>I', dmfile.read(4))[0]

    little_endian = byte_order == 1
    dmfile.seek(0)

    return dm_header(version, root_length, little_endian)


def read_root_dir(dmfile):
    """
    Read the root directory information from a DM3 or Dm4 file.
    File seek position is at the end dm file header.
    """
    version = _get_version(dmfile)
    endian = _get_endian(dmfile)

    # Get header size
    header_size = version * 4
    dmfile.seek(header_size)

    is_sorted = struct.unpack_from(endian + 'b', dmfile.read(1))[0]
    is_closed = struct.unpack_from(endian + 'b', dmfile.read(1))[0]
    if version == 4:
        num_tags = struct.unpack_from('>Q', dmfile.read(8))[0]
    elif version == 3:
        num_tags = struct.unpack_from('>L', dmfile.read(4))[0]

    return root_dir('root', header_size, is_sorted, is_closed, num_tags)


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#   Read tag and dir one by one
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def _read_tag_and_dir(dmfile, version, endian):
    '''
    Read dm_tag or dm_dir object from current position of dmfile,
    and point to the start of next tag or dir after reading.
    '''
    # Get to the offset position
    offset = dmfile.tell()
    # read the tag type
    tag_type = struct.unpack_from(endian + 'B', dmfile.read(1))[0]
    # tag
    if tag_type == 21:
        return _read_tag(dmfile, version, offset)
    # tag directories
    elif tag_type == 20:
        return _read_dir(dmfile, version, offset)
    elif tag_type == 0:
        return None


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#   Read tags and tag directories according to offset
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def read_tag_and_dir(dmfile, offset):
    '''
    Read dm_tag or dm_dir object according to offset.
    '''
    version = _get_version(dmfile)
    endian = _get_endian(dmfile)
    offset = dmfile.seek(offset)
    # read the tag type
    tag_type = struct.unpack_from(endian + 'B', dmfile.read(1))[0]
    # tag
    if tag_type == 21:
        return _read_tag(dmfile, version, offset)
    # tag directories
    elif tag_type == 20:
        return _read_dir(dmfile, version, offset)
    elif tag_type == 0:
        return None


def _read_tag(dmfile, version, offset):
    '''
    Read tag and return dm_tag object
    '''
    tag_name = _read_tag_name(dmfile)
    if version == 4:
        # New feature for DM4
        byte_length = struct.unpack_from('>Q', dmfile.read(8))[0]
        data_offset = dmfile.tell()
        _read_tag_garbage_str(dmfile)
        (num_info, info_array) = _read_tag_data_info(dmfile, version)
        tag_data_type = info_array[0]
        # Point to next tag or dir
        dmfile.seek(data_offset + byte_length)
    elif version == 3:
        data_offset = dmfile.tell()
        _read_tag_garbage_str(dmfile)
        (num_info, info_array) = _read_tag_data_info(dmfile, version)
        tag_data_type = info_array[0]
        byte_length = _calculate_byte((num_info, info_array)) + (dmfile.tell() - data_offset)
        dmfile.seek(data_offset + byte_length)
    if num_info == 1:
        tag_data_type_str = 'single'
    elif num_info == 3:
        tag_data_type_str = 'array'
    elif (num_info - 3) % 2 == 0:
        if tag_data_type == 15:
            tag_data_type_str = 'group'
        elif tag_data_type == 20:
            tag_data_type_str = 'group_array'
    else:
        tag_data_type_str = None

    return dm_tag('tag', tag_name, offset, data_offset, tag_data_type_str, byte_length)


def _calculate_byte(tag_data_info):
    '''
    Calculate number of bytes after tag data information
    '''
    num_info, info_array = tag_data_info
    tag_data_type = info_array[0]
    # tag containing single entry
    if num_info == 1:
        return dmDataTypeDict[tag_data_type].num_bytes
    # tag containing an array
    elif num_info == 3:
        assert (tag_data_type == 20)
        array_data_type = info_array[1]
        num_elements = info_array[2]
        return num_elements * dmDataTypeDict[array_data_type].num_bytes
    elif (num_info - 3) % 2 == 0:
        # tag contianing a group of data
        # info_array strucutre:
        # 1. tag_data_type --> info_array[0]
        # 2. length of groupname --> info_array[1]
        # 3. num_elements --> info_array[2]
        # 4. fields --> info_array[3:]
        if tag_data_type == 15:
            num_element = info_array[2]
            fields = info_array[3:]
            data_bytes = sum([dmDataTypeDict[fields[(i * 2) + 1]].num_bytes for i in np.arange(0, num_element)])
            return data_bytes
        # tag contianing a group of array
        elif tag_data_type == 20:
            num_values = info_array[3]
            fields = info_array[4:]
            data_bytes = sum([dmDataTypeDict[fields[(i * 2) + 1]].num_bytes for i in np.arange(0, num_values)])
            num_elements = info_array[-1]
            return num_elements * data_bytes


def _read_dir(dmfile, version, offset):
    '''
    Read dir and return dm_dir object
    '''
    tag_name = _read_tag_name(dmfile)
    if version == 4:
        byte_length = struct.unpack_from('>Q', dmfile.read(8))[0]
        dmfile.seek(dmfile.tell() + 2)
        num_tags = struct.unpack_from('>Q', dmfile.read(8))[0]
    elif version == 3:
        byte_length = None
        dmfile.seek(dmfile.tell() + 2)
        num_tags = struct.unpack_from('>L', dmfile.read(4))[0]
    data_offset = dmfile.tell()
    return dm_dir('dir', tag_name, offset, data_offset, num_tags, byte_length)


def read_all_tags(dmfile):
    '''
    Read all the tags and dirs
    '''
    version = _get_version(dmfile)
    endian = _get_endian(dmfile)
    read_root_dir(dmfile)
    tag = _read_tag_and_dir(dmfile, version, endian)
    tag_list = []
    while (tag != None):
        tag_list.append(tag)
        tag = _read_tag_and_dir(dmfile, version, endian)
    return tag_list


def read_directory(dmfile, directory_tag=None):
    '''
    Read all the tags and dirs and return a tree-like structure dir_obj.
    '''
    version = _get_version(dmfile)
    endian = _get_endian(dmfile)
    if directory_tag == None:
        directory_tag = read_root_dir(dmfile)
    else:
        dmfile.seek(directory_tag.offset)
        _read_tag_and_dir(dmfile, version, endian)
    dir_item = dir_obj(directory_tag.name, directory_tag, {}, [], {}, [])
    for i in np.arange(0, directory_tag.num_tags):
        tag = _read_tag_and_dir(dmfile, version, endian)
        if tag == None:
            break
        # directory
        if tag.type == 'dir':
            if tag.name is None:
                dir_item.unnamed_subdirs.append(read_directory(dmfile, tag))
            else:
                dir_item.named_subdirs[tag.name] = read_directory(dmfile, tag)
        # tag
        elif tag.type == 'tag':
            if tag.name is None:
                dir_item.unnamed_tags.append(tag)
            else:
                dir_item.named_tags[tag.name] = tag
    return dir_item


def _read_tag_name(dmfile):
    '''
    Read the name of tags or tag direcotries.
    '''
    tag_name_len = struct.unpack_from('>H', dmfile.read(2))[0]  # dm specifies this property as always big endian
    tag_name = None
    if tag_name_len > 0:
        data = dmfile.read(tag_name_len)
        try:
            tag_name = data.decode('utf-8', errors='ignore')
        except UnicodeDecodeError as e:
            tag_name = None
            pass
    return tag_name


def _read_tag_garbage_str(dmfile):
    """
    DM has four bytes of % symbols in the tag.
    """
    garbage_str = dmfile.read(4).decode('utf-8')
    assert (garbage_str == '%%%%')
    return garbage_str


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#   Read tag data from tag
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def _read_tag_data_info(dmfile, version):
    if version == 4:
        num_info = struct.unpack_from('>Q', dmfile.read(8))[0]
        format_str = '>' + num_info * 'q'  # Big endian signed long
        info_array = struct.unpack_from(format_str, dmfile.read(8 * num_info))
    elif version == 3:
        num_info = struct.unpack_from('>L', dmfile.read(4))[0]
        format_str = '>' + num_info * 'l'
        info_array = struct.unpack_from(format_str, dmfile.read(4 * num_info))
    return (num_info, info_array)


def _read_tag_data_single(dmfile, tag):
    assert (tag.tag_data_type == 'single')
    version = _get_version(dmfile)
    endian = _get_endian(dmfile)
    # Skip garbage string
    dmfile.seek(tag.data_offset + 4)
    (num_info, info_array) = _read_tag_data_info(dmfile, version)
    data_type = info_array[0]
    return _read_tag_data_value(dmfile, endian, data_type)


def _read_tag_data_group(dmfile, tag):
    assert (tag.tag_data_type == 'group')
    version = _get_version(dmfile)
    endian = _get_endian(dmfile)
    # Skip garbage string
    dmfile.seek(tag.data_offset + 4)
    (num_info, info_array) = _read_tag_data_info(dmfile, version)
    num_element = info_array[2]
    fields = info_array[3:]
    field_types_list = []
    for i in range(0, num_element):
        fieldname_type = fields[(i * 2) + 1]
        field_types_list.append(fieldname_type)

    fields_data = []
    for field_type in field_types_list:
        field_data = _read_tag_data_value(dmfile, endian, field_type)
        fields_data.append(field_data)
    return fields_data


def _read_tag_data_array(dmfile, tag):
    assert (tag.tag_data_type == 'array')
    version = _get_version(dmfile)
    endian = _get_endian(dmfile)
    # Skip garbage string
    dmfile.seek(tag.data_offset + 4)
    (num_info, info_array) = _read_tag_data_info(dmfile, version)
    array_type = info_array[1]
    array_length = info_array[2]

    data_type = dmDataTypeDict[array_type]

    data = array.array(data_type.type_format)
    data.fromfile(dmfile, array_length)
    return data


def _read_tag_data_value(dmfile, endian, field_type):
    data_type = dmDataTypeDict[field_type]
    format_str = endian + data_type.type_format
    byte_data = dmfile.read(data_type.num_bytes)
    return struct.unpack_from(format_str, byte_data)[0]


def read_tag_data(dmfile, tag):
    assert (tag.type == 'tag')
    if tag.tag_data_type == 'group':
        return _read_tag_data_group(dmfile, tag)
    elif tag.tag_data_type == 'array':
        return _read_tag_data_array(dmfile, tag)
    elif tag.tag_data_type == 'single':
        return _read_tag_data_single(dmfile, tag)
    elif tag.tag_data_type == 'group_array':
        print("Missing type " + str(tag_data_type))
        return None


class DMfile:
    def __init__(self, file_name):
        self.file_handle = open(file_name, 'rb')
        self.version = _get_version(self.file_handle)
        self.endian = _get_endian(self.file_handle)
        self.header = read_header(self.file_handle)
        self.root = read_root_dir(self.file_handle)

    @property
    def data(self):
        tags = read_directory(self.file_handle)
        image_data_tag = tags.named_subdirs['ImageList'].unnamed_subdirs[1].named_subdirs['ImageData']
        image_tag = image_data_tag.named_tags['Data']
        num_dim = len(image_data_tag.named_subdirs['Dimensions'].unnamed_tags)
        XDim = read_tag_data(self.file_handle, image_data_tag.named_subdirs['Dimensions'].unnamed_tags[0])
        YDim = read_tag_data(self.file_handle, image_data_tag.named_subdirs['Dimensions'].unnamed_tags[1])
        if num_dim == 3:
            ZDim = read_tag_data(self.file_handle, image_data_tag.named_subdirs['Dimensions'].unnamed_tags[2])
            shape = (ZDim, YDim, XDim)
        else:
            shape = (YDim, XDim)
        # np_array = np.array(read_tag_data(self.file_handle, image_tag), dtype=np.uint32)
        # return np.reshape(np_array, shape)
        np_array = np.array(read_tag_data(self.file_handle, image_tag).tolist())
        return np_array.reshape(shape)


# IO utility functions
from skimage.io import imread
from skimage.io import imsave
import pickle


def normalize(image, low=0, high=255):
    img_max = image.max()
    img_min = image.min()
    img_norm = (image - img_min) / (img_max - img_min) * (high - low) + low
    return img_norm


def load_image(file_name, normalized=False):
    file_extension = np.char.split(file_name, sep='.').tolist()[-1]
    file_extension = '.' + file_extension
    # file_extension = os.path.splitext(file_name)[1]
    if file_extension.lower() in ['.dm4', '.dm3']:
        img = DMfile(file_name).data
    else:
        img = imread(file_name)
    if normalized is True:
        img = normalize(img)
    return img


def save_image(data, file_name, cmap=None):
    if cmap == None:
        if data.dtype == np.float64:
            data = np.array(data, dtype=np.float32)
        imsave(file_name, data)
    else:
        # Normaize image into [0, 255]
        data = normalize(data, 0, 255)
        data = np.round(data).astype(np.int)
        R, G, B = cmap(np.arange(256))[:, 0:3].T
        r, g, b = R[data], G[data], B[data]
        rgb = np.stack([r, g, b], axis=2)
        imsave(file_name, rgb)


def load_pickle(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data


def save_pickle(data, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)


def load_dataset(file_name):
    ds = load_pickle(file_name)
    return (ds['img'], ds['pts'], ds['size'])