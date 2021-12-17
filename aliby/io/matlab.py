"""Read and convert MATLAB files from Swain Lab platform.

TODO: Information that I need from lab members esp J and A
    * Lots of examples to try
    * Any ideas on what these Map objects are?

TODO: Update Swain Lab wiki

All credit to Matt Bauman for
the reverse engineering at https://nbviewer.jupyter.org/gist/mbauman/9121961
"""

import re
import struct
import sys
from collections import Iterable
from io import BytesIO

import h5py
import numpy as np
import pandas as pd
import scipy
from numpy.compat import asstr

# TODO only use this if scipy>=1.6 or so
from scipy.io import matlab
from scipy.io.matlab.mio5 import MatFile5Reader
from scipy.io.matlab.mio5_params import mat_struct

from aliby.io.utils import read_int, read_string, read_delim


def read_minimat_vars(rdr):
    rdr.initialize_read()
    mdict = {"__globals__": []}
    i = 0
    while not rdr.end_of_stream():
        hdr, next_position = rdr.read_var_header()
        name = asstr(hdr.name)
        if name == "":
            name = "var_%d" % i
            i += 1
        res = rdr.read_var_array(hdr, process=False)
        rdr.mat_stream.seek(next_position)
        mdict[name] = res
        if hdr.is_global:
            mdict["__globals__"].append(name)
    return mdict


def read_workspace_vars(fname):
    fp = open(fname, "rb")
    rdr = MatFile5Reader(fp, struct_as_record=True, squeeze_me=True)
    vars = rdr.get_variables()
    fws = vars["__function_workspace__"]
    ws_bs = BytesIO(fws.tostring())
    ws_bs.seek(2)
    rdr.mat_stream = ws_bs
    # Guess byte order.
    mi = rdr.mat_stream.read(2)
    rdr.byte_order = mi == b"IM" and "<" or ">"
    rdr.mat_stream.read(4)  # presumably byte padding
    mdict = read_minimat_vars(rdr)
    fp.close()
    return mdict


class matObject:
    """A python read-out of MATLAB objects
    The objects pulled out of the
    """

    def __init__(self, filepath):
        self.filepath = filepath  # For record
        self.classname = None
        self.object_name = None
        self.buffer = None
        self.version = None
        self.names = None
        self.segments = None
        self.heap = None
        self.attrs = dict()
        self._init_buffer()
        self._init_heap()
        self._read_header()
        self.parse_file()

    def __getitem__(self, item):
        return self.attrs[item]

    def keys(self):
        """Returns the names of the available properties"""
        return self.attrs.keys()

    def get(self, item, default=None):
        return self.attrs.get(item, default)

    def _init_buffer(self):
        fp = open(self.filepath, "rb")
        rdr = MatFile5Reader(fp, struct_as_record=True, squeeze_me=True)
        vars = rdr.get_variables()
        self.classname = vars["None"]["s2"][0].decode("utf-8")
        self.object_name = vars["None"]["s0"][0].decode("utf-8")
        fws = vars["__function_workspace__"]
        self.buffer = BytesIO(fws.tostring())
        fp.close()

    def _init_heap(self):
        super_data = read_workspace_vars(self.filepath)
        elem = super_data["var_0"][0, 0]
        if isinstance(elem, mat_struct):
            self.heap = elem.MCOS[0]["arr"]
        else:
            self.heap = elem["MCOS"][0]["arr"]

    def _read_header(self):
        self.buffer.seek(248)  # the start of the header
        version = read_int(self.buffer)
        n_str = read_int(self.buffer)

        offsets = read_int(self.buffer, n=6)

        # check that the next two are zeros
        reserved = read_int(self.buffer, n=2)
        assert all(
            [x == 0 for x in reserved]
        ), "Non-zero reserved header fields: {}".format(reserved)
        # check that we are at the right place
        assert self.buffer.tell() == 288, "String elemnts begin at 288"
        hdrs = []
        for i in range(n_str):
            hdrs.append(read_string(self.buffer))
        self.names = hdrs
        self.version = version
        # The offsets are actually STARTING FROM 248 as well
        self.segments = [x + 248 for x in offsets]  # list(offsets)
        return

    def parse_file(self):
        # Get class attributes from segment 1
        self.buffer.seek(self.segments[0])
        classes = self._parse_class_attributes(self.segments[1])
        # Get first set of properties from segment 2
        self.buffer.seek(self.segments[1])
        props1 = self._parse_properties(self.segments[2])
        # Get the property description from segment 3
        self.buffer.seek(self.segments[2])
        object_info = self._parse_prop_description(classes, self.segments[3])
        # Get more properties from segment 4
        self.buffer.seek(self.segments[3])
        props2 = self._parse_properties(self.segments[4])
        # Check that the last segment is empty
        self.buffer.seek(self.segments[4])
        seg5_length = (self.segments[5] - self.segments[4]) // 8
        read_delim(self.buffer, seg5_length)
        props = (props1, props2)
        self._to_attrs(object_info, props)

    def _to_attrs(self, object_info, props):
        """Re-organise the various classes and subclasses into a nested
        dictionary.
        :return:
        """
        for pkg_clss, indices, idx in object_info:
            pkg, clss = pkg_clss
            idx = max(indices)
            which = indices.index(idx)
            obj = flatten_obj(props[which][idx])
            subdict = self.attrs
            if pkg != "":
                subdict = self.attrs.setdefault(pkg, {})
            if clss in subdict:
                if isinstance(subdict[clss], list):
                    subdict[clss].append(obj)
                else:
                    subdict[clss] = [subdict[clss]]
                    subdict[clss].append(obj)
            else:
                subdict[clss] = obj

    def describe(self):
        describe(self.attrs)

    def _parse_class_attributes(self, section_end):
        """Read the Class attributes = the first segment"""
        read_delim(self.buffer, 4)
        classes = []
        while self.buffer.tell() < section_end:
            package_index = read_int(self.buffer) - 1
            package = self.names[package_index] if package_index > 0 else ""
            name_idx = read_int(self.buffer) - 1
            name = self.names[name_idx] if name_idx > 0 else ""
            classes.append((package, name))
            read_delim(self.buffer, 2)
        return classes

    def _parse_prop_description(self, classes, section_end):
        """Parse the description of each property = the third segment"""
        read_delim(self.buffer, 6)
        object_info = []
        while self.buffer.tell() < section_end:
            class_idx = read_int(self.buffer) - 1
            class_type = classes[class_idx]
            read_delim(self.buffer, 2)
            indices = [x - 1 for x in read_int(self.buffer, 2)]
            obj_id = read_int(self.buffer)
            object_info.append((class_type, indices, obj_id))
        return object_info

    def _parse_properties(self, section_end):
        """
        Parse the actual values of the attributes == segments 2 and 4
        """
        read_delim(self.buffer, 2)
        props = []
        while self.buffer.tell() < section_end:
            n_props = read_int(self.buffer)
            d = parse_prop(n_props, self.buffer, self.names, self.heap)
            if not d:  # Empty dictionary
                break
            props.append(d)
            # Move to next 8-byte aligned offset
            self.buffer.seek(self.buffer.tell() + self.buffer.tell() % 8)
        return props

    def to_hdf(self, filename):
        f = h5py.File(filename, mode="w")
        save_to_hdf(f, "/", self.attrs)


def describe(d, indent=0, width=4, out=None):
    for key, value in d.items():
        print(f'{"": <{width * indent}}' + str(key), file=out)
        if isinstance(value, dict):
            describe(value, indent + 1, out=out)
        elif isinstance(value, np.ndarray):
            print(
                f'{"": <{width * (indent + 1)}} {value.shape} array '
                f"of type {value.dtype}",
                file=out,
            )
        elif isinstance(value, scipy.sparse.csc.csc_matrix):
            print(
                f'{"": <{width * (indent + 1)}} {value.shape} '
                f"sparse matrix of type {value.dtype}",
                file=out,
            )
        elif isinstance(value, Iterable) and not isinstance(value, str):
            print(
                f'{"": <{width * (indent + 1)}} {type(value)} of len ' f"{len(value)}",
                file=out,
            )
        else:
            print(f'{"": <{width * (indent + 1)}} {value}', file=out)


def parse_prop(n_props, buff, names, heap):
    d = dict()
    for i in range(n_props):
        name_idx, flag, heap_idx = read_int(buff, 3)
        if flag not in [0, 1, 2] and name_idx == 0:
            n_props = flag
            buff.seek(buff.tell() - 1)  # go back on one byte
            d = parse_prop(n_props, buff, names, heap)
        else:
            item_name = names[name_idx - 1]
            if flag == 0:
                d[item_name] = names[heap_idx]
            elif flag == 1:
                d[item_name] = heap[heap_idx + 2]  # Todo: what is the heap?
            elif flag == 2:
                assert 0 <= heap_idx <= 1, (
                    "Boolean flag has a value other " "than 0 or 1 "
                )
                d[item_name] = bool(heap_idx)
            else:
                raise ValueError(
                    "unknown flag {} for property {} with heap "
                    "index {}".format(flag, item_name, heap_idx)
                )
    return d


def is_object(x):
    """Checking object dtype for structured numpy arrays"""
    if x.dtype.names is not None and len(x.dtype.names) > 1:  # Complex obj
        return all(x.dtype[ix] == np.object for ix in range(len(x.dtype)))
    else:  # simple object
        return x.dtype == np.object


def flatten_obj(arr):
    # TODO turn structured arrays into nested dicts of lists rather that
    #  lists of dicts
    if isinstance(arr, np.ndarray):
        if arr.dtype.names:
            arrdict = dict()
            for fieldname in arr.dtype.names:
                arrdict[fieldname] = flatten_obj(arr[fieldname])
            arr = arrdict
        elif arr.dtype == np.object and arr.ndim == 0:
            arr = flatten_obj(arr[()])
        elif arr.dtype == np.object and arr.ndim > 0:
            try:
                arr = np.stack(arr)
                if arr.dtype.names:
                    d = {k: flatten_obj(arr[k]) for k in arr.dtype.names}
                    arr = d
            except:
                arr = [flatten_obj(x) for x in arr.tolist()]
    elif isinstance(arr, dict):
        arr = {k: flatten_obj(v) for k, v in arr.items()}
    elif isinstance(arr, list):
        try:
            arr = flatten_obj(np.stack(arr))
        except:
            arr = [flatten_obj(x) for x in arr]
    return arr


def save_to_hdf(h5file, path, dic):
    """
    Saving a MATLAB object to HDF5
    """
    if isinstance(dic, list):
        dic = {str(i): v for i, v in enumerate(dic)}
    for key, item in dic.items():
        if isinstance(item, (int, float, str)):
            h5file[path].attrs.create(key, item)
        elif isinstance(item, list):
            if len(item) == 0 and path + key not in h5file:  # empty list empty group
                h5file.create_group(path + key)
            if all(isinstance(x, (int, float, str)) for x in item):
                if path not in h5file:
                    h5file.create_group(path)
                h5file[path].attrs.create(key, item)
            else:
                if path + key not in h5file:
                    h5file.create_group(path + key)
                save_to_hdf(
                    h5file, path + key + "/", {str(i): x for i, x in enumerate(item)}
                )
        elif isinstance(item, scipy.sparse.csc.csc_matrix):
            try:
                h5file.create_dataset(
                    path + key, data=item.todense(), compression="gzip"
                )
            except Exception as e:
                print(path + key)
                raise e
        elif isinstance(item, (np.ndarray, np.int64, np.float64)):
            if item.dtype == np.dtype("<U1"):  # Strings to 'S' type for HDF5
                item = item.astype("S")
            try:
                h5file.create_dataset(path + key, data=item, compression="gzip")
            except Exception as e:
                print(path + key)
                raise e
        elif isinstance(item, dict):
            if path + key not in h5file:
                h5file.create_group(path + key)
            save_to_hdf(h5file, path + key + "/", item)
        elif item is None:
            continue
        else:
            raise ValueError(f"Cannot save {type(item)} type at key {path + key}")


## NOT YET FULLY IMPLEMENTED!


class _Info:
    def __init__(self, info):
        self.info = info
        self._identity = None

    def __getitem__(self, item):
        val = self.info[item]
        if val.shape[0] == 1:
            val = val[0]
        if 0 in val[1].shape:
            val = val[0]
        if isinstance(val, scipy.sparse.csc.csc_matrix):
            return np.asarray(val.todense())
        if val.dtype == np.dtype("O"):
            # 3d "sparse matrix"
            if all(isinstance(x, scipy.sparse.csc.csc_matrix) for x in val):
                val = np.array([x.todense() for x in val])
            # TODO: The actual object data
        equality = val[0] == val[1]
        if isinstance(equality, scipy.sparse.csc.csc_matrix):
            equality = equality.todense()
        if equality.all():
            val = val[0]
        return np.squeeze(val)

    @property
    def categories(self):
        return self.info.dtype.names


class TrapInfo(_Info):
    def __init__(self, info):
        """
        The information on all of the traps in a given position.

        :param info: The TrapInfo structure, can be found in the heap of
        the CTimelapse at index 7
        """
        super().__init__(info)


class CellInfo(_Info):
    def __init__(self, info):
        """
        The extracted information of all cells in a given position.
        :param info: The CellInfo structure, can be found in the heap
        of the CTimelapse at index 15.
        """
        super().__init__(info)

    @property
    def identity(self):
        if self._identity is None:
            self._identity = pd.DataFrame(
                zip(self["trapNum"], self["cellNum"]), columns=["trapNum", "cellNum"]
            )
        return self._identity

    def index(self, trapNum, cellNum):
        query = "trapNum=={} and cellNum=={}".format(trapNum, cellNum)
        try:
            result = self.identity.query(query).index[0]
        except Exception as e:
            print(query)
            raise e
        return result

    @property
    def nucEstConv1(self):
        return np.asarray(self.info["nuc_est_conv"][0][0].todense())

    @property
    def nucEstConv2(self):
        return np.asarray(self.info["nuc_est_conv"][0][1].todense())

    @property
    def mothers(self):
        return np.where((self["births"] != 0).any(axis=1))[0]

    def daughters(self, mother_index):
        """
        Get daughters of cell with index `mother_index`.

        :param mother_index: the index of the mother within the data. This is
        different from the mother's cell/trap identity.
        """
        daughter_ids = np.unique(self["daughterLabel"][mother_index]).tolist()
        daughter_ids.remove(0)
        mother_trap = self.identity["trapNum"].loc[mother_index]
        daughters = [self.index(mother_trap, cellNum) for cellNum in daughter_ids]
        return daughters


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    if not hasattr(matobj, "_fieldnames"):
        return matobj
    d = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, matlab.mio5_params.mat_struct):
            d[strg] = _todict(elem)
        elif isinstance(elem, np.ndarray):
            d[strg] = _toarray(elem)
        else:
            d[strg] = elem
    return d


def _toarray(ndarray):
    """
    A recursive function which constructs ndarray from cellarrays
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    """
    if ndarray.dtype != "float64":
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_toarray(sub_elem))
            else:
                elem_list.append(sub_elem)
        return np.array(elem_list)
    else:
        return ndarray


from pathlib import Path


class Strain:
    """The cell info for all the positions of a strain."""

    def __init__(self, origin, strain):
        self.origin = Path(origin)
        self.files = [x for x in origin.iterdir() if strain in str(x)]
        self.cts = [matObject(x) for x in self.files]
        self.cinfos = [CellInfo(x.heap[15]) for x in self.cts]
        self._identity = None

    def __getitem__(self, item):
        try:
            return np.concatenate([c[item] for c in self.cinfos])
        except ValueError:  # If first axis is the channel
            return np.concatenate([c[item] for c in self.cinfos], axis=1)

    @property
    def categories(self):
        return set.union(*[set(c.categories) for c in self.cinfos])

    @property
    def identity(self):
        if self._identity is None:
            identities = []
            for pos_id, cinfo in enumerate(self.cinfos):
                identity = cinfo.identity
                identity["position"] = pos_id
                identities.append(identity)
            self._identity = pd.concat(identities, ignore_index=True)
        return self._identity

    def index(self, posNum, trapNum, cellNum):
        query = "position=={} and trapNum=={} and cellNum=={}".format(
            posNum, trapNum, cellNum
        )
        try:
            result = self.identity.query(query).index[0]
        except Exception as e:
            raise e
        return result

    @property
    def mothers(self):
        # At least two births are needed to be considered a mother cell
        return np.where(np.count_nonzero(self["births"], axis=1) > 3)[0]

    def daughters(self, mother_index):
        """
        Get daughters of cell with index `mother_index`.

        :param mother_index: the index of the mother within the data. This is
        different from the mother's pos/trap/cell identity.
        """
        daughter_ids = np.unique(self["daughterLabel"][mother_index]).tolist()
        if 0 in daughter_ids:
            daughter_ids.remove(0)
        mother_pos_trap = self.identity[["position", "trapNum"]].loc[mother_index]
        daughters = []
        for cellNum in daughter_ids:
            try:
                daughters.append(self.index(*mother_pos_trap, cellNum))
            except IndexError:
                continue
        return daughters
