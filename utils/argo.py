import io
import operator
from pathlib import Path, PosixPath
from collections import Counter
from datetime import datetime
import re
import csv

import numpy as np

from tqdm import tqdm

from logfile_parser import Parser
from omero.gateway import BlitzGateway, TagAnnotationWrapper


class OmeroExplorer:
    def __init__(self, host, user, password, min_date=(2020, 6, 1)):
        self.conn = BlitzGateway(user, password, host=host)
        self.conn.connect()

        self.min_date = min_date
        self.backups = {}
        self.removed = []

    @property
    def cache(self):
        if not hasattr(self, "_cache"):
            self._cache = {v.id: get_annotsets(v) for v in self.dsets}
        return self._cache

    @property
    def raw_log(self):
        return {k: v["log"] for k, v in self.cache.items()}

    @property
    def raw_log_end(self):
        if not hasattr(self, "_raw_log_end"):
            self._raw_log_end = {d.id: get_logfile(d) for d in self.dsets}
        return self._raw_log_end

    @property
    def log(self):
        return {k: parse_annot(v, "log") for k, v in self.raw_log.items()}

    @property
    def raw_acq(self):
        return {k: v["acq"] for k, v in self.cache.items()}

    @property
    def acq(self):
        return {k: parse_annot(v, "acq") for k, v in self.raw_acq.items()}

    def load(self, min_id=18000, min_date=None):
        """
        :min_id: int
        :min_date: tuple
        """
        if min_date is None:
            min_date = self.min_date
        self._dsets_bak = [
            d for d in self.conn.getObjects("Dataset") if d.getId() > min_id
        ]

        if min_date:
            if len(min_date) < 3:
                min_date = min_date + tuple([1 for i in range(3 - len(min_date))])
            min_date = datetime(*min_date)

            # sort by dates
            dates = [d.getDate() for d in self._dsets_bak]
            self._dsets_bak[:] = [a for b, a in sorted(zip(dates, self._dsets_bak))]

            self._dsets_bak = [d for d in self._dsets_bak if d.getDate() >= min_date]

        self.dsets = self._dsets_bak
        self.n_dsets

    def dset(self, n):
        try:
            return [x for x in self.dsets if x.id == n][0]
        except:
            return

    def channels(self, setkey, present=True):
        """
        :setkey: str indicating a set of channels
        :present: bool indicating whether the search should or shold not be in the dset
        """
        self.dsets = [
            v for v in self.acqs.values() if present == has_channels(v, setkey)
        ]
        self.n_dsets

    def update_cache(self):
        if not hasattr(self, "acq") or not hasattr(self, "log"):
            for attr in ["acq", "log"]:
                print("Updating raw ", attr)
                setattr(
                    self,
                    "raw_" + attr,
                    {v.id: get_annotsets(v)[attr] for v in self.dsets},
                )
                setattr(
                    self,
                    attr,
                    {
                        v.id: parse_annot(getattr(self, "raw_" + attr)[v.id], attr)
                        for v in self.dsets
                    },
                )
        else:

            for attr in ["acq", "log", "raw_acq", "raw_log"]:
                setattr(
                    self, attr, {i.id: getattr(self, attr)[i.id] for i in self.dsets}
                )

    @property
    def dsets(self):
        if not hasattr(self, "_dsets"):
            self._dsets = self.load()

        return self._dsets

    @dsets.setter
    def dsets(self, dsets):
        if hasattr(self, "_dsets"):
            if self._dsets is None:
                self._dsets = []
            self.removed += [
                x for x in self._dsets if x.id not in [y.id for y in dsets]
            ]

        self._dsets = dsets

    def tags(self, tags, present=True):
        """
        :setkey str tags to filter
        """
        if type(tags) is not list:
            tags = [str(tags)]

        self.dsets = [v for v in self.dsets if present == self.has_tags(v, tags)]
        self.n_dsets

    @property
    def all_tags(self):
        if not hasattr(self, "_tags"):
            self._tags = {
                d.id: [
                    x.getValue()
                    for x in d.listAnnotations()
                    if isinstance(x, TagAnnotationWrapper)
                ]
                for d in self.dsets
            }
        return self._tags

    def get_timepoints(self):
        self.image_wrappers = {d.id: list(d.listChildren())[0] for d in self.dsets}

        return {k: i.getSizeT() for k, i in self.image_wrappers.items()}

    def timepoints(self, n, op="greater"):
        "Filter experiments using the number of timepoints"
        op = operator.gt if op == "greater" else operator.le
        self._timepoints = self.get_timepoints()

        self.dsets = [v for v in tqdm(self.dsets) if op(self._timepoints[v.id], n)]

    def microscope(self, microscope):
        self.microscopes = {
            dset.id: self.get_microscope(self.log[dset.id]) for dset in self.dsets
        }

        self.n_dsets

    def get_microscope(self, parsed_log):
        return parsed_log["microscope"]

    def reset(self, backup_id=None):
        self.dsets = self.backups.get(backup_id, self._dsets_bak)
        self.n_dsets

    def backup(self, name):
        self.backups[name] = self.dsets

    def reset_backup(self, name):
        self.dsets = self.backups[name]

    def cExperiment(self, present=True):
        self.dsets = [
            v
            for v in self.dsets
            if present
            * sum(
                [
                    "cExperiment" in x.getFileName()
                    for x in v.listAnnotations()
                    if hasattr(x, "getFileName")
                ]
            )
        ]
        self.n_dsets

    @staticmethod
    def is_complete(logfile):
        return logfile.endswith("Experiment completed\r\r\n")

    @staticmethod
    def contains_regex(logfile):
        pass
        # return re.

    def tiler_cells(self, present=True):
        self.__dsets = [v for v in self.dsets if present == tiler_cells_load(v)]

    @property
    def n_dsets(self):
        print("{} datasets.".format(len(self.dsets)))

    @property
    def desc(self):
        for d in self.dsets:
            print(
                "{}\t{}\t{}\t{}".format(
                    d.getDate().strftime("%x"),
                    d.getId(),
                    d.getName(),
                    d.getDescription(),
                )
            )

    @property
    def ids(self):
        return [d.getId() for d in self.dsets]

    # @property
    # def acqs(self):
    #     if not hasattr(self, "_acqs") or len(self.__dict__.get("_acqs", [])) != len(
    #         self.dsets
    #     ):
    #         self._acqs = [get_annot(get_annotsets(d), "acq") for d in self.dsets]
    #     return self._acqs

    def get_ph_params(self):
        t = [
            {
                ch: [exp, v]
                for ch, exp, v in zip(j["channel"], j["exposure"], j["voltage"])
                if ch in {"GFPFast", "pHluorin405"}
            }
            for j in [i["channels"] for i in self.acqs]
        ]

        ph_param_pairs = [(tuple(x.values())) for x in t if np.all(list(x.values()))]

        return Counter([str(x) for x in ph_param_pairs])

    def find_duplicate_candidates(self, days_tol=2):
        # Find experiments with the same name or Aim and from similar upload dates
        # and group them for cleaning
        pass

    def group_by_date(tol=1):
        dates = [x.getDate() for x in self.dsets]
        distances = np.array(
            [[abs(convert_to_hours(a - b)) for a in dates] for b in dates]
        )
        return explore_booldiag(distances > tol, 0, [])

    @property
    def complete(self):
        self.completed = {k: self.is_complete(v) for k, v in self.raw_log_end.items()}
        self.dsets = [dset for dset in self.dsets if self.completed[dset.id]]
        return self.n_dsets

    def save(self, fname):
        with open(fname + ".tsv", "w") as f:
            writer = csv.writer(f, delimiter="\t")
            for d in self.dsets:
                writer.writerow(
                    [
                        d.getDate().strftime("%x"),
                        d.getId(),
                        d.getName(),
                        d.getDescription(),
                    ]
                )

    @property
    def positions(self):
        return {x.id: len(list(x.listChildren())) for x in self.dsets}

    def has_tags(self, d, tags):
        if set(tags).intersection(self.all_tags[d.id]):
            return True


def explore_booldiag(bool_field, current_position, cluster_start_end):
    # Recursively find the square clusters over the diagonal. Allows for duplicates
    # returns a list of tuples with the start, end of clusters
    if current_position < len(bool_field) - 1:
        elements = np.where(bool_field[current_position])
        if len(elements[0]) > 1:
            start = elements[0][0]
            end = elements[0][-1]
        else:
            start = elements[0][0]
            end = elements[0][0]

        cluster_start_end.append((start, end))
        return explore_square(bool_field, end + 1, cluster_start_end)
    else:
        return cluster_start_end
    _


def convert_to_hours(delta):
    total_seconds = delta.total_seconds()
    hours = int(total_seconds // 3600)
    return hours


class Argo(OmeroExplorer):
    def __init__(self,*args, **kwargs):
        super().__init__(*args,**kwargs)


def get_creds():
    return (
        "upload",
        "***REMOVED***",  # OMERO Password
    )


def list_files(dset):
    return {x for x in dset.listAnnotations() if hasattr(x, "getFileName")}


def annot_from_dset(dset, kind):
    v = [x for x in dset.listAnnotations() if hasattr(x, "getFileName")]
    infname = kind if kind is "log" else kind.title()
    try:
        acqfile = [x for x in v if x.getFileName().endswith(infname + ".txt")][0]
        decoded = list(acqfile.getFileInChunks())[0].decode("utf-8")
        acq = parse_annot(decoded, kind)
    except:
        return {}

    return acq


def check_channels(acq, channels, _all=True):
    I = set(acq["channels"]["channel"]).intersection(channels)

    condition = False
    if _all:
        if len(I) == len(channels):
            condition = True
    else:
        if len(I):
            condition = True

    return condition


def get_chs(exptype):
    exptypes = {
        "dual_ph": ("GFP", "pHluorin405", "mCherry"),
        "ph": ("GFP", "pHluorin405"),
        "dual": ("GFP", "mCherry"),
        "mCherry": ("mCherry"),
    }
    return exptypes[exptype]


def load_annot_from_cache(exp_id, cache_dir="cache/"):
    if type(cache_dir) is not PosixPath:
        cache_dir = Path(cache_dir)

    annot_sets = {}
    for fname in cache_dir.joinpath(exp_id).rglob("*.txt"):
        fmt = fname.name[:3]
        with open(fname, "r") as f:
            annot_sets[fmt] = f.read()

    return annot_sets


def get_annot(annot_sets, fmt):
    """
    Get parsed annotation file
    """
    str_io = annot_sets.get(fmt, None)
    return parse_annot(str_io, fmt)


def parse_annot(str_io, fmt):
    parser = Parser("multiDGUI_" + fmt + "_format")
    return parser.parse(io.StringIO(str_io))


def get_log_date(annot_sets):
    log = get_annot(annot_sets, "log")
    return log.get("date", None)


def get_log_microscope(annot_sets):
    log = get_annot(annot_sets, "log")
    return log.get("microscope", None)


def get_annotsets(dset):
    annot_files = [
        annot.getFile() for annot in dset.listAnnotations() if hasattr(annot, "getFile")
    ]
    annot_sets = {
        ftype[:-4].lower(): annot
        for ftype in ("log.txt", "Acq.txt", "Pos.txt")
        for annot in annot_files
        if ftype in annot.getName()
    }
    annot_sets = {
        key: list(a.getFileInChunks())[0].decode("utf-8")
        for key, a in annot_sets.items()
    }
    return annot_sets


# def has_tags(d, tags):
#     if set(tags).intersection(annot_from_dset(d, "log").get("omero_tags", [])):
#         return True


def load_acq(dset):
    try:
        acq = annot_from_dset(dset, kind="acq")
        return acq
    except:
        print("dset", dset.getId(), " failed acq loading")
        return False


def has_channels(dset, exptype):
    acq = load_acq(dset)
    if acq:
        return check_channels(acq, get_chs(exptype))
    else:
        return


def get_id_from_name(exp_name, conn=None):
    if conn is None:
        conn = BlitzGateway(*get_creds(), host="islay.bio.ed.ac.uk", port=4064)

    if not conn.isConnected():
        conn.connect()

    cand_dsets = [
        d
        for d in conn.getObjects("Dataset")  # , opts={'offset': 10600,
        #      'limit':500})
        if exp_name in d.name
    ]  # increase the offset for better speed

    # return cand_dsets
    if len(cand_dsets) > 1:
        # Get date and try to find it using date and microscope name and date

        # found = []
        # for cand in cand_dsets:
        #     annot_sets = get_annotsets(cand)
        #     date = get_log_date(annot_sets)
        #     microscope = get_log_microscope(annot_sets)
        #     if date==date_name and microscope == microscope_name:
        #         found.append(cand)

        # if True:#len(found)==1:
        #     return best_cand.id#best_cand = found[0]
        if True:

            print("Multiple options found. Selecting the one with most children")

            max_dset = np.argmax(
                [
                    len(list(conn.getObject("Dataset", c.id).listChildren()))
                    for c in cand_dsets
                ]
            )

            best_cand = cand_dsets[max_dset]

            return best_cand.id
    elif len(cand_dsets) == 1:
        return cand_dsets[0].id


# Custom functions
def compare_dsets_voltages_exp(dsets):
    a = {}
    for d in dsets:
        try:
            acq = annot_from_dset(d, kind="acq")["channels"]
            a[d.getId()] = {
                k: (v, e)
                for k, v, e in zip(acq["channel"], acq["voltage"], acq["exposure"])
            }

        except:
            print(d, "didnt work")

    return a


def get_logfile(dset):
    annot_file = [
        annot.getFile()
        for annot in dset.listAnnotations()
        if hasattr(annot, "getFile") and annot.getFileName().endswith("log.txt")
    ][0]
    return list(annot_file.getFileInChunks())[-1].decode("utf-8")


# 19920 -> 19300/19310
#
