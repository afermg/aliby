#!/usr/bin/env jupyter

import typing as t
from abc import ABC
from os.path import join

import numpy as np
from skimage.measure import regionprops_table

from aliby.track.utils import calc_barycentre, pick_baryfun


class FeatureCalculator(ABC):
    """
    Base class for making use of regionprops-based features.
    If no features are offered it uses most of them.

    This class is not to be used directly
    """

    a_ind = None
    ma_ind = None
    x_ind = None
    y_ind = None

    def __init__(
        self,
        feats2use: t.Collection[str],
        trapfeats: t.Optional[t.Collection[str]] = None,
        extrafeats: t.Optional[t.Collection[str]] = None,
        aweights: t.Optional[bool] = None,
        pixel_size: t.Optional[float] = None,
    ) -> None:

        self.feats2use = feats2use

        if trapfeats is None:
            trapfeats = ()
        self.trapfeats = tuple(trapfeats)

        if extrafeats is None:
            extrafeats = ()
        self.extrafeats = tuple(extrafeats)

        if aweights is None:
            aweights = None
        self.aweights = aweights

        if pixel_size is None:
            pixel_size = 0.182
        self.pixel_size = pixel_size

        self.outfeats = self.get_outfeats()

        self.set_named_ids()

        self.tfeats = self.outfeats + self.tmp_outfeats + self.trapfeats
        self.ntfeats = len(self.tfeats)

    def get_outfeats(
        self, feats2use: t.Optional[t.Collection[str]] = None
    ) -> tuple:
        if feats2use is None:
            feats2use = self.feats2use
        outfeats = tuple(
            regionprops_table(np.diag((1, 0)), properties=feats2use).keys()
        )
        return outfeats

    def set_named_ids(self):
        # Manage calling feature outputs by name
        d = {"centroid-0": "xind", "centroid-1": "yind", "area": "aind"}
        tmp_d = {
            "barydist": ["centroid", "area"],
            "baryangle": ["centroid", "area"],
            "distance": ["centroid"],
        }

        nonbase_feats = self.trapfeats + self.extrafeats

        tmp_infeats = np.unique([j for x in nonbase_feats for j in tmp_d[x]])
        self.tmp_infeats = tuple(
            [f for f in tmp_infeats if f not in self.feats2use]
        )

        # feats that are only used to calculate others
        tmp_outfeats = (
            self.get_outfeats(feats2use=tmp_infeats)
            if len(tmp_infeats)
            else []
        )

        self.tmp_outfeats = []
        for feat in tmp_outfeats:
            if feat in self.outfeats:
                setattr(self, d[feat], self.outfeats.index(feat))
            else:  # Only add them if not in normal outfeats
                self.tmp_outfeats.append(feat)
                setattr(
                    self,
                    d[feat],
                    len(self.outfeats) + self.tmp_outfeats.index(feat),
                )

        self.tmp_outfeats = tuple(self.tmp_outfeats)
        self.out_merged = self.outfeats + self.tmp_outfeats

    def load_model(self, path, fname):
        model_file = join(path, fname)
        with open(model_file, "rb") as file_to_load:
            model = pickle.load(file_to_load)

        return model

    def calc_feats_from_mask(
        self,
        masks: np.ndarray,
        feats2use: t.Optional[t.Tuple[str]] = None,
        trapfeats: t.Optional[t.Tuple[str]] = None,
        scale: t.Optional[bool] = True,
        pixel_size: t.Optional[float] = None,
    ):
        """
        Calculate feature ndarray from ndarray of cell masks
        ---
        input

        :masks: ndarray (ncells, x_size, y_size), typically dtype bool
        :feats2use: list of strings with the feature properties to extract.
            If it is None it uses the ones set in self.feats2use.
        :trapfeats: List of str with additional features to use
            calculated immediately after basic features.
        :scale: bool, if True scales mask to a defined pixel_size.
        :pixel_size: float, used to rescale the object features.

        returns

        (ncells, nfeats) ndarray of features for input masks
        """
        if pixel_size is None:
            pixel_size = self.pixel_size

        if feats2use is None:
            feats2use = self.feats2use + self.tmp_infeats

        if trapfeats is None:
            trapfeats = self.trapfeats

        ncells = masks.shape[0] if masks.ndim == 3 else masks.max()
        feats = np.empty((ncells, self.ntfeats))  # ncells * nfeats
        if masks.any():
            if masks.ndim == 3:  # Individual cells in dim 0
                assert masks.sum(
                    axis=(1, 2)
                ).all(), "Dimension with at least one empty outline slice"

                cell_feats = np.array(
                    [
                        [
                            x[0]
                            for x in regionprops_table(
                                mask.astype(int), properties=feats2use
                            ).values()
                        ]
                        for mask in masks
                    ]
                )

            elif masks.ndim == 2:  # No overlap between cells
                cell_feats = np.array(
                    [
                        x
                        for x in regionprops_table(
                            masks.astype(int), properties=feats2use
                        ).values()
                    ]
                ).T
            else:
                raise Exception(
                    "TrackerException: masks do not have the appropiate dimensions"
                )

            if scale:
                cell_feats = self.scale_feats(cell_feats, pixel_size)

            # Fill first sector, with directly extracted features
            feats[:, : len(self.out_merged)] = cell_feats
            if trapfeats:  # Add additional features

                tfeats = self.calc_trapfeats(feats)
                feats[:, len(self.out_merged) :] = tfeats
        else:

            feats = np.zeros((0, self.ntfeats))

        return feats

    def calc_trapfeats(self, basefeats):
        """
        Calculate trap-based features
        using basic ones.
        :basefeats: (n basic outfeats) 1-D array with features outputed by
            skimage.measure.regionprops_table


        requires
            self.aind
            self.aweights
            self.xind
            self.yind
            self.trapfeats

        returns
        (ntrapfeats) 1-D array with
        """
        if self.aweights is not None:
            weights = basefeats[:, self.aind]
        else:
            weights = None

        barycentre = calc_barycentre(
            basefeats[:, [self.xind, self.yind]], weights=weights
        )

        trapfeat_nd = np.empty((basefeats.shape[0], len(self.trapfeats)))
        for i, trapfeat in enumerate(self.trapfeats):
            trapfeat_nd[:, i] = pick_baryfun(trapfeat)(
                basefeats[:, [self.xind, self.yind]], barycentre
            )

        return trapfeat_nd

    def scale_feats(self, feats: np.ndarray, pixel_size: float):
        """
        input

        :feats: np.ndarray (ncells * nfeatures)
        :pixel_size: float Value used to normalise the images.

        returns
        Rescaled list of feature values
        """
        area = pixel_size**2

        scaling = {None: 1, "linear": pixel_size, "square": area}
        degrees_feats = {
            None: ["eccentricity", "extent", "orientation", "solidity"],
            "linear": [
                "centroid-0",
                "centroid-1",
                "minor_axis_length",
                "major_axis_length",
                "perimeter",
                "perimeter_crofton",
                "equivalent_diameter",
            ],
            "square": [
                "area",
                "convex_area",
                "bbox_area",
                "equivalent_diameter",
            ],
        }

        scaler = {
            feat: scaling[k]
            for k, feats in degrees_feats.items()
            for feat in feats
        }
        # for k in feats.keys():
        #     feats[k] /= scaler[k]

        return feats * [scaler[feat] for feat in self.out_merged]
