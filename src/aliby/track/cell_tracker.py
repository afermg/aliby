#!/usr/bin/env python
"""
TrackerCoordinator class to coordinate cell tracking and bud assignment.
"""
import pickle
import typing as t
from collections import Counter
from os.path import dirname, join
from pathlib import Path, PosixPath

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from agora.track_abc import FeatureCalculator

models_path = join(dirname(__file__), "../models")


class CellTracker(FeatureCalculator):
    """
    Class used to manage cell tracking. You can call it using an existing model or
    use the inherited CellTrainer to get a new one.

    Initialization parameters:

    :model: sklearn.ensemble.RandomForestClassifier object
    :trapfeats: Features to manually calculate within a trap
    :extrafeats: Additional features to calculate
    :model: Model to use, if provided ignores all other args but threshs
    :bak_model: Backup mode to use when prediction is unsure
    :nstepsback: int Number of timepoints to go back
    :thresh: float Cut-off value to assume a cell is not new
    :low_thresh: Lower thresh for model switching
    :high_thresh: Higher thresh for model switching.
        Probabilities between these two thresholds summon the
        backup model.

    :aweights: area weight for barycentre calculations


    # Feature order in array features
    1. basic features
    2. trap features (within a trap)
    3. extra features (process between imgs)

    """

    def __init__(
        self,
        feats2use=None,
        trapfeats=None,
        extrafeats=None,
        model=None,
        bak_model=None,
        thresh=None,
        low_thresh=None,
        high_thresh=None,
        nstepsback=None,
        aweights=None,
        red_fun=None,
        max_distance=None,
        **kwargs,
    ):

        if trapfeats is None:
            trapfeats = ()

        if extrafeats is None:
            extrafeats = ()

        if type(model) is str or type(model) is PosixPath:
            with open(Path(model), "rb") as f:
                model = pickle.load(f)

        if type(bak_model) is str or type(bak_model) is PosixPath:
            with open(Path(bak_model), "rb") as f:
                bak_model = pickle.load(f)

        if aweights is None:
            self.aweights = None

        if feats2use is None:  # Ignore this block when training
            if model is None:
                model = self.load_model(models_path, "ct_XGBC_20220703_11.pkl")
            if bak_model is None:
                bak_model = self.load_model(
                    models_path, "ct_XGBC_20220703_10.pkl"
                )
            self.model = model
            self.bak_model = bak_model

            main_feats = model.all_ifeats
            bak_feats = bak_model.all_ifeats
            feats2use, trapfeats, extrafeats = [
                tuple(sorted(set(main).union(bak)))
                for main, bak in zip(main_feats, bak_feats)
            ]

        # Training AND non-training part
        super().__init__(
            feats2use, trapfeats=trapfeats, extrafeats=extrafeats, **kwargs
        )

        self.extrafeats = tuple(extrafeats)

        self.all_ofeats = self.outfeats + trapfeats + extrafeats

        self.noutfeats = len(self.all_ofeats)

        if hasattr(self, "bak_model"):  # Back to non-training only

            self.mainof_ids = [
                self.all_ofeats.index(f) for f in self.model.all_ofeats
            ]

            self.bakof_ids = [
                self.all_ofeats.index(f) for f in self.bak_model.all_ofeats
            ]

        if nstepsback is None:
            nstepsback = 3
        self.nstepsback = nstepsback

        if thresh is None:
            thresh = 0.5
        self.thresh = thresh
        if low_thresh is None:
            low_thresh = 0.4
        if high_thresh is None:
            high_thresh = 0.6
        self.low_thresh, self.high_thresh = low_thresh, high_thresh

        if red_fun is None:
            red_fun = np.nanmax
        self.red_fun = red_fun

        if max_distance is None:
            self.max_distance = max_distance

    def get_feats2use(self):
        """
        Return feats to be used from a loaded random forest model model
        """
        nfeats = get_nfeats_from_model(self.model)
        nfeats_bak = get_nfeats_from_model(self.bak_model)
        # max_nfeats = max((nfeats, nfeats_bak))

        return (switch_case_nfeats(nfeats), switch_case_nfeats(nfeats_bak))

    def calc_feat_ndarray(self, prev_feats, new_feats):
        """
        Calculate feature ndarray using two ndarrays of features.
        ---

        input

        :prev_feats: ndarray (ncells, nfeats) of timepoint 1
        :new_feats: ndarray (ncells, nfeats) of timepoint 2

        returns

        :n3darray: ndarray (ncells_prev, ncells_new, nfeats) containing a
            cell-wise substraction of the features in the input ndarrays.
        """
        if not (new_feats.any() and prev_feats.any()):
            return np.array([])

        n3darray = np.empty((len(prev_feats), len(new_feats), self.ntfeats))

        # print('self: ', self, ' self.ntfeats: ', self.ntfeats, ' featsshape: ', new_feats.shape)
        for i in range(self.ntfeats):
            n3darray[..., i] = np.subtract.outer(
                prev_feats[:, i], new_feats[:, i]
            )

        n3darray = self.calc_dtfeats(n3darray)

        return n3darray

    def calc_dtfeats(self, n3darray):
        """
        Calculates features obtained between timepoints, such as distance
        for every pair of cells from t1 to t2.
        ---

        input

        :n3darray: ndarray (ncells_prev, ncells_new, ntfeats) containing a
            cell-wise substraction of the features in the input ndarrays.

        returns

        :newarray: 3d array taking the features specified in self.outfeats and self.trapfeats
        and adding dtfeats
        """
        newarray = np.empty(n3darray.shape[:-1] + (self.noutfeats,))
        newarray[..., : len(self.outfeats)] = n3darray[
            ..., : len(self.outfeats)
        ]
        newarray[
            ..., len(self.outfeats) : len(self.outfeats) + len(self.trapfeats)
        ] = n3darray[..., len(self.out_merged) :]
        for i, feat in enumerate(self.all_ofeats):
            if feat == "distance":
                newarray[..., i] = np.sqrt(
                    n3darray[..., self.xind] ** 2
                    + n3darray[..., self.yind] ** 2
                )

        return newarray

    def assign_lbls(
        self,
        prob_backtrace: np.ndarray,
        prev_lbls: t.List[t.List[int]],
        red_fun=None,
    ):
        """Assign labels using a prediction matrix of nxmxl where n is the number
        of cells in the previous image, m the number of steps back considered
        and l in the new image. It assigns the
        number zero if it doesn't find the cell.
        ---
        input

        :prob_backtrace: Probability n x m x l array obtained as an output of rforest
        :prev_labels: List of cell labels for previous timepoint to be compared.
        :red_fun: Function used to collapse the previous timepoints into one.
            If none provided it uses maximum and ignores np.nans.

        returns

        :new_lbls: ndarray of newly assigned labels obtained, new cells as
        zero.
        """
        if red_fun is None:
            red_fun = self.red_fun

        new_lbls = np.zeros(prob_backtrace.shape[2], dtype=int)
        pred_matrix = red_fun(prob_backtrace, axis=1)

        if pred_matrix.any():
            # assign available hits
            row_ids, col_ids = linear_sum_assignment(-pred_matrix)
            for i, j in zip(row_ids, col_ids):
                if pred_matrix[i, j] > self.thresh:
                    new_lbls[j] = prev_lbls[i]

        return new_lbls

    def predict_proba_from_ndarray(
        self,
        array_3d: np.ndarray,
        model: str = None,
        boolean: bool = False,
        max_distance: float = None,
    ):
        """

        input

        :array_3d: (ncells_tp1, ncells_tp2, out_feats) ndarray
        :model: str, {'model', 'bak_model'} can force a unique model instead of an ensemble
        :boolean: bool, if False returns probability, if True returns prediction
        :max_distance: float Maximum distance (in um) to be considered. If None it uses the instance's value,
        if zero it skips checking distances.

        requires
        :self.model:
        :self.mainof_ids: list of indices corresponding to the main model's features
        :self.bakof_ids: list of indices corresponding to the backup model's features

        returns

        (ncells_tp1, ncells_tp2) ndarray with probabilities or prediction
            of cell identities depending on "boolean" arg.

        """

        if array_3d.size == 0:
            return np.array([])

        if model is None:
            model2use = self.model
            bak_model2use = self.bak_model
            bakof_ids = self.bakof_ids
            mainof_ids = self.mainof_ids
        else:
            model2use = getattr(self, "model")
            bak_model2use = model2use
            bakof_ids = [
                self.all_ofeats.index(f) for f in model2use.all_ofeats
            ]
            mainof_ids = [
                self.all_ofeats.index(f) for f in model2use.all_ofeats
            ]

        fun2use = "predict" if boolean else "predict_proba"
        predict_fun = getattr(model2use, fun2use)
        bak_pred_fun = getattr(bak_model2use, fun2use)

        if max_distance is None:
            max_distance = self.max_distance

        orig_shape = array_3d.shape[:2]

        # Ignore cells that are too far away to possibly be the same
        cells_near = np.ones(orig_shape, dtype=bool)
        if max_distance and set(self.all_ofeats).intersection(
            ("distance", "centroid-0", "centroid-1")
        ):
            if "distance" in self.all_ofeats:
                cells_near = (
                    array_3d[..., self.all_ofeats.index("distance")]
                    < max_distance
                )
            else:  # Calculate euclidean distance
                cells_near = (
                    np.sqrt(
                        array_3d[..., self.all_ofeats.index("centroid-0")]
                        + array_3d[..., self.all_ofeats.index("centroid-1")]
                    )
                    < max_distance
                )

        pred_matrix = np.zeros(orig_shape)
        prob = np.zeros(orig_shape)
        if cells_near.any():
            prob = predict_fun(array_3d[cells_near][:, mainof_ids])[:, 1]
            uncertain_dfeats = (self.low_thresh < prob) & (
                prob < self.high_thresh
            )
            if uncertain_dfeats.any():
                bak_prob = bak_pred_fun(
                    array_3d[cells_near][uncertain_dfeats][:, bakof_ids]
                )[:, 1]
                probs_compared = np.stack((prob[uncertain_dfeats], bak_prob))
                most_confident_proba = abs((probs_compared - 0.5)).argmax(
                    axis=0
                )
                prob[uncertain_dfeats] = probs_compared[
                    most_confident_proba, range(most_confident_proba.shape[0])
                ]

            pred_matrix[cells_near] = prob

        return pred_matrix

    def get_new_lbls(
        self,
        new_img,
        prev_lbls,
        prev_feats,
        max_lbl,
        new_feats=None,
        pixel_size=None,
        **kwargs,
    ):
        """
        Core function to calculate the new cell labels.

        ----

        input

        :new_img: ndarray (len, width, ncells) containing the cell outlines
        :max_lbl: int indicating the last assigned cell label
        :prev_feats: list of ndarrays of size (ncells x nfeatures)
        containing the features of previous timepoints
        :prev_lbls: list of list of ints corresponding to the cell labels in
            the previous timepoints
        :new_feats: (optional) Directly give a feature ndarray. It ignores
            new_img if given.
        :kwargs: Additional keyword values passed to self.predict_proba_from_ndarray

        returns

        :new_lbls: list of labels assigned to new timepoint
        :new_feats: list of ndarrays containing the updated features
        :new_max: updated max cell label assigned

        """

        if new_feats is None:
            new_feats = self.calc_feats_from_mask(new_img)

        if new_feats.any():
            if np.any([len(prev_feat) for prev_feat in prev_feats]):
                counts = Counter(
                    [lbl for lbl_set in prev_lbls for lbl in lbl_set]
                )
                lbls_order = list(counts.keys())
                probs = np.full(
                    (len(lbls_order), self.nstepsback, len(new_feats)), np.nan
                )

                for i, (lblset, prev_feat) in enumerate(
                    zip(prev_lbls, prev_feats)
                ):
                    if len(prev_feat):
                        feats_3darray = self.calc_feat_ndarray(
                            prev_feat, new_feats
                        )

                        pred_matrix = self.predict_proba_from_ndarray(
                            feats_3darray,
                            **kwargs,
                        )

                        for j, lbl in enumerate(lblset):
                            probs[lbls_order.index(lbl), i, :] = pred_matrix[
                                j, :
                            ]

                new_lbls = self.assign_lbls(probs, lbls_order)
                new_cells_pos = new_lbls == 0
                new_max = max_lbl + sum(new_cells_pos)
                new_lbls[new_cells_pos] = [*range(max_lbl + 1, new_max + 1)]

                # ensure that label output is consistently a list
                new_lbls = new_lbls.tolist()

            else:
                new_lbls = [*range(max_lbl + 1, max_lbl + len(new_feats) + 1)]

                new_max = max_lbl + len(new_feats)

        else:
            return ([], [], max_lbl)
        return (new_lbls, new_feats, new_max)

    def probabilities_from_impair(
        self,
        image_t1: np.ndarray,
        image_t2: np.ndarray,
        kwargs_feat_calc: dict = {},
        **kwargs,
    ):
        """
        Convenience function to test tracking between two time-points

        :image_t1: np.ndarray containing mask of first time-point
        :image_t2: np.ndarray containing mask of second time-point
        :kwargs_feat_calc: are passed to self.calc_feats_from_mask calls
        :kwargs: are passed to self.predict_proba_from_ndarray
        """
        feats_t1 = self.calc_feats_from_mask(image_t1, **kwargs_feat_calc)
        feats_t2 = self.calc_feats_from_mask(image_t2, **kwargs_feat_calc)

        probability_matrix = np.array([])
        if feats_t1.any() and feats_t2.any():
            feats_3darray = self.calc_feat_ndarray(feats_t1, feats_t2)

            probability_matrix = self.predict_proba_from_ndarray(
                feats_3darray, **kwargs
            )

        return probability_matrix

    # Step CellTracker
    def run_tp(
        self,
        masks: np.ndarray,
        state: t.Dict[str, t.Union[int, list]] = None,
        **kwargs,
    ) -> t.Dict[str, t.Union[t.List[int], t.Dict[str, t.Union[int, list]]]]:
        """Assign labels to new masks using a state dictionary.

        Parameters
        ----------
        masks : np.ndarray
            Cell masks to label.
        state : t.Dict[str, t.Union[int, list]]
            Dictionary containing maximum cell label, and previous cell labels
            and features for those cells.
        kwargs : keyword arguments
            Keyword arguments passed to self.get_new_lbls

        Returns
        -------
        t.Dict[str, t.Union[t.List[int], t.Dict[str, t.Union[int, list]]]]
            New labels and new state dictionary.

        Examples
        --------
        FIXME: Add example beyond the trivial one.

        import numpy as np
        from baby.tracker.core import celltracker
        from tqdm import tqdm

        # overlapping outlines are of shape (t,z,x,y)
        masks = np.zeros((5, 3, 20, 20), dtype=bool)
        masks[0, 0, 2:6, 2:6] = true
        masks[1:, 0, 13:14, 13:14] = true
        masks[:, 1, 8:12, 8:12] = true
        masks[:, 2, 14:18, 14:18] = true

        # 13um pixel size
        ct = celltracker(pixel_size=0.185)
        labels = []

        state = none
        for masks_tp in tqdm(masks):
            new_labels, state = ct.run_tp(masks_tp, state=state)
            labels.append(new_labels)

        # should result in state['cell_lbls']
        # [[1, 2, 3], [4, 2, 3], [4, 2, 3], [4, 2, 3], [4, 2, 3]]
        """
        if state is None:
            state = {}

        max_lbl = state.get("max_lbl", 0)
        cell_lbls = state.get("cell_lbls", [])
        prev_feats = state.get("prev_feats", [])

        # Get features for cells at this time point
        feats = self.calc_feats_from_mask(masks)

        lastn_lbls = cell_lbls[-self.nstepsback :]
        lastn_feats = prev_feats[-self.nstepsback :]

        new_lbls, _, max_lbl = self.get_new_lbls(
            masks, lastn_lbls, lastn_feats, max_lbl, feats, **kwargs
        )

        state = {
            "max_lbl": max_lbl,
            "cell_lbls": cell_lbls + [new_lbls],
            "prev_feats": prev_feats + [feats],
        }

        return (new_lbls, state)


# Helper functions


def switch_case_nfeats(nfeats):
    """
    Convenience TEMPORAL function to determine whether to use distance/location
    as a feature for tracking or not (nfeats=5 for no distance, 7 for distance)
    input
    number of feats

    returns
    list of main and extra feats based on the number of feats
    """
    main_feats = {
        4: [
            ("area", "minor_axis_length", "major_axis_length", "bbox_area"),
            (),
            (),
        ],
        5: [
            (
                "area",
                "minor_axis_length",
                "major_axis_length",
                "bbox_area",
                "perimeter",
            ),
            (),
            (),
        ],
        6: [
            (
                "area",
                "minor_axis_length",
                "major_axis_length",
                "bbox_area",
                "perimeter",
            ),
            (),
            ("distance",),
        ],
        # Including centroid
        # 7 : [('centroid', 'area', 'minor_axis_length', 'major_axis_length',
        #       'bbox_area', 'perimeter'), () , ()],
        7: [
            (
                "area",
                "minor_axis_length",
                "major_axis_length",
                "bbox_area",
                "perimeter",
            ),
            ("baryangle", "barydist"),
            (),
        ],
        8: [  # Minus centroid
            (
                "area",
                "minor_axis_length",
                "major_axis_length",
                "bbox_area",
                # "eccentricity",
                "equivalent_diameter",
                # "solidity",
                # "extent",
                "orientation",
                # "perimeter",
            ),
            ("baryangle", "barydist"),
            (),
        ],
        10: [  # Minus distance
            (
                "centroid",
                "area",
                "minor_axis_length",
                "major_axis_length",
                "bbox_area",
                # "eccentricity",
                "equivalent_diameter",
                # "solidity",
                # "extent",
                "orientation",
                # "perimeter",
            ),
            ("baryangle", "barydist"),
            (),
        ],
        11: [  # Minus computationally-expensive features
            (
                "centroid",
                "area",
                "minor_axis_length",
                "major_axis_length",
                "bbox_area",
                # "eccentricity",
                "equivalent_diameter",
                # "solidity",
                # "extent",
                "orientation",
                # "perimeter",
            ),
            ("baryangle", "barydist"),
            ("distance",),
        ],
        15: [  # All features
            (
                "centroid",
                "area",
                "minor_axis_length",
                "major_axis_length",
                "bbox_area",
                "eccentricity",
                "equivalent_diameter",
                "solidity",
                "extent",
                "orientation",
                "perimeter",
            ),
            ("baryangle", "barydist"),
            ("distance",),
        ],
    }

    assert nfeats in main_feats.keys(), "invalid nfeats"
    return main_feats.get(nfeats, [])


def get_nfeats_from_model(model) -> int:
    if isinstance(model, SVC):
        nfeats = model.support_vectors_.shape[-1]
    elif isinstance(model, RandomForestClassifier):
        nfeats = model.n_features_

    return nfeats
