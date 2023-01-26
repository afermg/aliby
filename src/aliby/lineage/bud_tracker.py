#!/usr/bin/env jupyter

import pickle
from os.path import join

import numpy as np
from skimage.draw import polygon

from agora.track_abc import FeatureCalculator

models_path = join(dirname(__file__), "./models")


class BudTracker(FeatureCalculator):
    def __init__(self, model=None, feats2use=None, **kwargs):

        if model is None:
            model_file = join(models_path, "mb_model_20201022.pkl")
            with open(model_file, "rb") as file_to_load:
                model = pickle.load(file_to_load)
        self.model = model

        if feats2use is None:
            feats2use = ["centroid", "area", "minor_axis_length"]
        super().__init__(feats2use, **kwargs)

        self.a_ind = self.outfeats.index("area")
        self.ma_ind = self.outfeats.index("minor_axis_length")

    ### Assign mother-
    def calc_mother_bud_stats(self, p_budneck, p_bud, masks, feats=None):
        """
        ---

        input

        :p_budneck: 2d ndarray (size_x, size_y) giving the probability that a
            pixel corresponds to a bud neck
        :p_bud: 2d ndarray (size_x, size_y) giving the probability that a pixel
            corresponds to a bud
        :masks: 3d ndarray (ncells, size_x, size_y)
        :feats: ndarray (ncells, nfeats)

        NB: ASSUMES FEATS HAVE ALREADY BEEN NORMALISED!

        returns

        :n2darray: 2d ndarray (ncells x ncells, n_feats) specifying,
            for each pair of cells in the masks array, the features used for
            mother-bud pair prediction (as per 'feats2use')
        """

        if feats is None:
            feats = self.calc_feats_from_mask(masks)
        elif len(feats) != len(masks):
            raise Exception("number of features must match number of masks")

        ncells = len(masks)

        # Entries will be NaN unless validly specified below
        p_bud_mat = np.nan * np.ones((ncells, ncells))
        p_budneck_mat = np.nan * np.ones((ncells, ncells))
        budneck_ratio_mat = np.nan * np.ones((ncells, ncells))
        size_ratio_mat = np.nan * np.ones((ncells, ncells))
        adjacency_mat = np.nan * np.ones((ncells, ncells))

        for m in range(ncells):
            for d in range(ncells):
                if m == d:
                    # Mother-bud pairs can only be between different cells
                    continue

                p_bud_mat[m, d] = np.mean(p_bud[masks[d].astype("bool")])

                a_i = self.a_ind
                size_ratio_mat[m, d] = feats[m, a_i] / feats[d, a_i]

                # Draw rectangle
                r_points = self.get_rpoints(feats, d, m)
                if r_points is None:
                    continue
                rr, cc = polygon(
                    r_points[0, :], r_points[1, :], p_budneck.shape
                )
                if len(rr) == 0:
                    # Rectangles with zero size are not informative
                    continue

                r_im = np.zeros(p_budneck.shape, dtype="bool")
                r_im[rr, cc] = True

                # Calculate the mean of bud neck probabilities greater than some threshold
                pbn = p_budneck[r_im].flatten()

                pbn = pbn[pbn > 0.2]
                p_budneck_mat[m, d] = np.mean(pbn) if len(pbn) > 0 else 0

                # Normalise number of bud-neck positive pixels by the scale of
                # the bud (a value proportional to circumference):
                raw_circumf_est = np.sqrt(feats[d, a_i]) / self.pixel_size
                budneck_ratio_mat[m, d] = pbn.sum() / raw_circumf_est

                # Adjacency is the proportion of the joining rectangle that overlaps the mother daughter union
                md_union = masks[m] | masks[d]
                adjacency_mat[m, d] = np.sum(md_union & r_im) / np.sum(r_im)

        return np.hstack(
            [
                s.flatten()[:, np.newaxis]
                for s in (
                    p_bud_mat,
                    size_ratio_mat,
                    p_budneck_mat,
                    budneck_ratio_mat,
                    adjacency_mat,
                )
            ]
        )

    def predict_mother_bud(self, p_budneck, p_bud, masks, feats=None):
        """
        ---

        input

        :p_budneck: 2d ndarray (size_x, size_y) giving the probability that a
            pixel corresponds to a bud neck
        :p_bud: 2d ndarray (size_x, size_y) giving the probability that a pixel
            corresponds to a bud
        :masks: 3d ndarray (ncells, size_x, size_y)
        :feats: ndarray (ncells, nfeats)

        returns

        :n2darray: 2d ndarray (ncells, ncells) giving probability that a cell
            (row) is a mother to another cell (column)
        """

        ncells = len(masks)

        mb_stats = self.calc_mother_bud_stats(p_budneck, p_bud, masks, feats)

        good_stats = ~np.isnan(mb_stats).any(axis=1)
        # Assume probability of bud assignment for any rows that are NaN will
        # be zero
        ba_probs = np.zeros(ncells**2)
        if good_stats.any():
            ba_probs[good_stats] = self.model.predict_proba(
                mb_stats[good_stats, :]
            )[:, 1]
        ba_probs = ba_probs.reshape((ncells,) * 2)

        return ba_probs

    def get_rpoints(self, feats, d, m):
        """
        Draw a rectangle in the budneck of cells
        ---

        NB: ASSUMES FEATS HAVE ALREADY BEEN NORMALISED!

        input

        feats: 2d ndarray (ncells, nfeats)

        returns

        r_points: 2d ndarray (2,4) with the coordinates of the rectangle corner

        """

        # Get un-scaled features for m-d pair
        descaled_feats = feats / self.pixel_size
        m_centre = descaled_feats[m, :2]
        d_centre = descaled_feats[d, :2]
        r_width = np.max((2, descaled_feats[d, self.ma_ind] * 0.25))

        # Draw connecting rectangle
        r_hvec = d_centre - m_centre
        r_wvec = np.matmul(np.array([[0, -1], [1, 0]]), r_hvec)
        r_wvec_len = np.linalg.norm(r_wvec)
        if r_wvec_len == 0:
            return None
        r_wvec = r_width * r_wvec / r_wvec_len
        r_points = np.zeros((2, 4))
        r_points[:, 0] = m_centre - 0.5 * r_wvec
        r_points[:, 1] = r_points[:, 0] + r_hvec
        r_points[:, 2] = r_points[:, 1] + r_wvec
        r_points[:, 3] = r_points[:, 2] - r_hvec

        return r_points
