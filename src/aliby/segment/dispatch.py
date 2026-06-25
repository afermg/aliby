#!/usr/bin/env jupyter
"""

See all the available models at
https://cellpose.readthedocs.io/en/latest/models.html#full-built-in-models
"""

from functools import partial

import numpy as np
from skimage.segmentation import relabel_sequential


def _to_uint16_labels(labels: np.ndarray) -> np.ndarray:
    if labels.size and labels.max() >= np.iinfo(np.uint16).max:
        raise OverflowError(
            f"Segmentation produced {labels.max()} labels; uint16 cast unsafe."
        )
    return labels.astype(np.uint16, copy=False)


def dispatch_segmenter(
    kind: str, channel_to_segment: int, address: str = None, **kwargs
) -> callable:
    match kind:
        case "nahual_baby":
            # TODO update this with setup_process_dispatch
            from nahual.client.baby import load_model, process_data

            # Have a sensible set of defaults
            extra_args = {
                "refine_outlines": ("", "true"),
                "with_edgemasks": ("", "true"),
                "with_masks": ("", "true"),
            }

            modelset = kwargs.pop("modelset")
            assert modelset is not None, f"Missing modelset on {kind} segmentation"
            session_id = load_model(address, modelset)

            if "extra_args" in kwargs:
                for k, v in kwargs["extra_args"]:
                    extra_args[k] = v

            _process = partial(
                process_data,
                address=address,
                session_id=session_id,
                channel_to_segment=channel_to_segment,
                extra_args=tuple(extra_args.items()),
                return_metadata=True,
            )

            def segment(pixels):
                # With return_metadata=True, nahual already collapses each
                # tile to a 2-D label image and returns a dict with both the
                # masks (for extract via passed_data) and the per-tile
                # tracking/lineage metadata (consumed by
                # ``aliby.pipe_baby._save_baby_tracking_lineage``).
                result = _process(pixels)
                masks = [_to_uint16_labels(m) for m in result["masks"]]
                return {"masks": masks, "metadata": result["metadata"]}

            return segment
        case "nahual_cellpose":
            # Examples over at https://github.com/afermg/nahual/blob/master/examples/
            # Cellpose via a nahual running server
            from nahual.process import dispatch_setup_process

            assert address is not None, "You must provide an address if using Nahual."

            tool = kind.removeprefix("nahual_")

            setup, process = dispatch_setup_process(tool)

            setup_params = kwargs.get("setup_params", {})
            # eval_params = kwargs.get("eval_params", {})

            info = setup(setup_params, address=address)
            print(f"Cellpose via nahual set up. Remote returned {info}")

            remote = partial(process, address=address)

            def segment(*args, **kwargs):
                result = remote(*args, **kwargs)
                if isinstance(result, list):
                    return [_to_uint16_labels(r) for r in result]
                return _to_uint16_labels(result)

            return segment
        case "nahual_spotiflow":
            # Spotiflow (https://github.com/afermg/spotiflow/tree/nahual-wrap)
            # is a fluorescence-puncta detector. The wrap rasterises each
            # detected spot as a small disk so the server already returns a
            # ``(N, H, W)`` int32 instance-style label mask — drop-in
            # equivalent to cellpose for downstream feature extraction.
            #
            # Spotiflow isn't in nahual's OUTPUT_SIGNATURES registry yet, so
            # we pass the (dict, numpy) signature explicitly. The setup +
            # process roundtrip is otherwise identical to the cellpose path.
            from nahual.process import dispatch_setup_process

            assert address is not None, "You must provide an address if using Nahual."

            tool = kind.removeprefix("nahual_")

            # ``signature=("dict", "numpy")`` — first call (a dict) re-runs
            # setup on the server; subsequent numpy arrays go to process().
            setup, process = dispatch_setup_process(tool, signature=("dict", "numpy"))

            setup_params = kwargs.get("setup_params", {})
            info = setup(setup_params, address=address)
            print(f"Spotiflow via nahual set up. Remote returned {info}")

            remote = partial(process, address=address)

            def segment(pixels: np.ndarray, **kw):
                # Spotiflow is a 2D, single-channel detector. Aliby feeds
                # an FCZYX/FZYX array; pick the first channel of the first
                # Z-plane per batch entry — same selection cellpose does
                # for its 2D mode. The server itself iterates over N (the
                # batch dim) and returns ``(N, Y, X)`` int32.
                arr = np.asarray(pixels)
                if arr.ndim == 6:  # TFCZYX → drop T
                    arr = arr[0]
                if arr.ndim == 5:  # FCZYX
                    arr = arr[:, channel_to_segment : channel_to_segment + 1]
                elif arr.ndim == 4:  # FZYX (channel already selected upstream)
                    arr = arr[:, None]
                else:
                    raise ValueError(
                        f"nahual_spotiflow: unexpected pixel ndim={arr.ndim} "
                        f"(shape={arr.shape}); want FCZYX or TFCZYX."
                    )
                result = remote(arr)
                if isinstance(result, list):
                    return [_to_uint16_labels(r) for r in result]
                return _to_uint16_labels(result)

            return segment
        case "cellpose":  # One of the cellpose models
            # cellpose does without all the ABC stuff
            # It returns a function to segment
            from cellpose.models import CellposeModel

            # Meta parameters
            setup_params = kwargs.get("setup_params", {})
            gpu = setup_params.pop("gpu", True)
            device = setup_params.pop("device", None)

            # use custom models if fullpath is provided
            pretrained = {}

            model = CellposeModel(
                **pretrained,
                gpu=gpu,
                device=device,
            )

            # ensure it returns only masks
            # TODO generalise so it does not assume a 1-tile file
            def segment(
                pixels: np.ndarray,
                do_3D: bool = False,
                stitch_threshold: int | None = None,
                **kwargs,
            ) -> list[np.ndarray]:
                """Preprocess the input numpy array to feed into Cellpose.
                Assumes FCZYX pixels shape."""
                z_size = pixels.shape[2]

                if pixels.ndim > 5:  # If time dim was passed remove it
                    pixels = pixels[0]  # FCZYX

                pixels = pixels[:, channel_to_segment]  # FZYX
                z_axis = None
                # Cellpose gets annoying if we keep the z-dimension with one stack
                if do_3D and z_size > 1:
                    z_axis = 1
                    stitch_threshold = 0.01
                    normalize = dict(norm3D=False)
                else:
                    z_axis = None
                    stitch_threshold = 0.0
                    normalize = True
                    if z_size > 1:  # FYX
                        pixels = pixels.max(axis=1)
                    else:
                        pixels = pixels[:, 0]

                result = model.eval(
                    pixels,
                    do_3D=do_3D,
                    stitch_threshold=stitch_threshold,
                    normalize=normalize,
                    z_axis=z_axis,
                    **kwargs,
                )
                labels = result[0]
                ndim = labels.ndim
                if ndim == 3:  # Cellpose squeezes dims!
                    # TODO Check that this is the best way to project 3-D labels into 2D
                    labels = labels.max(axis=0)

                    # Cover case where the reduction on z removes an entire item
                    labels = relabel_sequential(labels)[0]

                elif not 1 < ndim < 4:
                    raise Exception(
                        f"Segmentation yielded {result.ndim} dimensions instead of 3"
                    )

                if labels.max() >= np.iinfo(np.uint16).max:
                    raise OverflowError(
                        f"Segmentation produced {labels.max()} labels; uint16 cast unsafe."
                    )
                return labels.astype(np.uint16, copy=False)

        case _:
            raise Exception(f"Invalid segmentation method {kind}")

    return segment
