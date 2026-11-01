# Authorized 282-position computational expansion

User authorization: proceed with the remaining 282 local primary positions; defer human/biological review until computational processing and integrity gates finish. Never duplicate the completed 104 positions or separate Gcd2 repair. `14190058:gcd7_020` remains excluded pending verified alternate raw data.

## Fixed scope and independent inputs

The new plan is the exact complement of the original 104-job plan plus Gcd2 and the raw-blocked site in the 388-primary-archive manifest. It contains 71,858 full-series timepoints, 16 experiments, and 22 experiment/channel-layout/acquired-Z families. The model assignments are 277 five-plane BF and five single-plane BF sites. The single-plane and five-plane model identities are unchanged from previous repairs. Models were copied privately, not downloaded into or modified in the shared cache; actual loaded CNN depths were checked.

`corpus_expansion_plan.py` preserves acquisition proposals and renders per-site raw channel previews. Representative previews for all 22 families and per-site pixel evidence expose acquisition-table/storage ordering differences in 57 sites in experiments 673 and 921. Their explicit stored order is GFP C0/BF C1; original sidecars/proposals remain intact. Fluorescence Z selections are independent and channel-specific. Camera/pixel calibration, spectral identity, image usability, treatment conflicts and biological interpretation are not certified.

## Staging and bounds

`corpus_assay_expansion.py` first processes one representative serially. Every representative gets fresh 3-frame sequential/async runs with exact submitted-input, mask and semantic-table parity, then fresh 12-frame paired runs, then a fresh complete-series async run. Following the first successful representative, remaining representatives run in disjoint batches of at most four sites. Nonrepresentative admission requires its family's intact parity and completed representative evidence, then a fresh three-frame smoke and a fresh full-series session.

There are at most four sites in flight and four BABY workers per site. The controller imposes a four-hour total site wall budget including preparation and diagnostics, a 48-hour ceiling for each disjoint batch, and a 300-GiB free-space reserve. The overall 282-position campaign consists of multiple disjoint batches, not one unbounded 48-hour batch. It stops on the first failure, drains/stops its owned processes, retains failed evidence and does not retry or resume. Operator `STOP` also stops admission. No alternative-model or ROI fallback exists.

All stages revalidate sealed plan/source/model evidence, raw archive identity, selected pixel availability, full-series geometric ROI provenance, submitted dimensions and HTTP acknowledgments, masks, exact profile/tracking keys, repeated masks and owned-process shutdown. Full-series geometric exclusions remain retrospective. Tile history is bounded at one frame; full feature and stateful lineage histories remain intact. New releases use no-replace publication. Sequential is the reference/default; async is explicit for this campaign only.

## Evidence

Analysis root: `/work/users/amunoz/aliby_baby/analyses/corpus-expansion-20261101`.

- `plan.json`, `metadata.json`, `proposals.json`: fixed scope/configuration and preserved earlier channel proposals.
- `sealed-inputs.json`, `source-identity.json`, `evidence/`: pinned source/model/configuration, regression results and launch receipts.
- `review/`: raw channel-identity previews, not human/biological sign-off.
- `campaign/status.json`, `events.jsonl`, eventual `terminal.json`: live computational state.
- `jobs/<record>--<site>/prepared/` and `attempts/`: independent input preparation, fresh stateful attempts, validation and shutdown receipts.
- `campaign/family-*-gate.json`: immutable representative evidence inventory, checked before admitting other family members.

The old releases, failed attempts, original checkouts and forecasting quarantine are untouched. Human segmentation/tracking review follows computation; all computational releases retain `biological_approval: false`.
