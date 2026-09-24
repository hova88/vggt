# NuScenes CAM_FRONT road-state experiment

The road classifier reads four ordered `CAM_FRONT` frames, takes VGGT's final
aggregator features, and predicts five classes: elevated up/down, main road,
side road, and intersection. The road head and VGGT wrapper live in
[`vggt/heads/road_probability_head.py`](../vggt/heads/road_probability_head.py).
Training, evaluation, calibration, and inference commands are in
[`scripts/README_road_head.md`](../scripts/README_road_head.md).

NuScenes has no ground truth for these five classes. The dataset reads
external `label` or `soft_label` values by `sample_token`; the label applies
to the final frame. Scene-disjoint splitting prevents adjacent frames from
crossing the normal train/validation boundary. The dedicated two-GPU
memorization script deliberately evaluates on its training samples.

## Pretrained weights and checkpoints

`scripts/road_common.py` loads the VGGT aggregator from a local `model.pt`
when `model.checkpoint` is set. Otherwise it loads the official Hugging Face
checkpoint. The road head is initialized separately. `head_only` freezes the
aggregator; `last_blocks` trains the final frame and global blocks with a
smaller learning rate. Training checkpoints store only the road head and
updated aggregator parameters, plus the local base-checkpoint path.

The standard script selects `best.pt` by validation NLL. The calibration
script fits one positive temperature on held-out validation logits. Inference
reports posterior probabilities and entropy; optional prior correction yields
an HMM observation score, not a trained HMM.

## v1.0-mini proposal archive

The full `v1.0-mini` archive covers 404 `CAM_FRONT` sample keyframes in 10
scenes. The first three targets of each scene use repeated earliest frames
to form a four-frame context. Non-keyframe `sweeps/CAM_FRONT` images are outside
this sample-token archive.

| Qwen Flash proposal | Count |
| --- | ---: |
| Elevated up | 0 |
| Elevated down | 0 |
| Main road | 305 |
| Side road | 2 |
| Intersection | 71 |
| Abstain | 26 |

The proposals are in
[`road_labels_v1mini_all_samples_qwen_flash_proposals.jsonl`](../outputs/road_head/road_labels_v1mini_all_samples_qwen_flash_proposals.jsonl).
They use `proposed_label` and `review_status=unverified_model_proposal`, not
the training parser's `label` field. Visual review found false intersection
proposals, especially in `scene-1100`. The 26 abstentions belong to the
parking/internal-road segment of `scene-0916`; a Qwen Max comparison covers
23 earlier abstentions. These files are evidence for annotation review, not
verified labels for SFT or probability calibration.

Run `python scripts/validate_road_proposals.py` to verify token coverage and
read [`v1mini_flash_labeling_summary.json`](../outputs/road_head/v1mini_flash_labeling_summary.json).
