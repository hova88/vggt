# NuScenes CAM_FRONT road-state experiment

The five labels are **external human annotations**: 0 elevated_up, 1 elevated_down,
2 main_road, 3 side_road, 4 intersection. NuScenes does not supply them. Each JSONL row
must have `sample_token` and exactly one of `label` (integer 0–4) or `soft_label`
(five nonnegative probabilities summing to 1). `scene_token` is optional and validated.
The label describes the **last** frame in the sequence.

The commands below use the `uniad2.0` environment. The current config automatically
selects `v1.0-mini` when available and keeps train/validation scenes disjoint.

```bash
conda run -n uniad2.0 python scripts/build_nuscenes_road_manifest.py \
  --nuscenes-root /data/nuscenes \
  --output /data/nuscenes/road_labels_template.jsonl --max-samples 500
```

Add real labels to the JSONL rows and save the finished file as
`/data/nuscenes/road_labels.jsonl`. Do not give a label to an uncertain sample unless
you have an actual soft annotation. The generator never writes labels.

```bash
# One complete shape/gradient/attention check with fabricated labels and random VGGT.
# This checks code only and does not create a training checkpoint.
conda run -n uniad2.0 python scripts/train_road_head.py \
  --config configs/road_head_nuscenes_small.yaml \
  --dummy-labels --random-vggt --smoke --visualize-smoke

# After real annotation exists, test overfitting 16–32 examples.
conda run -n uniad2.0 python scripts/train_road_head.py \
  --config configs/road_head_nuscenes_small.yaml --tiny-overfit

# Scene-split small experiment, then evaluation and validation-only calibration.
conda run -n uniad2.0 python scripts/train_road_head.py \
  --config configs/road_head_nuscenes_small.yaml
conda run -n uniad2.0 python scripts/eval_road_head.py \
  --config configs/road_head_nuscenes_small.yaml --checkpoint outputs/road_head/best.pt
conda run -n uniad2.0 python scripts/calibrate_road_head.py \
  --config configs/road_head_nuscenes_small.yaml --checkpoint outputs/road_head/best.pt
conda run -n uniad2.0 python scripts/infer_road_head.py \
  --config configs/road_head_nuscenes_small.yaml --checkpoint outputs/road_head/best.pt \
  --sample-token YOUR_TARGET_SAMPLE_TOKEN
```

For four explicit ordered images, replace `--sample-token` with
`--images oldest.jpg older.jpg newer.jpg newest.jpg`. Inference loads
`temperature.json` automatically. `road_prob` is the calibrated posterior; with
`hmm.use_prior_correction: true`, it also returns the normalized posterior / training
prior score as `hmm_observation`. This is a pseudo emission score, not a learned HMM.

By default `model.pretrained: true` loads `facebook/VGGT-1B` from Hugging Face. A local
VGGT state dict can instead be set with `model.checkpoint`. The current small config
uses VGGT's original 518-width preprocess followed by a common bicubic reduction to
280 pixels, with the height rounded to a multiple of 14, to fit T=4 on a 16 GB GPU.
Set `data.image_width: 518` for original resolution if memory permits. No horizontal
flip or random crop is applied. The head-only mode freezes the aggregator and runs it
without gradients. `training.finetune_mode: last_blocks` unfreezes the final
`frame_blocks` and `global_blocks`, leaving the DINO patch embed frozen.

`best.pt` is selected by validation NLL. Metrics and visualizations are written under
`outputs/road_head`. A validation set with fewer than 50 sequences triggers a
calibration warning. Tiny-overfit uses its training subset for validation by design;
do not interpret those metrics as generalization.

## Model-assisted annotation archive

The complete `v1.0-mini` CAM_FRONT **sample-keyframe** archive has 404 targets in
`outputs/road_head/road_labels_v1mini_all_samples_template.jsonl`. The paired
`road_labels_v1mini_all_samples_qwen_flash_proposals.jsonl` contains model proposals,
including 30 scene-start targets whose missing history was repeated. Run
`python scripts/validate_road_proposals.py` to check coverage and write the summary.
These JSONL rows intentionally contain `proposed_label`, not `label`: the `View Image`
audit found false `intersection` predictions, so the file is not ground truth and
must not be used as the default supervised annotation manifest. Non-keyframe
`sweeps/CAM_FRONT` images are outside this sample-token archive.
