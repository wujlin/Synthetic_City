#!/usr/bin/env bash
# PI experiments:
# 1) output examples
# 2) convergence (save_epochs + checkpoint evaluation)
# 3) MC stability
#
# 注意：本脚本不使用 set -e，报错后不会自动退出，便于排错。

cd "$(dirname "$0")/../.." || {
  echo ">>> [WARN] cannot cd to repo root."
}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dpl

DATA_ROOT="${DATA_ROOT:-/home/jinlin/data/geoexplicit_data/synthetic_city/data}"
OUT_ROOT="${OUT_ROOT:-outputs}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

JOINT5="${JOINT5:-$DATA_ROOT/us/processed/pums/puma_5var_joint_2023_5-Year/puma_5var_joint_wide.csv}"
PUMA_ZIP="${PUMA_ZIP:-$DATA_ROOT/us/raw/geo/tiger/cb_2020_us_puma20_500k.zip}"
US_HET_JSON="${US_HET_JSON:-outputs/_tmp_puma5var_us_smoke/heterogeneity_diagnostic.json}"
MI_HET_JSON="${MI_HET_JSON:-outputs/_tmp_puma5var_mi_smoke/heterogeneity_diagnostic.json}"

RUN_EXP2="$OUT_ROOT/_exp2_convergence_train_${TS}"
RUN_EXP23="$OUT_ROOT/_exp23_eval_${TS}"
FIG_DIR="Essay/figures"
mkdir -p "$RUN_EXP2" "$RUN_EXP23" "$FIG_DIR"

echo ">>> Input check"
ls -lh "$JOINT5" "$PUMA_ZIP" "$US_HET_JSON" "$MI_HET_JSON"

echo ">>> [Exp2-Train] pairwise model with save_epochs"
python -u tools/train_us_puma_5var_diffusion.py \
  --joint_wide_csv "$JOINT5" \
  --eval_mode leave_mi_out \
  --conditions pairwise \
  --timesteps 1000 \
  --epochs 10000 \
  --batch_size 512 \
  --hidden_dims 1024,1024 \
  --condition_injection concat \
  --save_epochs "100,200,500,1000,2000,3000,5000,7000,10000" \
  --save_final_model \
  --posthoc_ipf_policy marginal \
  --n_eval_joint_samples 128 \
  --device cuda \
  --seed 0 \
  --out_dir "$RUN_EXP2" 2>&1 | tee "$RUN_EXP2/run.log"
echo ">>> [Exp2-Train] RC=${PIPESTATUS[0]}"

CKPT_ROOT="$RUN_EXP2/checkpoints/pairwise/leave_mi_out"
FINAL_CKPT="$CKPT_ROOT/final.pt"

echo ">>> [Exp2-Eval] convergence curve from saved checkpoints"
python -u tools/essay/exp2_convergence_curve.py \
  --joint_wide_csv "$JOINT5" \
  --ckpt_root "$CKPT_ROOT" \
  --condition pairwise \
  --eval_mode leave_mi_out \
  --n_eval_joint_samples 128 \
  --posthoc_ipf \
  --ipf_iters 200 \
  --out_json "$RUN_EXP23/exp2_convergence.json" 2>&1 | tee "$RUN_EXP23/exp2_convergence.log"
echo ">>> [Exp2-Eval] RC=${PIPESTATUS[0]}"

echo ">>> [Exp3] MC stability"
python -u tools/essay/exp3_mc_stability.py \
  --joint_wide_csv "$JOINT5" \
  --checkpoint "$FINAL_CKPT" \
  --condition pairwise \
  --eval_mode leave_mi_out \
  --draw_counts "1,2,4,8,16,32,64,128,256,512" \
  --seeds "0,1,2,3,4,5,6,7,8,9" \
  --posthoc_ipf \
  --ipf_iters 200 \
  --out_json "$RUN_EXP23/exp3_mc_stability.json" 2>&1 | tee "$RUN_EXP23/exp3_mc.log"
echo ">>> [Exp3] RC=${PIPESTATUS[0]}"

echo ">>> [Exp1] output examples"
python -u tools/essay/exp1_output_examples.py \
  --joint_wide_csv "$JOINT5" \
  --checkpoint "$FINAL_CKPT" \
  --condition pairwise \
  --eval_mode leave_mi_out \
  --n_eval_joint_samples 128 \
  --posthoc_ipf \
  --ipf_iters 200 \
  --n_examples 4 \
  --out_pdf "$FIG_DIR/fig_03_output_examples.pdf" \
  --out_png "$FIG_DIR/fig_03_output_examples.png" \
  --out_json "$RUN_EXP23/exp1_output_examples.json" 2>&1 | tee "$RUN_EXP23/exp1_output_examples.log"
echo ">>> [Exp1] RC=${PIPESTATUS[0]}"

echo ">>> Build fig_verification (a,b,c)"
python -u tools/essay/plot_verification.py \
  --raw_metrics_json "$RUN_EXP2/metrics/internal_acs_holdout.json" \
  --convergence_json "$RUN_EXP23/exp2_convergence.json" \
  --mc_json "$RUN_EXP23/exp3_mc_stability.json" \
  --condition pairwise \
  --out_pdf "$FIG_DIR/fig_04_verification.pdf" \
  --out_png "$FIG_DIR/fig_04_verification.png" 2>&1 | tee "$RUN_EXP23/plot_verification.log"
echo ">>> [fig_verification] RC=${PIPESTATUS[0]}"

echo ">>> Split Figure 1"
python -u tools/essay/split_fig1_heterogeneity.py \
  --puma_zip "$PUMA_ZIP" \
  --us_heterogeneity_json "$US_HET_JSON" \
  --mi_heterogeneity_json "$MI_HET_JSON" \
  --out_map_pdf "$FIG_DIR/fig_01_map.pdf" \
  --out_stats_pdf "$FIG_DIR/fig_s02_heterogeneity_stats.pdf" \
  --out_map_png "$FIG_DIR/fig_01_map.png" \
  --out_stats_png "$FIG_DIR/fig_s02_heterogeneity_stats.png" 2>&1 | tee "$RUN_EXP23/split_fig1.log"
echo ">>> [split_fig1] RC=${PIPESTATUS[0]}"

echo ">>> Outputs"
ls -lh \
  "$FIG_DIR/fig_03_output_examples.pdf" \
  "$FIG_DIR/fig_04_verification.pdf" \
  "$FIG_DIR/fig_01_map.pdf" \
  "$FIG_DIR/fig_s02_heterogeneity_stats.pdf"

echo ">>> DONE"
echo "RUN_EXP2=$RUN_EXP2"
echo "RUN_EXP23=$RUN_EXP23"
