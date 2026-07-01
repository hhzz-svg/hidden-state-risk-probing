@echo off
REM Windows 一键运行第二次实验的核心 probe 部分
REM 请把本文件放在项目根目录运行

python scripts\21_train_layerwise_probe.py --hidden_file outputs\qwen05b\experiment1\experiment1_hidden_extraction\hidden_states_pilot.pt --data_file data\experiment1\prompts_pilot.jsonl --out_dir outputs\qwen05b\experiment2\experiment2_layerwise_probe

echo.
echo Layer-wise probe finished.
echo Check outputs\qwen05b\experiment2\experiment2_layerwise_probe\layer_auc_curve.png and outputs\qwen05b\experiment2\experiment2_layerwise_probe\layerwise_probe_metrics.csv
pause
