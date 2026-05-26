# Overall
pred_dir="${1:-/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/results/llavarad_finetune}"
prefix="${2:-finetune1}"
dataset="${3:-MIMIC-CXR}"

python /home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/llava/eval/rrg_eval/run.py $pred_dir/merged_preds.jsonl --run_name ${prefix}overall --output_dir $pred_dir


# stratify the results based on demographic information, it should also save green score
python /home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/scripts/stratify_results.py --pred_dir $pred_dir --dataset $dataset

# RACE (Only present in MIMIC-CXR)
dataset_lower=$(printf '%s' "$dataset" | tr '[:upper:]' '[:lower:]')
if [ "$dataset_lower" = "mimic-cxr" ]; then
    [ -f "$pred_dir/race_hispanic.jsonl" ] && python /home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/llava/eval/rrg_eval/run.py $pred_dir/race_hispanic.jsonl --run_name ${prefix}race_hispanic
    [ -f "$pred_dir/race_asian.jsonl" ] && python /home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/llava/eval/rrg_eval/run.py $pred_dir/race_asian.jsonl --run_name ${prefix}race_asian
    [ -f "$pred_dir/race_black.jsonl" ] && python /home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/llava/eval/rrg_eval/run.py $pred_dir/race_black.jsonl --run_name ${prefix}race_black
    [ -f "$pred_dir/race_white.jsonl" ] && python /home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/llava/eval/rrg_eval/run.py $pred_dir/race_white.jsonl --run_name ${prefix}race_white
    [ -f "$pred_dir/race_other.jsonl" ] && python /home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/llava/eval/rrg_eval/run.py $pred_dir/race_other.jsonl --run_name ${prefix}race_other
fi

# Age
[ -f "$pred_dir/age_0_44.jsonl" ] && python /home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/llava/eval/rrg_eval/run.py $pred_dir/age_0_44.jsonl --run_name ${prefix}age_0_44
[ -f "$pred_dir/age_44_65.jsonl" ] && python /home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/llava/eval/rrg_eval/run.py $pred_dir/age_44_65.jsonl --run_name ${prefix}age_44_65
[ -f "$pred_dir/age_65_inf.jsonl" ] && python /home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/llava/eval/rrg_eval/run.py $pred_dir/age_65_inf.jsonl --run_name ${prefix}age_65_inf

# Gender
[ -f "$pred_dir/gender_f.jsonl" ] && python /home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/llava/eval/rrg_eval/run.py $pred_dir/gender_f.jsonl --run_name ${prefix}gender_f
[ -f "$pred_dir/gender_m.jsonl" ] && python /home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/llava/eval/rrg_eval/run.py $pred_dir/gender_m.jsonl --run_name ${prefix}gender_m