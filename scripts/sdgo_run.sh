################### Safety Eval ###################

## llama3.1 vanilla
python sdgo_infer_and_eval.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --datasets sdgo_autodan_test sdgo_renellm_test sdgo_deepinception_test sdgo_codeattack_test sdgo_test_all \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

python sdgo_safety_gap.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --generation_datasets sdgo_test_all \
    --judgment_datasets sdgo_test_all_dis \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

## llama3.1 self-reminder
python sdgo_infer_and_eval.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --datasets sdgo_self_reminder_autodan_test sdgo_self_reminder_renellm_test sdgo_self_reminder_deepinception_test sdgo_self_reminder_codeattack_test sdgo_self_reminder_all_test \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

python sdgo_safety_gap.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --generation_datasets sdgo_self_reminder_all_test \
    --judgment_datasets sdgo_self_reminder_all_test_dis \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

## llama3.1 icd
python sdgo_infer_and_eval.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --datasets sdgo_icd_autodan_test sdgo_icd_renellm_test sdgo_icd_deepinception_test sdgo_icd_codeattack_test sdgo_icd_all_test \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

python sdgo_safety_gap.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --generation_datasets sdgo_icd_all_test \
    --judgment_datasets sdgo_icd_all_test_dis \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

## llama3.1 sft
python sdgo_infer_and_eval.py \
    --model_path /path/to/sft/Llama-3.1-8B-Instruct \
    --datasets sdgo_autodan_test sdgo_renellm_test sdgo_deepinception_test sdgo_codeattack_test sdgo_test_all \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

python sdgo_safety_gap.py \
    --model_path /path/to/sft/Llama-3.1-8B-Instruct \
    --generation_datasets sdgo_test_all \
    --judgment_datasets sdgo_test_all_dis \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

## llama3.1 sdgo
python sdgo_infer_and_eval.py \
    --model_path /path/to/sdgo/Llama-3.1-8B-Instruct \
    --datasets sdgo_autodan_test sdgo_renellm_test sdgo_deepinception_test sdgo_codeattack_test sdgo_test_all \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

python sdgo_safety_gap.py \
    --model_path /path/to/sdgo/Llama-3.1-8B-Instruct \
    --generation_datasets sdgo_test_all \
    --judgment_datasets sdgo_test_all_dis \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192


################### Helpful Eval ###################

## llama3.1 vanilla
python sdgo_helpful_eval.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --datasets gsm8k_100 mmlu_100 alpaca_eval_100 xstest_prompts_safe \
    --original_files data/gsm8k_100.json data/mmlu_100.json data/alpaca_eval_100.json data/xstest_prompts_safe.json \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192


## llama3.1 self-reminder
python sdgo_helpful_eval.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --datasets sdgo_self_reminder_gsm8k_100 sdgo_self_reminder_mmlu_100 sdgo_self_reminder_alpaca_eval_100 xstest_prompts_safe_self_reminder \
    --original_files data/sdgo_self_reminder_gsm8k_100.json data/sdgo_self_reminder_mmlu_100.json data/sdgo_self_reminder_alpaca_eval_100.json data/xstest_prompts_safe_self_reminder.json \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192


## llama3.1 icd
python sdgo_helpful_eval.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --datasets sdgo_icd_gsm8k_100 sdgo_icd_mmlu_100 sdgo_icd_alpaca_eval_100 xstest_prompts_safe_icd \
    --original_files data/sdgo_icd_gsm8k_100.json data/sdgo_icd_mmlu_100.json data/sdgo_icd_alpaca_eval_100.json data/xstest_prompts_safe_icd.json \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

## llama3.1 sft
python sdgo_helpful_eval.py \
    --model_path /path/to/sft/Llama-3.1-8B-Instruct \
    --datasets gsm8k_100 mmlu_100 alpaca_eval_100 xstest_prompts_safe \
    --original_files data/gsm8k_100.json data/mmlu_100.json data/alpaca_eval_100.json data/xstest_prompts_safe.json \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

## llama3.1 sdgo
python sdgo_helpful_eval.py \
    --model_path /path/to/sdgo/Llama-3.1-8B-Instruct \
    --datasets gsm8k_100 mmlu_100 alpaca_eval_100 xstest_prompts_safe \
    --original_files data/gsm8k_100.json data/mmlu_100.json data/alpaca_eval_100.json data/xstest_prompts_safe.json \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

################### OverDenfense Eval ###################

## llama3.1 vanilla
python sdgo_helpful_eval.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --datasets xstest_prompts_safe \
    --original_files data/xstest_prompts_safe.json \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192


## llama3.1 self-reminder
python sdgo_helpful_eval.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --datasets xstest_prompts_safe_self_reminder \
    --original_files data/xstest_prompts_safe_self_reminder.json \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192


## llama3.1 icd
python sdgo_helpful_eval.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --datasets xstest_prompts_safe_icd \
    --original_files data/xstest_prompts_safe_icd.json \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

## llama3.1 sft
python sdgo_helpful_eval.py \
    --model_path /path/to/sft/Llama-3.1-8B-Instruct \
    --datasets xstest_prompts_safe \
    --original_files data/xstest_prompts_safe.json \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

## llama3.1 sdgo
python sdgo_helpful_eval.py \
    --model_path /path/to/sdgo/Llama-3.1-8B-Instruct \
    --datasets xstest_prompts_safe \
    --original_files data/xstest_prompts_safe.json \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192


################### OOD Attack Eval ###################

## llama3.1 vanilla
python sdgo_infer_and_eval.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --datasets pair_qwen-2.5 gptfuzzer_advbench renellm_maliciousinstruct codeattack_maliciousinstruct \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

python sdgo_safety_gap.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --generation_datasets sdgo_ood \
    --judgment_datasets sdgo_ood_dis \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

## llama3.1 self-reminder
python sdgo_infer_and_eval.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --datasets self_reminder_pair_qwen-2.5 self_reminder_gptfuzzer_advbench self_reminder_renellm_maliciousinstruct self_reminder_codeattack_maliciousinstruct \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

python sdgo_safety_gap.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --generation_datasets self_reminder_ood \
    --judgment_datasets self_reminder_ood_dis \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

## llama3.1 icd
python sdgo_infer_and_eval.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --datasets icd_pair_qwen-2.5 icd_gptfuzzer_advbench icd_renellm_maliciousinstruct icd_codeattack_maliciousinstruct \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

python sdgo_safety_gap.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --generation_datasets icd_ood \
    --judgment_datasets icd_ood_dis \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

## llama3.1 sft
python sdgo_infer_and_eval.py \
    --model_path /path/to/sft/Llama-3.1-8B-Instruct \
    --datasets pair_qwen-2.5 gptfuzzer_advbench renellm_maliciousinstruct codeattack_maliciousinstruct \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192

python sdgo_safety_gap.py \
    --model_path /path/to/sft/Llama-3.1-8B-Instruct \
    --generation_datasets sdgo_ood \
    --judgment_datasets sdgo_ood_dis \
    --template llama3 \
    --temperature 0 \
    --max_new_tokens 8192
