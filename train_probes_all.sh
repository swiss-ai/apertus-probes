# ./submit.sh cache --model llama-base --dataset_name mmlu_high_school 
# ./submit.sh cache --model llama-base --dataset_name mmlu_professional
# ./submit.sh cache --model llama-base --dataset_name sms_spam 
# ./submit.sh cache --model llama-base --dataset_name ARC-Easy
# ./submit.sh cache --model llama-base --dataset_name ARC-Challenge

# ./submit.sh cache --model apertus-base --dataset_name sujet_finance_yesno_5k
# ./submit.sh cache --model apertus-instruct --dataset_name sujet_finance_yesno_5k
# ./submit.sh cache --model llama-base --dataset_name sujet_finance_yesno_5k
# ./submit.sh cache --model llama-instruct --dataset_name sujet_finance_yesno_5k

# ./submit.sh postprocess --model apertus-base --dataset_name mmlu_high_school --overwrite
# ./submit.sh postprocess --model apertus-instruct --dataset_name mmlu_high_school --overwrite
# ./submit.sh postprocess --model llama-base --dataset_name mmlu_high_school --overwrite
# ./submit.sh postprocess --model llama-instruct --dataset_name mmlu_high_school --overwrite


# ./submit.sh --local postprocess --model llama-base --dataset_name ARC-Challenge
# ./submit.sh --local postprocess --model llama-base --dataset_name ARC-Easy
# ./submit.sh --local postprocess --model llama-base --dataset_name sms_spam
# ./submit.sh --local postprocess --model llama-base --dataset_name mmlu_professional
# ./submit.sh --local postprocess --model llama-base --dataset_name mmlu_high_school 
# ./submit.sh --local postprocess --model apertus-base --dataset_name ARC-Challenge
# ./submit.sh --local postprocess --model apertus-base --dataset_name ARC-Easy
# ./submit.sh --local postprocess --model apertus-base --dataset_name sms_spam
# ./submit.sh --local postprocess --model apertus-base --dataset_name mmlu_professional
# ./submit.sh --local postprocess --model apertus-base --dataset_name mmlu_high_school

# ./submit.sh run_probes --model apertus-base --datasets mmlu_high_school --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model apertus-base --datasets mmlu_professional  --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model apertus-base --datasets sms_spam --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model apertus-base --datasets ARC-Challenge --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model apertus-base --datasets ARC-Easy --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00

# ./submit.sh run_probes --model llama-base --datasets ARC-Challenge  --max-workers 60 --save-name logit_intercept --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model llama-base --datasets ARC-Easy  --max-workers 60 --save-name logit_intercept --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model llama-base --datasets mmlu_high_school  --max-workers 60 --save-name logit_intercept --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model llama-base --datasets mmlu_professional  --max-workers 60 --save-name logit_intercept --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model llama-base --datasets sms_spam  --max-workers 60 --save-name logit_intercept --alphas 0.01 0.02 0.05 --time 01:00:00
 
# ./submit.sh run_probes --model llama-instruct --datasets mmlu_professional  --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model llama-instruct --datasets sms_spam  --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model llama-instruct --datasets ARC-Challenge  --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model llama-instruct --datasets ARC-Easy  --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model llama-instruct --datasets mmlu_high_school  --max-workers 50 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model llama-instruct --datasets sujet_finance_yesno_5k  --max-workers 50 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00

# ./submit.sh run_probes --model apertus-instruct --datasets mmlu_high_school --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model apertus-instruct --datasets mmlu_professional  --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model apertus-instruct --datasets sms_spam --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model apertus-instruct --datasets ARC-Challenge --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model apertus-instruct --datasets ARC-Easy --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model apertus-instruct --datasets sujet_finance_yesno_5k --max-workers 50 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00


# ./submit.sh run_probes --model llama-base --datasets ARC-Challenge  --max-workers 50 --save-name logit_intercept --alphas 0.02 0.05 0.1 --time 01:00:00 --use-logit


# ./submit.sh run_probes --model llama-instruct --datasets ARC-Challenge ARC-Easy mmlu_high_school mmlu_professional sms_spam sujet_finance_yesno_5k --max-workers 40 --save-name linear_intercept --alphas 0.02 0.05 --time 09:00:00
# ./submit.sh run_probes --model apertus-instruct --datasets sms_spam sujet_finance_yesno_5k --max-workers 70 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 06:00:00
# ./submit.sh run_probes --model apertus-instruct --datasets mmlu_high_school mmlu_professional sujet_finance_yesno_5k --max-workers 70 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 06:00:00
# ./submit.sh run_probes --model apertus-instruct --datasets mmlu_high_school mmlu_professional ARC-Challenge ARC-Easy sujet_finance_yesno_5k --max-workers 40 --save-name linear_intercept --alphas 0.02 0.05 --time 09:00:00


# ./submit.sh run_probes --model apertus-base --datasets sujet_finance_yesno_5k --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model apertus-instruct --datasets sujet_finance_yesno_5k --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model llama-base --datasets sujet_finance_yesno_5k --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model llama-instruct --datasets sujet_finance_yesno_5k --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00

# ./submit.sh steer_multi --model llama-instruct --dataset_name "sujet_finance_yesno_5k" --regression-model-type linear --time 07:00:00
# ./submit.sh steer_multi --model llama-instruct --dataset_name "mmlu_high_school" --regression-model-type linear --time 07:00:00
# ./submit.sh steer_multi --model llama-instruct --dataset_name "mmlu_professional" --regression-model-type linear --time 07:00:00
# ./submit.sh steer_multi --model llama-instruct --dataset_name "sms_spam" --regression-model-type linear --time 07:00:00
# ./submit.sh steer_multi --model llama-instruct --dataset_name "ARC-Challenge" --regression-model-type linear --time 07:00:00
# ./submit.sh steer_multi --model llama-instruct --dataset_name "ARC-Easy" --regression-model-type linear --time 07:00:00

# ./submit.sh run_probes --model apertus-base --datasets mmlu_high_school mmlu_professional ARC-Challenge ARC-Easy sms_spam sujet_finance_yesno_5k --max-workers 70 --save-name both --alphas 0.01 0.02 --time 09:00:00
# ./submit.sh run_probes --model llama-base --datasets mmlu_high_school mmlu_professional ARC-Challenge ARC-Easy sms_spam sujet_finance_yesno_5k --max-workers 70 --save-name both --alphas 0.01 0.02 --time 09:00:00

# Use a mixture-trained probe to steer on a single dataset
# ./submit.sh steer_multi --model apertus-base --dataset_name mmlu_high_school --probe-dataset-name "mmlu_high_school+mmlu_professional+ARC-Challenge+ARC-Easy+sms_spam+sujet_finance_yesno_5k" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model apertus-base --dataset_name mmlu_professional --probe-dataset-name "mmlu_high_school+mmlu_professional+ARC-Challenge+ARC-Easy+sms_spam+sujet_finance_yesno_5k" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model apertus-base --dataset_name ARC-Challenge --probe-dataset-name "mmlu_high_school+mmlu_professional+ARC-Challenge+ARC-Easy+sms_spam+sujet_finance_yesno_5k" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model apertus-base --dataset_name ARC-Easy --probe-dataset-name "mmlu_high_school+mmlu_professional+ARC-Challenge+ARC-Easy+sms_spam+sujet_finance_yesno_5k" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model apertus-base --dataset_name sms_spam --probe-dataset-name "mmlu_high_school+mmlu_professional+ARC-Challenge+ARC-Easy+sms_spam+sujet_finance_yesno_5k" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model apertus-base --dataset_name sujet_finance_yesno_5k --probe-dataset-name "mmlu_high_school+mmlu_professional+ARC-Challenge+ARC-Easy+sms_spam+sujet_finance_yesno_5k" --regression-model-type logit --time 06:00:00


# ./submit.sh run_probes --model llama-base --datasets mmlu_high_school --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model llama-base --datasets mmlu_professional  --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model llama-base --datasets sms_spam --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model llama-base --datasets ARC-Challenge --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model llama-base --datasets ARC-Easy --max-workers 60 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00
# ./submit.sh run_probes --model llama-base --datasets sujet_finance_yesno_5k --max-workers 50 --save-name both --alphas 0.01 0.02 0.05 --time 01:00:00

# ./submit.sh steer_multi --model llama-instruct --dataset_name mmlu_high_school --probe-dataset-name "sms_spam" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model llama-instruct --dataset_name mmlu_professional --probe-dataset-name "sms_spam" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model llama-instruct --dataset_name ARC-Easy --probe-dataset-name "sms_spam" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model llama-instruct --dataset_name ARC-Challenge --probe-dataset-name "sms_spam" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model llama-instruct --dataset_name sujet_finance_yesno_5k --probe-dataset-name "sms_spam" --regression-model-type logit --time 06:00:00

# ./submit.sh steer_multi --model apertus-instruct --dataset_name sujet_finance_yesno_5k --probe-dataset-name "sujet_finance_yesno_5k" --regression-model-type linear --time 06:00:00

# ./submit.sh steer_multi --model apertus-instruct --dataset_name ARC-Challenge --probe-dataset-name "ARC-Challenge" --regression-model-type linear --time 06:00:00
# ./submit.sh steer_multi --model apertus-base --dataset_name ARC-Challenge --probe-dataset-name "ARC-Challenge" --regression-model-type linear --time 06:00:00
# ./submit.sh steer_multi --model llama-instruct --dataset_name ARC-Challenge --probe-dataset-name "ARC-Challenge" --regression-model-type linear --time 06:00:00
# ./submit.sh steer_multi --model llama-base --dataset_name ARC-Challenge --probe-dataset-name "ARC-Challenge" --regression-model-type linear --time 06:00:00
# ./submit.sh steer_multi --model apertus-instruct --dataset_name mmlu_high_school --probe-dataset-name "mmlu_high_school" --regression-model-type linear --time 06:00:00

# # mmlu professional
# ./submit.sh steer_multi --model llama-instruct --dataset_name mmlu_professional --probe-dataset-name "mmlu_professional" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model apertus-instruct --dataset_name mmlu_professional --probe-dataset-name "mmlu_professional" --regression-model-type logit --time 06:00:00

# # arc easy
# ./submit.sh steer_multi --model llama-instruct --dataset_name ARC-Easy --probe-dataset-name "ARC-Easy" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model apertus-instruct --dataset_name ARC-Easy --probe-dataset-name "ARC-Easy" --regression-model-type logit --time 06:00:00

# # sms spam
# ./submit.sh steer_multi --model llama-instruct --dataset_name sms_spam --probe-dataset-name "sms_spam" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model apertus-instruct --dataset_name sms_spam --probe-dataset-name "sms_spam" --regression-model-type logit --time 06:00:00


# # Missing steering dir for ARC-Challenge
# ./submit.sh steer_multi --model llama-instruct --dataset_name ARC-Challenge --probe-dataset-name "ARC-Challenge" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model apertus-instruct --dataset_name ARC-Challenge --probe-dataset-name "ARC-Challenge" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model apertus-base --dataset_name ARC-Challenge --probe-dataset-name "ARC-Challenge" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model llama-base --dataset_name ARC-Challenge --probe-dataset-name "ARC-Challenge" --regression-model-type logit --time 06:00:00
# # Missing steering dir for mmlu_high_school
# ./submit.sh steer_multi --model llama-instruct --dataset_name mmlu_high_school --probe-dataset-name "mmlu_high_school" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model apertus-instruct --dataset_name mmlu_high_school --probe-dataset-name "mmlu_high_school" --regression-model-type logit --time 06:00:00

# # Missing steering dir for sujet_finance_yesno_5k
# ./submit.sh steer_multi --model llama-instruct --dataset_name sujet_finance_yesno_5k --probe-dataset-name "sujet_finance_yesno_5k" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model apertus-instruct --dataset_name sujet_finance_yesno_5k --probe-dataset-name "sujet_finance_yesno_5k" --regression-model-type logit --time 06:00:00


# ./submit.sh steer_multi --model apertus-instruct --dataset_name mmlu_professional --probe-dataset-name "sms_spam" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model apertus-instruct --dataset_name ARC-Easy --probe-dataset-name "sms_spam" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model apertus-instruct --dataset_name ARC-Challenge --probe-dataset-name "sms_spam" --regression-model-type logit --time 06:00:00
# ./submit.sh steer_multi --model apertus-instruct --dataset_name sujet_finance_yesno_5k --probe-dataset-name "sms_spam" --regression-model-type logit --time 06:00:00
