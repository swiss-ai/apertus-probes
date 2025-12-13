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

# ./submit.sh run_probes --model apertus-base --datasets mmlu_high_school --no-transform-targets --max-workers 35 --save-name logit_intercept --alphas 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model apertus-base --datasets mmlu_professional --no-transform-targets --max-workers 35 --save-name logit_intercept --alphas 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model apertus-base --datasets sms_spam --no-transform-targets --max-workers 35 --save-name logit_intercept --alphas 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model apertus-base --datasets ARC-Challenge --no-transform-targets --max-workers 35 --save-name logit_intercept --alphas 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model apertus-base --datasets ARC-Easy --no-transform-targets --max-workers 35 --save-name logit_intercept --alphas 0.05 0.1 --time 01:00:00
 


# ./submit.sh run_probes --model apertus-instruct --datasets mmlu_high_school sms_spam --no-transform-targets --max-workers 30 --save-name logit
# ./submit.sh run_probes --model apertus-base --datasets mmlu_high_school sms_spam --no-transform-targets --max-workers 30 --save-name logit
# ./submit.sh run_probes --model apertus-base --datasets mmlu_high_school mmlu_professional sms_spam ARC-Challenge ARC-Easy --no-transform-targets --max-workers 5 --save-name logit --time 09:00:00 --alphas 0.05
# ./submit.sh run_probes --model apertus-instruct --datasets mmlu_high_school mmlu_professional sms_spam ARC-Challenge ARC-Easy --no-transform-targets --max-workers 5 --save-name logit --time 09:00:00 --alphas 0.05


# ./submit.sh run_probes --model llama-instruct --datasets mmlu_professional  --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model llama-instruct --datasets sms_spam  --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model llama-instruct --data;sets ARC-Challenge  --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model llama-instruct --datasets ARC-Easy  --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model llama-instruct --datasets mmlu_high_school  --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00

# ./submit.sh run_probes --model llama-base --datasets mmlu_professional  --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model llama-base --datasets sms_spam  --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model llama-base --data;sets ARC-Challenge  --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model llama-base --datasets ARC-Easy  --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model llama-base --datasets mmlu_high_school  --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00

# ./submit.sh run_probes --model llama-base --datasets ARC-Challenge  --max-workers 50 --save-name logit_intercept --alphas 0.02 0.05 0.1 --time 01:00:00 --use-logit
# ./submit.sh run_probes --model llama-base --datasets ARC-Easy  --max-workers 50 --save-name logit_intercept --alphas 0.02 0.05 0.1 --time 01:00:00 --use-logit
# ./submit.sh run_probes --model llama-base --datasets mmlu_high_school  --max-workers 50 --save-name logit_intercept --alphas 0.02 0.05 0.1 --time 01:00:00 --use-logit
# ./submit.sh run_probes --model llama-base --datasets mmlu_professional  --max-workers 50 --save-name logit_intercept --alphas 0.02 0.05 0.1 --time 01:00:00 --use-logit
# ./submit.sh run_probes --model llama-base --datasets sms_spam  --max-workers 50 --save-name logit_intercept --alphas 0.02 0.05 0.1 --time 01:00:00 --use-logit

# ./submit.sh run_probes --model llama-base --datasets ARC-Challenge  --max-workers 50 --save-name logit_intercept --alphas 0.02 0.05 0.1 --time 01:00:00 --use-logit


./submit.sh run_probes --model llama-instruct --datasets ARC-Challenge ARC-Easy mmlu_high_school mmlu_professional sms_spam sujet_finance_yesno_5k --max-workers 40 --save-name linear_intercept --alphas 0.02 0.05 --time 09:00:00
# ./submit.sh run_probes --model apertus-instruct --datasets sms_spam sujet_finance_yesno_5k --max-workers 70 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 06:00:00
# ./submit.sh run_probes --model apertus-instruct --datasets mmlu_high_school mmlu_professional sujet_finance_yesno_5k --max-workers 70 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 06:00:00
# ./submit.sh run_probes --model apertus-instruct --datasets mmlu_high_school mmlu_professional ARC-Challenge ARC-Easy sujet_finance_yesno_5k --max-workers 40 --save-name linear_intercept --alphas 0.02 0.05 --time 09:00:00


# ./submit.sh run_probes --model apertus-base --datasets sujet_finance_yesno_5k --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model apertus-instruct --datasets sujet_finance_yesno_5k --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model llama-base --datasets sujet_finance_yesno_5k --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00
# ./submit.sh run_probes --model llama-instruct --datasets sujet_finance_yesno_5k --max-workers 50 --save-name linear_intercept --alphas 0.02 0.05 0.1 --time 01:00:00

