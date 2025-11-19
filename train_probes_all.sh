./submit.sh train_probes --model apertus --dataset_name mmlu_high_school --save_name leave --transform_targets "False"
./submit.sh train_probes --model apertus --dataset_name mmlu_professional --save_name leave --transform_targets "False"
./submit.sh train_probes --model apertus --dataset_name sms_spam --save_name leave --transform_targets "False"
./submit.sh train_probes --model apertus --dataset_name ARC-Easy --save_name leave --transform_targets "False"
./submit.sh train_probes --model apertus --dataset_name ARC-Challenge --save_name leave --transform_targets "False"

./submit.sh train_probes --model apertus --dataset_name mmlu_high_school --save_name "transform" --transform_targets "True"
./submit.sh train_probes --model apertus --dataset_name mmlu_professional --save_name "transform" --transform_targets "True"
./submit.sh train_probes --model apertus --dataset_name sms_spam --save_name "transform" --transform_targets "True"
./submit.sh train_probes --model apertus --dataset_name ARC-Easy --save_name "transform" --transform_targets "True"
./submit.sh train_probes --model apertus --dataset_name ARC-Challenge --save_name "transform" --transform_targets "True"

