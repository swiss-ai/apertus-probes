from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import numpy as np
import requests
from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from datasets import concatenate_datasets
from src.utils import *


def generate_text_variants(word: str, remove_lower: bool = False) -> list:
    transformations = [word.lower(), word.upper(), word.capitalize()]
    if remove_lower:
        del transformations[0]
    spaces = [" ", ""]  # ' ', , '\n'
    return [
        f"{prefix}{variant}" for variant in transformations for prefix in spaces
    ]  # for suffix in spaces {suffix}


dataset_info = {
    "sms_spam": {
        "CLASSES": ["ham", "spam"],
        "CLASS_LABEL_TO_INDEX": {"ham": 0, "spam": 1},
        "CLASS_INDEX_TO_LABEL": {0: "ham", 1: "spam"},
        "CLASS_LABEL_SEMANTIC": {
            "ham": generate_text_variants("ham"),
            "spam": generate_text_variants("spam"),
        },
        "DATASET_NAME_HF": "sms_spam",
        "MAX_LENGTH": 350,
        "MAX_NEW_TOKENS": 100,
        "LABEL_NAME": "label",
        "NR_TRAINING_SAMPLES": 3000,
    },

    "mmlu_pro_natural_science": {
        "CLASSES": [chr(i) for i in range(65, 91)],
        "CLASS_LABEL_TO_INDEX": {chr(i): i - 65 for i in range(65, 91)},
        "CLASS_INDEX_TO_LABEL": {i - 65: chr(i) for i in range(65, 91)},
        "CLASS_LABEL_SEMANTIC": {
            chr(i): generate_text_variants(chr(i), remove_lower=True)
            for i in range(65, 91)
        },
        "SUB_TASKS": ["computer science", "physics", "math", "engineering"],
        "DATASET_NAME_HF": "mmlu-pro",
        "MAX_LENGTH": 150,
        "MAX_NEW_TOKENS": 100,
        "LABEL_NAME": "answer",
        "NR_TRAINING_SAMPLES": 3000,
        "NR_REF_SAMPLES": 250,
        "NR_TEST_SAMPLES": 250,
    },
    "mmlu_high_school": {
        "CLASSES": [chr(i) for i in range(65, 69)],
        "CLASS_LABEL_TO_INDEX": {chr(i): i - 65 for i in range(65, 69)},
        "CLASS_INDEX_TO_LABEL": {i - 65: chr(i) for i in range(65, 69)},
        "CLASS_LABEL_SEMANTIC": {
            # chr(i): [f"{chr(i)}", f"{chr(i)} ", f" {chr(i)}"] for i in range(65, 69)
            chr(i): generate_text_variants(chr(i), remove_lower=True)
            for i in range(65, 69)
        },
        "SUB_TASKS": [
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_european_history",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_mathematics",
            "high_school_microeconomics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_statistics",
            "high_school_us_history",
            "high_school_world_history",
        ],
        "DATASET_NAME_HF": "mmlu",
        "MAX_LENGTH": 250,
        "MAX_NEW_TOKENS": 100,
        "LABEL_NAME": "answer",
        "NR_TRAINING_SAMPLES": 3000,
        "NR_REF_SAMPLES": 210,
        "NR_TEST_SAMPLES": 210,
    },
    "mmlu_professional": {
        "CLASSES": [chr(i) for i in range(65, 69)],
        "CLASS_LABEL_TO_INDEX": {chr(i): i - 65 for i in range(65, 69)},
        "CLASS_INDEX_TO_LABEL": {i - 65: chr(i) for i in range(65, 69)},
        "CLASS_LABEL_SEMANTIC": {
            chr(i): generate_text_variants(chr(i), remove_lower=True)
            for i in range(65, 69)
        },
        "DATASET_NAME_HF": "mmlu",
        "MAX_LENGTH": 250,
        "MAX_NEW_TOKENS": 100,
        "LABEL_NAME": "answer",
        "SUB_TASKS": [
            "professional_accounting",
            "professional_law",
            "professional_medicine",
            "professional_psychology",
        ],
        "NR_TRAINING_SAMPLES": 2601,
        "NR_REF_SAMPLES": 210,
        "NR_TEST_SAMPLES": 210,
    },
    "ARC-Easy": {
        "CLASSES": [chr(i) for i in range(65, 69)],
        "CLASS_LABEL_TO_INDEX": {chr(i): i - 65 for i in range(65, 69)},
        "CLASS_INDEX_TO_LABEL": {i - 65: chr(i) for i in range(65, 69)},
        "CLASS_LABEL_SEMANTIC": {
            chr(i): generate_text_variants(chr(i), remove_lower=True)
            for i in range(65, 69)
        },
        "DATASET_NAME_HF": "ai2_arc",
        "MAX_LENGTH": 250,
        "MAX_NEW_TOKENS": 100,
        "NR_TRAINING_SAMPLES": 5000,
        "NR_REF_SAMPLES": 250,
        "NR_TEST_SAMPLES": 250,
    }
}


@dataclass
class TaskConfig:
    """Assumes downloaded in cache dir."""

    token: str
    cache_dir: str
    dataset_name: str
    device: Optional[str]
    nr_devices: int
    model_name: str
    flexible_match: bool
    batch_size: int = 1
    model_kwargs: Dict[str, Any] = dict()
    tokenizer_kwargs: Dict[str, Any] = dict()
    nr_samples: Optional[int] = None
    nr_test_samples: Optional[int] = None
    nr_ref_samples: Optional[int] = None

    def __post_init__(self):
        print(f"[INFO] Initalising {self.dataset_name}")
        self.dataset_info = dataset_info[self.dataset_name]
        if self.nr_samples is None:
            self.nr_samples = self.dataset_info["NR_TRAINING_SAMPLES"]
        if self.nr_test_samples is None:
            self.nr_test_samples = self.dataset_info["NR_TEST_SAMPLES"]
        if self.nr_ref_samples is None:
            self.nr_ref_samples = self.dataset_info["NR_REF_SAMPLES"]

        self.dataset_name = self.dataset_name
        self.dataset_name_hf = self.dataset_info["DATASET_NAME_HF"]
        self.model_kwargs = self.model_kwargs or {
            # "token": self.token, # this way you don't have to provide it in the command line
            "cache_dir": self.cache_dir,
            #"device_map": "auto",
            "device_map": {"": self.device} if self.device is not None else "auto",
            "offload_folder": f"{self.cache_dir}/offload/",
            # "pad_token_id": tokenizer.eos_token_id,
            # "torch_dtype": torch.float16,
            # "device": self.device,s
            # "nr_devices": self.nr_devices,
        }
        self.tokenizer_kwargs = self.tokenizer_kwargs or {
            "padding": False,
            "truncation": False,
            "return_tensors": "pt",
        }


class ModelHandler:
    def __init__(self, config: TaskConfig):
        print(f"[INFO] Loading model and tokenizer...")
        self.config = config
        self.tokenizer = self._load_tokenizer()
        self.model : PreTrainedModel  = self._load_model()
        self.nr_layers = len(self.model.model.layers)
        self.tokenizer_kwargs = self.config.tokenizer_kwargs
        print(f"[INFO] \n... Done.")

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, **self.config.model_kwargs
        )
        return tokenizer

    def _load_model(self):
        
        self.config.model_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        print("[DEBUG] Model kwargs:", self.config.model_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, **self.config.model_kwargs
        )
        model.eval()
        return model


class DatasetHandler:
    """
    DatasetHandler is a class responsible for managing and processing datasets for various tasks. 
    It loads datasets, filters them based on configuration, and prepares samples, labels, and prompts 
    for training, testing, and reference purposes.

    Attributes:
        config (TaskConfig): Configuration object containing dataset and task-specific settings.
        tokenizer: Tokenizer object used for processing text data.
        device: Device (e.g., CPU or GPU) specified in the configuration.
        dataset_info (dict): Information about the dataset, including label mappings and task-specific details.
        ds: The loaded dataset object.
        ds_samples: Samples from the dataset for training or evaluation.
        y_true_labels: Ground truth labels in human-readable format for the training/evaluation samples.
        y_true: Ground truth labels in numerical format for the training/evaluation samples.
        prompts: Generated prompts for the training/evaluation samples.
        ds_samples_test: Test samples from the dataset (if specified in the configuration).
        y_true_labels_test: Ground truth labels in human-readable format for the test samples.
        y_true_test: Ground truth labels in numerical format for the test samples.
        prompts_test: Generated prompts for the test samples.
        ds_samples_ref: Reference samples from the dataset (if specified in the configuration).
        y_true_labels_ref: Ground truth labels in human-readable format for the reference samples.
        y_true_ref: Ground truth labels in numerical format for the reference samples.
        prompts_ref: Generated prompts for the reference samples.

    Methods:
        __init__(config, tokenizer):
            Initializes the DatasetHandler, loads the dataset, and prepares samples, labels, and prompts.

        _load_dataset():
            Loads the dataset from disk and applies filtering based on the configuration.

        _get_samples(end_idx, start_idx=0):
            Retrieves a subset of samples from the dataset based on the specified range.

        _get_y_true_labels(samples):
            Extracts ground truth labels in both human-readable and numerical formats from the given samples.

        _update_dataset_info_with_token_ids():
            Updates the dataset information with token IDs for valid ground truth labels.

        _get_prompts(samples):
            Generates prompts for the given samples based on the dataset and task configuration.

        __len__():
            Returns the number of training/evaluation samples.
    """
    def __init__(self, config: TaskConfig, tokenizer):
        print(f"[INFO] Loading dataset...")
        self.config = config
        self.tokenizer = tokenizer
        self.device = config.device
        self.dataset_info = self.config.dataset_info
        self._update_dataset_info_with_token_ids() 

        self.ds : Union[Dataset, DatasetDict] = self._load_dataset() # Loads HF dataset into a Dataset object
        self.ds_samples : Dataset = self._get_samples(end_idx=self.config.nr_samples, start_idx=0)
        self.y_true_labels, self.y_true = self._get_y_true_labels(self.ds_samples)
        self.prompts : List[str] = self._get_prompts(self.ds_samples)
        


        print(f"[INFO] ... Done.")
        print(f"[DEBUG] Loaded HF dataset: {self.config.dataset_name_hf}")
        print(f"[DEBUG] Task config name: {self.config.dataset_name}")
        print(f"[DEBUG] Dataset size after filtering: {len(self.prompts)}")


    def _load_dataset(self) -> Union[Dataset, DatasetDict]:
        ds = load_from_disk(f"{self.config.cache_dir}{self.config.dataset_name_hf}.hf")
        if self.config.dataset_name_hf == "mmlu_pro":
            ds = ds.filter(lambda x: x["category"] in self.dataset_info["SUB_TASKS"])
        elif self.config.dataset_name_hf == "mmlu":
            if self.config.dataset_name == "mmlu_professional":
                # Filter all splits
                subsets = []
                for split in ["test", "validation", "dev"]:
                    if split in ds:
                        filtered = ds[split].filter(lambda x: x["subject"].startswith("professional_"))
                        subsets.append(filtered)
                ds = concatenate_datasets(subsets)
            else:
                ds = ds.filter(lambda x: x["subject"] in self.dataset_info["SUB_TASKS"])

        return ds

    def _get_samples(self, end_idx: int, start_idx: int = 0) -> Dataset:

        if self.config.dataset_name == "mmlu_professional":
            return self.ds.select(range(start_idx, end_idx))
        elif self.config.dataset_name_hf in ["mmlu_pro", "imdb"]:  # == "mmlu_pro":
            return self.ds.select(range(start_idx, end_idx))
        elif self.config.dataset_name_hf in ["mmlu"]:  # , "imdb"]:
            return self.ds["test"].select(range(start_idx, end_idx))

    def _get_y_true_labels(self, samples : Dataset) -> Tuple[List[str], List[int]]:
        return [
            self.dataset_info["CLASS_INDEX_TO_LABEL"][
                s[self.dataset_info["LABEL_NAME"]]
            ]
            for s in samples
        ], [s[self.dataset_info["LABEL_NAME"]] for s in samples]
        

    def _update_dataset_info_with_token_ids(self):
        """Updates `VALID_GROUND_TRUTH_TOKEN_IDS` with the last token ID for each semantic label variant."""
        semantic_variants = self.dataset_info.get("CLASS_LABEL_SEMANTIC", {})
        self.dataset_info["VALID_GROUND_TRUTH_TOKEN_IDS"] = {}

        for label, synonyms in semantic_variants.items():
            self.dataset_info["VALID_GROUND_TRUTH_TOKEN_IDS"][label] = {}
            for word in [label] + synonyms:
                tokenized_word = self.tokenizer([word], return_tensors="pt").input_ids[
                    0
                ]
                token_id = tokenized_word[-1].item()
                self.dataset_info["VALID_GROUND_TRUTH_TOKEN_IDS"][label][
                    word
                ] = token_id
                # print(f"{word}: {token_id}")

        print(
            f"[INFO] VALID_GROUND_TRUTH_TOKEN_IDS updated: {self.dataset_info['VALID_GROUND_TRUTH_TOKEN_IDS']}"
        )

    def _get_prompts(self, samples: Dataset) -> List[str]:
        prompts = []

        if "mmlu" in self.config.dataset_name:
            options = samples["choices"]

            for i, option in enumerate(options):
                formatted_options = "\n".join(
                    [
                        f"{self.dataset_info['CLASS_INDEX_TO_LABEL'][i]}. {choice}"
                        for i, choice in enumerate(option)
                    ]
                )
                answer_options = ", ".join(
                    [
                        f"{self.dataset_info['CLASS_INDEX_TO_LABEL'][i]}"
                        for i in range(len(option))
                    ]
                )
                prompt = f"""Question: {samples['question'][i]}
                \nOptions:\n{formatted_options}
                \nPlease select the correct answer. 
                Only return one letter {answer_options}. 
                Answer:\n """
                prompts.append(prompt)

        elif self.config.dataset_name == "sms_spam":
            sms_inputs = [s["sms"] + " " for s in samples]
            for sms in sms_inputs:
                prompt = f"""
                This SMS (text message): "{sms.strip()}" is classified as either spam or ham.
                \nPlease evaluate the content of the SMS and select the correct classification.
                \nOnly return one word: "ham" or "spam".
                Answer:\n """
                prompts.append(prompt)

        elif self.config.dataset_name == "imdb":
            texts = [s["text"] + " " for s in samples]
            for text in texts:
                prompt = f"""
                Review: "{text.strip()}"
                \nPlease classify this review as either "positive or "negative" sentiment.
                \nOnly return one word: "positive" for a positive review or "negative" for a negative review.
                Answer:\n """
                prompts.append(prompt)

        return prompts

    def __len__(self):
        return len(self.ds_samples)
