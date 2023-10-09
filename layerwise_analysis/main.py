import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from typing import *

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

from modeling import LlamaForCausalLM_GLA
from utils.prompter import Prompter


def new_lora_target_modules_based_on_layerwise_and_last_k_layers(
    lora_target_modules, train_layers
):
    new_lora_target_modules: List[str] = []

    for target_module in lora_target_modules:
        target_module = ").self_attn." + target_module
        for i, block in enumerate(train_layers):
            for j, layer in enumerate(block):
                target_module = f"{layer}" + target_module
                if i != len(train_layers) - 1 or j != len(block) - 1:
                    target_module = "|" + target_module
        target_module = "(" + target_module
        new_lora_target_modules.append(target_module)

    return new_lora_target_modules


def load_head_state_dict(model, path="head.pt"):
    
    head_state_dict = torch.load(path)
    
    new_head_state_dict = {}
    for key in head_state_dict.keys():
        orig_key = key
        new_head_state_dict.update({key.replace('_orig_mod.base_model.model.',''): head_state_dict[orig_key]})
    
    model.load_state_dict(new_head_state_dict, strict=False)


def prepare_gla_model(
    base_model, 
    load_8bit,
    gla_lora_weights,
    train_layers, 
    train_with_fix_head, 
    head_path
):
    model_class = LlamaForCausalLM_GLA
    additional_args = {
        "layerwise": True,
        "train_layers": train_layers,
    }
    model = model_class.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        **additional_args,
    )    
    model = PeftModel.from_pretrained(
        model,
        gla_lora_weights,
        torch_dtype=torch.float16,
    )
    
    if not train_with_fix_head:
        load_head_state_dict(model, head_path)
    
    return model


def prepare_lora_model(
    base_model, 
    load_8bit,
    lora_weights
):
    model_class = LlamaForCausalLM
    model = model_class.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )    
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )
    
    return model


def prepare_baseline_model(
    base_model, 
    load_8bit,
):
    model_class = LlamaForCausalLM
    
    model = model_class.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    return model


def layerwise_analyze(
    # new params
    train_layers: Union[None, str, List[List[int]]] = None,
    train_with_fix_head: bool = False,
    head_path: str = "",
    # model/data params
    base_model: str = "decapoda-research/llama-7b-hf",
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    lora_weights: str = "tloen/alpaca-lora-7b",
    gla_lora_weights: str = "",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # others
    load_8bit: bool = False,
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    # === Training Mode ===
    # - full network
    
    lora_target_modules = (
        new_lora_target_modules_based_on_layerwise_and_last_k_layers(
            lora_target_modules, train_layers
        )
    )

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            "====================\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_layers: {train_layers}\n"
            "====================\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = False

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None
    
    gla_model = prepare_gla_model(
        base_model, 
        load_8bit, 
        gla_lora_weights,
        train_layers,
        train_with_fix_head,
        head_path
    )
    
    lora_model = prepare_lora_model(
        base_model, 
        load_8bit, 
        lora_weights,
    )
    
    base_model = prepare_baseline_model(
        base_model,
        load_8bit
    )
    
    test_field(
        gla_model,
        lora_model,
        base_model,
        train_layers,
        train_data,
        val_data
    )



def test_field(
    gla_model,
    lora_model,
    base_model,
    train_layers,
    train_data,
    val_data
):
    pass


if __name__ == "__main__":
    fire.Fire(layerwise_analyze)