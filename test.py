# import libraries
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import argparse
import os
import random


# argparse
def arg():
    args = argparse.ArgumentParser()

    args.add_argument("--n",type=int,default=0)
    # do_sample=True,
                    # max_new_tokens=50,
                    # temperature=0.7,
                    # top_k=50, 
                    # top_p
    args.add_argument("--do_sample", 
                      action=argparse.BooleanOptionalAction,
                      default=False)

    args.add_argument("--max_new_tokens",type=int,default=50)
    args.add_argument("--temperature",type=float,default=0.7)
    args.add_argument("--top_k",type=int,default=50)
    args.add_argument("--top_p",type=float,default=0.95)

    return args.parse_args()


# import build dataset
def import_data(text=True):
    print("S1")
    if text:
        dataset = load_dataset("knkarthick/dialogsum")
        indices = random.sample(range(200),31)
        dataset_train = dataset["test"].select(indices[:-1])
        dataset_test = dataset["test"].select([indices[-1]])
        print("data loaded.")
    return dataset_train, dataset_test


def build_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,local_files_only=True)
    print("model loaded.")
    return model, tokenizer




# n shots test
# input: n=? shots examples; 
        # template
#        parameters:
            # max_new_tokens
            # do_sample
            # temperature
            # top_k
            # top_p

# conduct test:
    # try 0 shot, with prompt template
    # try 1 shot, with prompt template
    # try 3+ shots, with prompt template
def test_n_shots(tokenizer, 
                model, 
                n, 
                input_train_data,
                input_test_data,
                *,
                do_sample,
                max_new_tokens,
                temperature,
                top_k, 
                top_p):
    # prepare prompt template
    if n == 0:
        prompt_template = f"Please summarize the following dialogue:\n\n{input_test_data['dialogue']}:"
    else:
        prompt_template = f"""
        Follow the way of these examples to give a summary for the given dialogue:\n"""

        for index in range(n):
            example_dialogue = input_train_data[index]["dialogue"]
            example_summary = input_train_data[index]["summary"]
            prompt_template += f"""
            Example {index + 1}:\n
            Input Dialogue: \n {example_dialogue}\n
            Summary: \n {example_summary}\n
            """
        prompt_template +=  f"""Provide a summary for the following dialogue. 
        \n
        
        Dialogue:\n
        {input_test_data['dialogue'][0]}\n:


        Summary:\n
        """
        # prompt_template +=  f"Follow the above pattern, please summarize the following dialogue:\n\n{input_test_data['dialogue']}:"
# Don't generate any additional dialogue or explaination.
        
    # prepare input data
    inputs = tokenizer(prompt_template, return_tensors="pt")

    # check optional parameters
    parameters = {"do_sample":do_sample, "max_new_tokens": max_new_tokens}
    if do_sample == True:
        parameters.update({"temperature":temperature,
                        "top_k":top_k, 
                        "top_p":top_p})

    # generate output
    output = model.generate(
        inputs["input_ids"],
        **parameters
        )
    print("output matrix shape:",output.shape)

    # decode output
    prompt_len = inputs["input_ids"].shape[1]
    summary = tokenizer.decode(output[0,prompt_len:], skip_special_tokens=True)

    print(prompt_template)
    print("-".join('*' for i in range(30)))
    print(f"Generated Summary: {summary}")
    print(f"Original Summary: {input_test_data['summary'][0]}")

    return prompt_template, summary


if __name__ ==  "__main__":
    arguments = arg()
    # load data
    dataset_train, dataset_test = import_data(text=True)
    # print(dataset_train[0]["dialogue"],"\n",dataset_train[0]["summary"],"1st \n")
    # print(dataset_train[1]["dialogue"],"\n",dataset_train[1]["summary"],"2nd \n")
    # print(dataset_train[2]["dialogue"],"\n",dataset_train[2]["summary"],"3rd \n")

    # build model
    model_name = "Qwen/Qwen3-1.7B"
    model, tokenizer = build_model(model_name)

    # test n shots
    test_n_shots(tokenizer, 
                model, 
                n=arguments.n, 
                input_train_data=dataset_train,
                input_test_data=dataset_test,
                do_sample=arguments.do_sample,
                max_new_tokens=arguments.max_new_tokens,
                temperature=arguments.temperature,
                top_k=arguments.top_k, 
                top_p=arguments.top_p)