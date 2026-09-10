# import libraries
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import argparse
import os
import random
import pandas as pd
from pathlib import Path


# argparse
def arg():
    args = argparse.ArgumentParser()
    args.add_argument("--instructions",
                       action=argparse.BooleanOptionalAction,
                       default=True)
    args.add_argument("--max_new_tokens",type=int,default=50)
    # args.add_argument("--n",type=int,default=0)
    # args.add_argument("--do_sample", 
    #                   action=argparse.BooleanOptionalAction,
    #                   default=False)
    # args.add_argument("--temperature",type=float,default=0.7)
    # args.add_argument("--top_k",type=int,default=50)
    # args.add_argument("--top_p",type=float,default=0.95)
    return args.parse_args()


# import build dataset
def import_data(seed):
    dataset = load_dataset("knkarthick/dialogsum")
    rd = random.Random(seed)
    indices = rd.sample(range(200),31)
    dataset_train = dataset["test"].select(indices[:-1])
    dataset_test = dataset["test"].select([indices[-1]])
    print("data loaded.")
    return dataset_train, dataset_test


def build_local_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,local_files_only=True)
    print("model loaded.")
    return model, tokenizer


def generate_prompt(n,
        instructions,
        input_train_data,
        input_test_data):
     # prepare prompt template
    if not instructions:
        prompt_template = f"{input_test_data['dialogue'][0]}"
    else:
        if n == 0:
            prompt_template = f"You are a model that summarize dialogues. Please summarize the following dialogue consicely:\n\n{input_test_data['dialogue'][0]}:"
        else:
            prompt_template = f"""
            You are a model that summarize dialogues.
            
            ### Summary Guideline:
            Identify the main participant(s), the main action, and the main person, object, or event involved.
            A summary should mainly follow this structure:
            Somebody + main action + somebody/something
            """

            for index in range(n):
                example_dialogue = input_train_data[index]["dialogue"]
                example_summary = input_train_data[index]["summary"]
                prompt_template += f"""
                ### Example {index + 1} \n
                Dialogue: \n {example_dialogue}\n
                Summary: \n {example_summary}\n
                """
            prompt_template +=  f"""
            ### Task: Summarize the following dialogue consicely in the same style as the examples above.  
            Dialogue:\n
            {input_test_data['dialogue'][0]}\n

            Summary:\n
            """
        # old prompt:
        # prompt_template +=  f"Follow the above pattern, 
        # please summarize the following dialogue:\n\n{input_test_data['dialogue']}:"
        # Don't generate any additional dialogue or explaination.
        # """
        # Summarize the target dialogue in the same style as the examples below.
        # Write a concise summary that captures the main events or information.
        # Output only the summary.
        # """

    return prompt_template


def test_n_shots_api(prompt_template,
                    do_sample,
                    max_new_tokens,
                    temperature,
                    top_p):
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key="9e79d791c05e40648ae461d2502e474d.8yXFNanrggD6qxB8")

    parameters = {
        "max_tokens": max_new_tokens,
        "do_sample": do_sample
    }

    if do_sample:
        parameters.update({
            "temperature": temperature,
            "top_p": top_p
        })
    print("parameters:",parameters)

    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {
                "role":"user",
                "content":prompt_template
            }
        ],
        **parameters
    )

    message = response.choices[0].message

    # print("Reasoning:")
    # print(message.reasoning_content)

    # print("\nFinal answer:")
    # print(message.content)

    # print("\nFinish reason:")
    # print(response.choices[0].finish_reason)

    return response.choices[0].message.content


# n shots test
def test_n_shots_local(tokenizer, 
                model, 
                prompt,
                *, # below are parameters for model.generate()
                do_sample,
                max_new_tokens,
                temperature,
                top_k, 
                top_p):

    # prepare input data
    inputs = tokenizer(prompt, return_tensors="pt")

    # check optional parameters
    parameters = {"do_sample":do_sample, "max_new_tokens": max_new_tokens}
    if do_sample == True:
        parameters.update({"temperature":temperature,
                        "top_k":top_k, 
                        "top_p":top_p})
    print("parameters:",parameters)

    # generate output
    output = model.generate(
        inputs["input_ids"],
        **parameters
        )

    # decode output
    prompt_len = inputs["input_ids"].shape[1]
    summary = tokenizer.decode(output[0,prompt_len:], skip_special_tokens=True)

    return summary


if __name__ ==  "__main__":
    arguments = arg()
    # load data
    seed=5
    dataset_train, dataset_test = import_data(seed)

    # test grid
    shot_list = [0, 1, 3, 10, 18]
    # shot_list = [0]

    configs = [
        {
            "name": "no_sampling",
            "do_sample": False,
            "temperature": None,
            "top_p": None,
        },

        # temperature
        {
            "name": "temperature_0.2",
            "do_sample": True,
            "temperature": 0.2,
            "top_p": None,
        },
        {
            "name": "temperature_0.5",
            "do_sample": True,
            "temperature": 0.5,
            "top_p": None,
        },
        {
            "name": "temperature_0.8",
            "do_sample": True,
            "temperature": 0.8,
            "top_p": None,
        },
        {
            "name": "temperature_1.0",
            "do_sample": True,
            "temperature": 1.0,
            "top_p": None,
        },

        # Only change top_p
        {
            "name": "top_p_0.3",
            "do_sample": True,
            "temperature": None,
            "top_p": 0.3,
        },
        {
            "name": "top_p_0.6",
            "do_sample": True,
            "temperature": None,
            "top_p": 0.6,
        },
        {
            "name": "top_p_0.9",
            "do_sample": True,
            "temperature": None,
            "top_p": 0.9,
        },
        {
            "name": "top_p_1.0",
            "do_sample": True,
            "temperature": None,
            "top_p": 1.0,
        },
    ]

    results = []
    for n in shot_list:
        for config in configs:
            prompt = generate_prompt(n=n,
                                    instructions=arguments.instructions,
                                    input_train_data=dataset_train,
                                    input_test_data=dataset_test)

            # build local model
            # model_name = "Qwen/Qwen3-1.7B"
            # model, tokenizer = build_local_model(model_name)
            # # test n shots (local)
            # summary = test_n_shots_local(tokenizer, 
            #             model, 
            #             prompt,
            #             do_sample=arguments.do_sample,
            #             max_new_tokens=arguments.max_new_tokens,
            #             temperature=arguments.temperature,
            #             top_k=arguments.top_k, 
            #             top_p=arguments.top_p)


            # test few shots with API. ZhipuAI does not have top_k
            summary = test_n_shots_api(prompt_template=prompt,
                        do_sample=config["do_sample"],
                        max_new_tokens=arguments.max_new_tokens,
                        temperature=config["temperature"],
                        top_p=config["top_p"])  


            # print("-" * 30)
            # print(f"Generated Summary: {summary}")
            # print("-" * 30)
            # print(f"Original Summary: {dataset_test['summary'][0]}")
            results.append({"shots": n,
                            **config,
                            "Generated Summary":summary})

        Path(f"prompt_{n}_shots_seed{seed}.txt").write_text(prompt, encoding="utf-8")
    results.append({"Dialogue": [dataset_test['dialogue'][0],dataset_test['summary'][0]]})
    pd.DataFrame(results).to_csv(f"results_seed{seed}.csv", index=False)
