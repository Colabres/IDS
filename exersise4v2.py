# %% [markdown]
# # Introduction to Data Science 2025
# 
# # Week 4

# %% [markdown]
# In this week's exercise, we look at prompting and zero- and few-shot task settings. Below is a text generation example from https://github.com/TurkuNLP/intro-to-nlp/blob/master/text_generation_pipeline_example.ipynb demonstrating how to load a text generation pipeline with a pre-trained model and generate text with a given prompt. Your task is to load a similar pre-trained generative model and assess whether the model succeeds at a set of tasks in zero-shot, one-shot, and two-shot settings.
# 
# Note: Downloading and running the pre-trained model locally may take some time. Alternatively, you can open and run this notebook on Google Colab (https://colab.research.google.com/), as assumed in the following example.

# %% [markdown]
# ## Text generation example
# 
# This is a brief example of how to run text generation with a causal language model and pipeline.
# 
# Install transformers (https://huggingface.co/docs/transformers/index) python package. This will be used to load the model and tokenizer and to run generation.

# %%
#!pip install --quiet transformers

# %% [markdown]
# Import the AutoTokenizer, AutoModelForCausalLM, and pipeline classes. The first two support loading tokenizers and generative models from the Hugging Face repository (https://huggingface.co/models), and the last wraps a tokenizer and a model for convenience.

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# %% [markdown]
# Load a generative model and its tokenizer. You can substitute any other generative model name here (e.g. other TurkuNLP GPT-3 models (https://huggingface.co/models?sort=downloads&search=turkunlp%2Fgpt3)), but note that Colab may have issues running larger models. 

# %%
MODEL_NAME = 'models/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

from llama_cpp import Llama

llm = Llama(
    model_path=MODEL_NAME,
    n_ctx=4096,
    seed=0,
    n_threads=None,   
)

def pipe(prompt, max_new_tokens=25, do_sample=False, temperature=0.0, top_p=0.9, **kw):
    
    temp = temperature if do_sample else 0.0
    out = llm(
        prompt=prompt,
        max_tokens=max_new_tokens,
        temperature=temp,
        top_p=top_p,
        stop=kw.get("stop"),  # optional
    )
    text = out["choices"][0]["text"]
    return [{"generated_text": prompt + text}]
# %% [markdown]
# Instantiate a text generation pipeline using the tokenizer and model.

# %%
# pipe = pipeline(
#     'text-generation',
#     model=model,
#     tokenizer=tokenizer,
#     device=model.device
# )

# %% [markdown]
# We can now call the pipeline with a text prompt; it will take care of tokenizing, encoding, generation, and decoding:

# %%
#output = pipe('Terve, miten menee?', max_new_tokens=25)

#print(output)

# %% [markdown]
# Just print the text

# %%
#print(output[0]['generated_text'])

# %% [markdown]
# We can also call the pipeline with any arguments that the model "generate" function supports. For details on text generation using transformers, see e.g. this tutorial (https://huggingface.co/blog/how-to-generate).
# 
# Example with sampling and a high temperature parameter to generate more chaotic output:

# %%
# output = pipe(
#     'Terve, miten menee?',
#     do_sample=True,
#     temperature=10.0,
#     max_new_tokens=25
# )

# print(output[0]['generated_text'])

# %% [markdown]
# ## Exercise 1
# 
# Your task is to assess whether a generative model succeeds in the following tasks in zero-shot, one-shot, and two-shot settings:
# 
# - binary sentiment classification (positive / negative)
# 
# - person name recognition
# 
# - two-digit addition (e.g. 11 + 22 = 33)
# 
# For example, for assessing whether a generative model can name capital cities, we could use the following prompts:
# 
# - zero-shot:
# 	"""
# 	Identify the capital cities of countries.
# 	
# 	Question: What is the capital of Finland?
# 	Answer:
# 	"""
# - one-shot:
# 	"""
# 	Identify the capital cities of countries.
# 	
# 	Question: What is the capital of Sweden?
# 	Answer: Stockholm
# 	
# 	Question: What is the capital of Finland?
# 	Answer:
# 	"""
# - two-shot:
# 	"""
# 	Identify the capital cities of countries.
# 	
# 	Question: What is the capital of Sweden?
# 	Answer: Stockholm
# 	
# 	Question: What is the capital of Denmark?
# 	Answer: Copenhagen
# 	
# 	Question: What is the capital of Finland?
# 	Answer:
# 	"""
# 
# You can do the tasks either in English or Finnish and use a generative model of your choice from the Hugging Face models repository, for example the following models:
# 
# - English: "gpt2-large"
# - Finnish: "TurkuNLP/gpt3-finnish-large"
# 
# You can either come up with your own instructions for the tasks or use the following:
# 
# - English:
# 	- binary sentiment classification: "Do the following texts express a positive or negative sentiment?"
# 	- person name recognition: "List the person names occurring in the following texts."
# 	- two-digit addition: "This is a first grade math exam."
# - Finnish:
# 	- binary sentiment classification: "Ilmaisevatko seuraavat tekstit positiivista vai negatiivista tunnetta?"
# 	- person name recognition: "Listaa seuraavissa teksteissä mainitut henkilönnimet."
# 	- two-digit addition: "Tämä on ensimmäisen luokan matematiikan koe."
# 
# Come up with at least two test cases for each of the three tasks, and come up with your own one- and two-shot examples.

# %%
# Use this cell for your code
first_promt =     '''       "Decide if the review is positive or negative. Use labels POS or NEG.\n\n"
        Review: "I loved this movie, it was amazing!"
        Label: "'''
#first test result "POS"
second_promt = '''       "Decide if the review is positive or negative. Use labels POS or NEG.\n\n"
        Review: "I loved this movie, it was amazing!"
        Label: "POS"
        Review: "This movie was the worst I have ever seen."
        Label: '''
#second test result "NEG"
third_promt = '''       "Decide if the review is positive or negative. Use labels POS or NEG.\n\n"
        Review: "I loved this movie, it was amazing!"
        Label: "POS"
        Review: "This movie was the worst I have ever seen."
        Label: "NEG"
        Review: "The soundtrack was absolutely beautiful."
        Label: '''
#third test result "POS"
promt21 = '''List the person names occurring in the text. Use a comma-separated list. If none, write None.
        Text: "Alice met Bob and Charlie at the cafe."
'''
#test 2.1 result "Step"
promt22 = '''List the person names occurring in the text. Use a comma-separated list. If none, write None.
        Text: "Alice met Bob and Charlie at the cafe."
        Names: [Alice, Bob, Charlie]
        Text: "Alice loves Bob and Alice loves cofe"
        Names:
'''

#test 2.2 result "Text: "Bob
promt23 = '''List the person names occurring in the text. Use a comma-separated list. If none, write None.
        Text: "Alice met Bob and Charlie at the cafe."
        Names: [Alice, Bob, Charlie]
        Text: "Alice loves Bob and Alice loves cofe"
        Names: [Alice, Bob]
        Test: "There are no names in this text"
        Names: 
'''
#test 2.3 result "Text: "Alice
promt31 = "2+2="
#test 31 result "4"
promt32 = "2+2=4 25+27="
#test 32 result "52"
promt33 = "2+2=4 25+27=52 573+1992="
#test 32 result "2565"
output = pipe(
    promt33,
    do_sample=True,
    temperature=0.3,
    max_new_tokens=2
)

print(output[0]['generated_text'])
#First try was with gpt2 and the result was terible. The lLama3 result are much more steble but still the Listing names failed 3/3. With enoght promting the results are better but verry unrelieble
# %% [markdown]
# **Submit this exercise by submitting your code and your answers to the above questions as comments on the MOOC platform. You can return this Jupyter notebook (.ipynb) or .py, .R, etc depending on your programming preferences.**


