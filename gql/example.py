import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = """\
You are at state 3. Go left. You are at state 2. Receive a reward.
You are at state 2. Receive a reward.
You are at state 0. Go right. You are at state 1. Go right. You are at state 2. Receive a reward.
You are at state 4. Go left. You are at 3. Go left. You are at state 2. Receive a reward.
You are at state 1. Go right. You are at state 2. Receive a reward.\
"""
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    temperature=0.1,
    max_tokens=200,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)
for i, choice in enumerate(response.choices):
    print("Output", i)
    print(choice.text)
