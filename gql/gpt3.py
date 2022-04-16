import shelve
import sys
from dataclasses import dataclass
from typing import cast

import openai
from env import MAX_TOKENS


@dataclass
class GPT3:
    db: shelve.DbfilenameShelf
    debug: bool = False

    def __call__(self, prompt, pause=True):
        self.print("<", end="")
        if prompt in self.db:
            completion = cast(str, self.db[prompt])
            # print("Completion:")
            # print(value)
            self.print(">", end="")
            return completion

        # print("Prompt:")
        # print(prompt)
        # breakpoint()
        #
        while True:
            # print("Prompt:", prompt.split("\n")[-1])
            sys.stdout.flush()
            choice, *_ = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                temperature=0.1,
                max_tokens=len(prompt) + MAX_TOKENS + 1,
            ).choices
            completion = choice.text.lstrip()
            if "." in completion:
                self.db[prompt] = completion
                self.print(">", end="")
                # print("Completion:", completion.split("\n")[0])
                # breakpoint()
                return completion

    def print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)
