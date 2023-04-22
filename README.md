Example run command:
```
python icpi/main.py --config config.yml
```

Example config:

```
argmax: true
balance_prompts: true 
constrain_prompts: true
env_id: space-invaders
eval_interval: null
hint: true 
logprobs: 5
max_prompts: 5
max_resamples: 3
max_tokens: 100
min_successes: 3
model_name: code-davinci-002
predict_transitions: true  
seed:
  - 0
  - 1
  - 2
  - 3
sil: false
success_buffer_size: 8
t_threshold: 0
temperature: 0.1
top_p: 1
total_steps: 195
wait_time: 4
policy_hint: true
```
