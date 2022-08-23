## 实践课第三天

### 任务：Gym 倒立摆 作业要求: 提交通过并且在金榜的排名高于Jidi_random


---
### Env 👉请看 [ccgame.py](env/ccgame.py)

### Random 👉请看 [random/submission.py](examples/random/submission.py)

### 提交 👉请看 [submission.py](examples/random/submission.py)

---

### Install Gym
>pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gym==0.18.3

### How to test submission

Complete examples/submission/submission.py, and then set "policy_list" in line 176 of run_log.py
>python run_log.py 

If no errors, your submission is ready to go~

### Ready to submit
> random: [random/submission.py](examples/random/submission.py)

> DDPG: [ddpg/submission.py](examples/ddpg/submission.py) (To submit, change the `SUBMISSION` variable on line 181 to `True`) 
> and [ddpg/actor_200.pth](examples/ddpg/actor_200.pth)


### How to train DDPG agent
> python train.py 
>(`SUBMISSION` variable in [ddpg/submission.py](examples/ddpg/submission.py) 
> controls whether to train from scratch. `False` means to train from scratch.)
> 
> The model will be store under the folder [ddpg/trained_model](examples/ddpg/trained_model).

___
Have a good time~~~