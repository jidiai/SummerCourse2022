## Behaviour Cloning example submission

这里是一个是用了BC的提交例子，针对奥林匹克相扑环境（Olympics-wrestling）。用户只需要提交`submission.py`和 `actor_state_dict.pt`文件至环境提交页面。

`submission.py`文件里的`my_controller`函数为评测时所调用的策略主函数，输入为观测obs，输出为动作actions。注意格式对齐。提交前可以在`course1/run_log.py`文件内测试，若能跑通则提交也能通过。