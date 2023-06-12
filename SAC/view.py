from dm_control import suite
from dm_control import viewer
from dm_control.rl.control import Environment

from ee619.agent import Agent

if __name__ == '__main__':
        env: Environment = suite.load(domain_name='walker',
                                      task_name='run', task_kwargs={"random": 0})
        agent = Agent()
        agent.load()
        viewer.launch(env, policy=(lambda time_step: agent.act(time_step)), height=1080, width=1920)
