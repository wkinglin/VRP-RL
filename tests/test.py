import pytest

from parco.envs import HCVRPEnv, OMDCPDPEnv
from parco.models import PARCOPolicy as PARCOPolicyRouting


@pytest.mark.parametrize("env_class", [HCVRPEnv, OMDCPDPEnv])
def test_parco_routing(env_class):
    env = env_class(generator_params={"num_loc": 20, "num_agents": 3})
    td_test_data = env.generator(batch_size=[2])
    td_init = env.reset(td_test_data.clone())
    td_init_test = td_init.clone()
    policy = PARCOPolicyRouting(env_name=env.name)
    out = policy(td_init_test.clone(), env)
    assert out["reward"].shape == (2,)


## TODO @laurin small fix to make it work
##  RuntimeError: The size of tensor a (11) must match the size of tensor b (10) at non-singleton dimension 1
# def test_parco_scheduling():
#     num_machine, num_stage = 3, 4
#     env = FFSPEnv(generator_params={"num_machine":num_machine, "num_stage":num_stage})
#     td_test_data = env.generator(batch_size=[2])
#     td_init = env.reset(td_test_data.clone())
#     td_init_test = td_init.clone()
#     parco = PARCOMultiStagePolicy(env_name=env.name, init_embedding_kwargs={"one_hot_seed_cnt":num_machine}, num_stages=num_stage)
#     out = parco(td_init_test.clone(), env=env)
#     assert out["reward"].shape == (2,)
