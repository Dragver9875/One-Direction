import torch
from RL.src.onedir_ppo.data import EpisodeSample
from RL.src.onedir_ppo.features import action_mask, actor_observation, observation_dims, privileged_observation
from RL.src.onedir_ppo.models import PPOActorCritic

def make_sample():
    return EpisodeSample(1,torch.tensor([[0,1,-1],[1,2,3]]),torch.tensor([[True,True,False],[True,True,True]]),torch.randn(2,3,4),torch.tensor([0,1]),torch.ones(1,3,3,dtype=torch.bool),torch.randn(2,3,2),torch.randn(2,2),torch.tensor([0,2]),None)

def test_model_shapes():
    s=make_sample(); ad,cd,k=observation_dims(s); m=PPOActorCritic(ad,cd,k); a,lp,en,v=m.act(actor_observation(s,0,None),privileged_observation(s,0,None),action_mask(s,0)); assert a.numel()==1; assert lp.numel()==1; assert en.numel()==1; assert v.numel()==1
