from __future__ import annotations
import json, random
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from .data import RLDataset
from .features import action_mask, actor_observation, observation_dims
from .models import PPOActorCritic


def bc_tensors(dataset: RLDataset, k_max: int | None = None):
    obs=[]; masks=[]; labels=[]
    for sample in dataset.episodes:
        prev=None
        for t in range(sample.length):
            label=int(sample.gt_candidate_pos[t].item())
            if label < 0: continue
            obs.append(actor_observation(sample,t,prev,k_max)); masks.append(action_mask(sample,t,k_max)); labels.append(label); prev=label
    return torch.stack(obs), torch.stack(masks).bool(), torch.tensor(labels,dtype=torch.long)


def train_behavior_cloning(train_dataset: RLDataset, val_dataset: RLDataset, output_path: str | Path, device='cpu', epochs=10, batch_size=512, lr=1e-3, hidden_dim=256, num_layers=2, dropout=0.1, use_privileged_critic=True, seed=42, k_max=None):
    random.seed(seed); torch.manual_seed(seed)
    actor_dim, critic_dim, action_dim = observation_dims(train_dataset[0], k_max)
    model = PPOActorCritic(actor_dim, critic_dim, action_dim, hidden_dim, num_layers, dropout, use_privileged_critic).to(device)
    tr_obs,tr_masks,tr_labels=bc_tensors(train_dataset,k_max); va_obs,va_masks,va_labels=bc_tensors(val_dataset,k_max)
    loader=DataLoader(TensorDataset(tr_obs,tr_masks,tr_labels),batch_size=batch_size,shuffle=True)
    opt=torch.optim.AdamW(model.actor.parameters(),lr=lr)
    best=-1.0; history=[]; output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for obs,masks,labels in tqdm(loader,desc=f'bc epoch {epoch}',leave=False):
            loss=F.cross_entropy(model.masked_logits(obs.to(device),masks.to(device)),labels.to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            pred=model.masked_logits(va_obs.to(device),va_masks.to(device)).argmax(dim=-1).cpu()
            acc=float((pred==va_labels).float().mean().item())
        row={'epoch':epoch,'train_loss':sum(losses)/max(len(losses),1),'val_action_accuracy':acc}; history.append(row); print(json.dumps(row,indent=2))
        if acc>best:
            best=acc; torch.save({'model_state_dict':model.state_dict(),'actor_obs_dim':actor_dim,'critic_obs_dim':critic_dim,'action_dim':action_dim,'model_config':{'hidden_dim':hidden_dim,'num_layers':num_layers,'dropout':dropout,'use_privileged_critic':use_privileged_critic},'metrics':row},output_path)
    return {'history':history,'best_val_action_accuracy':best}
