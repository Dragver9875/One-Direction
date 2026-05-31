
#!/usr/bin/env python3
from __future__ import annotations
import argparse, itertools, json, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor

MASK_VALUE = -1.0e9
OFFROAD_EDGE_IDX = -999999

@dataclass
class Episode:
    trajectory_id: int
    candidate_edge_idx: Tensor
    candidate_mask: Tensor
    emission_features: Tensor
    transition_features: Tensor
    transition_mask: Tensor
    gt_candidate_pos: Tensor
    candidate_proj_xy: Tensor | None = None
    gt_edge_idx: Tensor | None = None
    gt_proj_xy: Tensor | None = None
    emission_feature_names: list[str] | None = None
    transition_feature_names: list[str] | None = None

    @property
    def T(self) -> int:
        return int(self.candidate_edge_idx.shape[0])

    @property
    def K(self) -> int:
        return int(self.candidate_edge_idx.shape[1])

@dataclass
class HMMParams:
    emission_scale: float = 1.0
    distance_weight: float = 3.0
    log_distance_weight: float = 0.8
    yaw_weight: float = 1.2
    rank_weight: float = 1.2
    speed_consistency_weight: float = 0.15
    oneway_weight: float = 0.05
    road_class_prior_weight: float = 0.15
    candidate_density_weight: float = 0.10
    yaw_reliability_weight: float = 0.35
    adaptive_sigma_enabled: bool = True
    sigma_base: float = 0.35
    sigma_min: float = 0.20
    sigma_max: float = 2.50
    sigma_speed_weight: float = 0.30
    sigma_ambiguity_weight: float = 0.40
    sigma_yaw_unreliable_weight: float = 0.30
    transition_scale: float = 0.35
    legal_bonus: float = 0.35
    illegal_penalty: float = 2.0
    same_edge_bonus: float = 0.35
    same_osm_way_bonus: float = 0.15
    same_road_class_bonus: float = 0.05
    route_distance_weight: float = 0.25
    route_gps_ratio_weight: float = 0.10
    route_minus_gps_weight: float = 0.15
    turn_weight: float = 0.10
    yaw_change_weight: float = 0.05
    uturn_penalty: float = 0.50
    sharp_turn_speed_weight: float = 0.25
    time_feasible_bonus: float = 0.10
    time_infeasible_penalty: float = 0.50
    rank_delta_weight: float = 0.05
    distance_delta_weight: float = 0.05
    offroad_enabled: bool = False
    offroad_emission_bias: float = -5.0
    offroad_enter_penalty: float = 3.0
    offroad_exit_penalty: float = 2.0
    offroad_stay_bonus: float = 0.5

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8", newline="\n")

def deep_get(d: dict[str, Any], key: str, default=None):
    cur = d
    for p in key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def parse_value(v: str):
    v = v.strip()
    if v.lower() in {"true", "false"}: return v.lower() == "true"
    if v.lower() in {"none", "null"}: return None
    try: return int(v)
    except ValueError: pass
    try: return float(v)
    except ValueError: return v

def deep_set(d: dict[str, Any], key: str, value):
    cur = d
    parts = key.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value

def apply_overrides(cfg: dict[str, Any], overrides: list[str]):
    for o in overrides:
        if "=" not in o: raise ValueError(f"Override must be key=value: {o}")
        k, v = o.split("=", 1)
        deep_set(cfg, k, parse_value(v))
    return cfg

def tensor(x, dtype=None):
    if x is None: return None
    y = x if isinstance(x, Tensor) else torch.as_tensor(x)
    return y.to(dtype=dtype) if dtype is not None else y

def extract_payload(obj):
    if isinstance(obj, (list, tuple)): return list(obj)
    if isinstance(obj, dict):
        for k in ["episodes", "trajectories", "samples", "items", "data"]:
            if isinstance(obj.get(k), (list, tuple)): return list(obj[k])
        if "candidate_edge_idx" in obj: return [obj]
    raise ValueError(f"Unsupported dataset payload: {type(obj)}")

def make_episode(raw: Any, i: int) -> Episode:
    if not isinstance(raw, dict):
        raw = vars(raw)
    edge = tensor(raw["candidate_edge_idx"], torch.long)
    mask = tensor(raw.get("candidate_mask", edge >= 0), torch.bool)
    ef = tensor(raw["emission_features"], torch.float32)
    tf = tensor(raw.get("transition_features"), torch.float32)
    tm = tensor(raw.get("transition_mask"), torch.bool)
    if tf is None:
        tf = torch.zeros(max(edge.shape[0] - 1, 0), edge.shape[1], edge.shape[1], 0)
    if tm is None:
        tm = torch.ones(max(edge.shape[0] - 1, 0), edge.shape[1], edge.shape[1], dtype=torch.bool)
    gt_key = "gt_candidate_pos" if "gt_candidate_pos" in raw else "gt_pos"
    return Episode(
        trajectory_id=int(raw.get("trajectory_id", raw.get("id", i))),
        candidate_edge_idx=edge,
        candidate_mask=mask,
        emission_features=ef,
        transition_features=tf,
        transition_mask=tm,
        gt_candidate_pos=tensor(raw[gt_key], torch.long).reshape(-1),
        candidate_proj_xy=tensor(raw.get("candidate_proj_xy"), torch.float32),
        gt_edge_idx=tensor(raw.get("gt_edge_idx"), torch.long),
        gt_proj_xy=tensor(raw.get("gt_proj_xy"), torch.float32),
        emission_feature_names=list(raw.get("emission_feature_names", [])),
        transition_feature_names=list(raw.get("transition_feature_names", [])),
    )

def load_dataset(path: Path) -> list[Episode]:
    if not path.exists(): raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return [make_episode(x, i) for i, x in enumerate(extract_payload(payload))]

def dataset_summary(ds: list[Episode]) -> dict:
    return {
        "episodes": len(ds),
        "points": sum(e.T for e in ds),
        "labelled_points": sum(int((e.gt_candidate_pos >= 0).sum()) for e in ds),
        "min_length": min(e.T for e in ds),
        "max_length": max(e.T for e in ds),
        "max_candidates": max(e.K for e in ds),
        "emission_feature_dim": int(ds[0].emission_features.shape[-1]),
        "transition_feature_dim": int(ds[0].transition_features.shape[-1]),
    }

def fidx(names, name, fallback=None):
    return names.index(name) if name in names else fallback

def feat(x: Tensor, names: list[str] | None, name: str, fallback=None):
    names = names or []
    idx = fidx(names, name, fallback)
    if idx is None or idx >= x.shape[-1]:
        return torch.zeros(x.shape[:-1], dtype=x.dtype)
    return x[..., idx]

def params_from_config(cfg: dict[str, Any]) -> HMMParams:
    em, tr, off = cfg.get("emission", {}), cfg.get("transition", {}), cfg.get("offroad", {})
    return HMMParams(
        emission_scale=float(em.get("emission_scale", 1.0)),
        distance_weight=float(em.get("distance_weight", 3.0)),
        log_distance_weight=float(em.get("log_distance_weight", 0.8)),
        yaw_weight=float(em.get("yaw_weight", 1.2)),
        rank_weight=float(em.get("rank_weight", 1.2)),
        speed_consistency_weight=float(em.get("speed_consistency_weight", 0.15)),
        oneway_weight=float(em.get("oneway_weight", 0.05)),
        road_class_prior_weight=float(em.get("road_class_prior_weight", 0.15)),
        candidate_density_weight=float(em.get("candidate_density_weight", 0.10)),
        yaw_reliability_weight=float(em.get("yaw_reliability_weight", 0.35)),
        adaptive_sigma_enabled=bool(em.get("adaptive_sigma_enabled", True)),
        sigma_base=float(em.get("sigma_base", 0.35)),
        sigma_min=float(em.get("sigma_min", 0.20)),
        sigma_max=float(em.get("sigma_max", 2.50)),
        sigma_speed_weight=float(em.get("sigma_speed_weight", 0.30)),
        sigma_ambiguity_weight=float(em.get("sigma_ambiguity_weight", 0.40)),
        sigma_yaw_unreliable_weight=float(em.get("sigma_yaw_unreliable_weight", 0.30)),
        transition_scale=float(tr.get("transition_scale", 0.35)),
        legal_bonus=float(tr.get("legal_bonus", 0.35)),
        illegal_penalty=float(tr.get("illegal_penalty", 2.0)),
        same_edge_bonus=float(tr.get("same_edge_bonus", 0.35)),
        same_osm_way_bonus=float(tr.get("same_osm_way_bonus", 0.15)),
        same_road_class_bonus=float(tr.get("same_road_class_bonus", 0.05)),
        route_distance_weight=float(tr.get("route_distance_weight", 0.25)),
        route_gps_ratio_weight=float(tr.get("route_gps_ratio_weight", 0.10)),
        route_minus_gps_weight=float(tr.get("route_minus_gps_weight", 0.15)),
        turn_weight=float(tr.get("turn_weight", 0.10)),
        yaw_change_weight=float(tr.get("yaw_change_weight", 0.05)),
        uturn_penalty=float(tr.get("uturn_penalty", 0.50)),
        sharp_turn_speed_weight=float(tr.get("sharp_turn_speed_weight", 0.25)),
        time_feasible_bonus=float(tr.get("time_feasible_bonus", 0.10)),
        time_infeasible_penalty=float(tr.get("time_infeasible_penalty", 0.50)),
        rank_delta_weight=float(tr.get("rank_delta_weight", 0.05)),
        distance_delta_weight=float(tr.get("distance_delta_weight", 0.05)),
        offroad_enabled=bool(off.get("enabled", False)),
        offroad_emission_bias=float(off.get("emission_bias", -5.0)),
        offroad_enter_penalty=float(off.get("enter_penalty", 3.0)),
        offroad_exit_penalty=float(off.get("exit_penalty", 2.0)),
        offroad_stay_bonus=float(off.get("stay_bonus", 0.5)),
    )

def obs_sigma(ep: Episode, p: HMMParams):
    f, names = ep.emission_features.float(), ep.emission_feature_names
    speed = feat(f, names, "speed_norm", None)
    if speed.shape != f.shape[:-1]: speed = torch.zeros(f.shape[:-1])
    speed_pt = speed.mean(dim=1)
    yaw_rel = feat(f, names, "yaw_reliability", 15).clamp(0, 1)
    yaw_pt = yaw_rel.mean(dim=1) if yaw_rel.shape == f.shape[:-1] else torch.zeros(ep.T)
    density = ep.candidate_mask.float().sum(dim=1) / max(ep.K, 1)
    sigma = p.sigma_base + p.sigma_speed_weight*(1-speed_pt.clamp(0,1)) + p.sigma_ambiguity_weight*density + p.sigma_yaw_unreliable_weight*(1-yaw_pt)
    return sigma.clamp(p.sigma_min, p.sigma_max)

def emission_scores(ep: Episode, p: HMMParams):
    x, names = ep.emission_features.float(), ep.emission_feature_names
    dist = feat(x, names, "distance_norm", 0).clamp_min(0)
    logdist = feat(x, names, "log_distance_norm", 1).clamp_min(0)
    yaw = feat(x, names, "abs_yaw_diff_norm", 3).clamp_min(0)
    rank = feat(x, names, "candidate_rank_norm", 10).clamp_min(0)
    speed_cons = feat(x, names, "speed_consistency", 11).clamp_min(0)
    oneway = feat(x, names, "oneway", 12).clamp_min(0)
    yaw_rel = feat(x, names, "yaw_reliability", 15).clamp(0, 1)
    speed = feat(x, names, "speed_norm", None)
    if speed.shape != dist.shape: speed = torch.zeros_like(dist)
    road_prior = feat(x, names, "road_class_prior", None)
    if road_prior.shape != dist.shape: road_prior = torch.zeros_like(dist)
    density = ep.candidate_mask.float().sum(dim=1, keepdim=True) / max(ep.K, 1)
    if p.adaptive_sigma_enabled:
        sigma = obs_sigma(ep, p).view(-1, 1)
        dist_score = -0.5*(dist/sigma).pow(2) - torch.log(sigma.clamp_min(1e-6))
    else:
        dist_score = -dist
    yaw_factor = (p.yaw_reliability_weight + (1-p.yaw_reliability_weight)*yaw_rel) * (0.25 + 0.75*speed.clamp(0,1))
    score = (
        p.distance_weight*dist_score
        - p.log_distance_weight*logdist
        - p.yaw_weight*yaw*yaw_factor
        - p.rank_weight*rank
        - p.speed_consistency_weight*speed_cons
        + p.oneway_weight*oneway
        + p.road_class_prior_weight*road_prior
        - p.candidate_density_weight*density
    ) * p.emission_scale
    score = score.masked_fill(~ep.candidate_mask.bool(), MASK_VALUE)
    if p.offroad_enabled:
        score = torch.cat([score, torch.full((ep.T,1), p.offroad_emission_bias)], dim=1)
    return score

def transition_scores(ep: Episode, p: HMMParams, mode: str):
    T, K = ep.T, ep.K
    if T <= 1:
        base = torch.empty(0, K, K)
    else:
        tf, names = ep.transition_features.float(), ep.transition_feature_names
        if tf.numel() == 0:
            base = torch.zeros(T-1, K, K)
        else:
            route = feat(tf, names, "route_dist_norm", 3).clamp_min(0)
            minus = feat(tf, names, "route_minus_gps_norm", 4).clamp_min(0)
            ratio = feat(tf, names, "route_gps_ratio_norm", 5).clamp_min(0)
            turn = feat(tf, names, "turn_norm", 6).clamp_min(0)
            yawc = feat(tf, names, "yaw_change_norm", 7).clamp_min(0)
            connected = feat(tf, names, "connected", 9)
            legal = feat(tf, names, "legal", 10)
            same_edge = feat(tf, names, "same_edge", 11)
            same_way = feat(tf, names, "same_osm_way", 12)
            same_class = feat(tf, names, "same_road_class", 13)
            rank_delta = feat(tf, names, "candidate_rank_delta_norm", 15).clamp_min(0)
            dist_delta = feat(tf, names, "distance_delta_norm", 18).clamp_min(0)
            time_ok = feat(tf, names, "time_feasible", 19).clamp(0, 1)
            uturn = feat(tf, names, "uturn", None)
            if uturn.shape != turn.shape: uturn = (turn > 0.85).float()
            legal_like = ((legal > 0.5) | (same_edge > 0.5) | (connected > 0.5)).float()
            infeasible = 1 - time_ok
            sharp = (turn > 0.65).float()
            base = (
                p.legal_bonus*legal_like + p.same_edge_bonus*same_edge + p.same_osm_way_bonus*same_way
                + p.same_road_class_bonus*same_class + p.time_feasible_bonus*time_ok
                - p.illegal_penalty*(1-legal_like) - p.time_infeasible_penalty*infeasible
                - p.route_distance_weight*route - p.route_minus_gps_weight*minus - p.route_gps_ratio_weight*ratio
                - p.turn_weight*turn - p.yaw_change_weight*yawc - p.uturn_penalty*uturn
                - p.sharp_turn_speed_weight*sharp*(1-infeasible)
                - p.rank_delta_weight*rank_delta - p.distance_delta_weight*dist_delta
            ) * p.transition_scale
            if mode == "hard": base = base.masked_fill(~ep.transition_mask.bool(), MASK_VALUE)
            elif mode == "soft": base = base - (~ep.transition_mask.bool()).float()*(p.transition_scale*p.illegal_penalty)
            elif mode == "none": pass
            else: raise ValueError(f"Unsupported transition_mode: {mode}")
    if p.offroad_enabled:
        if base.numel() == 0: return torch.empty(0, K+1, K+1)
        out = torch.full((base.shape[0], K+1, K+1), -p.offroad_enter_penalty)
        out[:, :K, :K] = base
        out[:, :K, K] = -p.offroad_enter_penalty
        out[:, K, :K] = -p.offroad_exit_penalty
        out[:, K, K] = p.offroad_stay_bonus
        return out
    return base

def viterbi(E: Tensor, A: Tensor, beam_size=None):
    T, K = E.shape
    if T == 0: return [], torch.empty(0)
    dp = E[0].clone(); hist=[dp.clone()]; back=[]
    if beam_size and 0 < beam_size < K:
        keep = torch.topk(dp, beam_size).indices
        m = torch.ones_like(dp, dtype=torch.bool); m[keep] = False
        dp = dp.masked_fill(m, MASK_VALUE)
    for t in range(1, T):
        s = dp[:, None] + A[t-1] if A.numel() else dp[:, None]
        best, prev = s.max(dim=0)
        dp = E[t] + best
        if beam_size and 0 < beam_size < K:
            keep = torch.topk(dp, beam_size).indices
            m = torch.ones_like(dp, dtype=torch.bool); m[keep] = False
            dp = dp.masked_fill(m, MASK_VALUE)
        hist.append(dp.clone()); back.append(prev)
    last = int(dp.argmax()); path=[last]
    for bp in reversed(back):
        last = int(bp[last]); path.append(last)
    path.reverse()
    return path, torch.stack(hist)

def fixed_lag(E: Tensor, A: Tensor, lag: int, beam_size=None):
    T, K = E.shape; path=[]; hist=[]
    for t in range(T):
        s = max(0, t-lag+1)
        Aw = A[s:t] if A.numel() and t > s else torch.empty(0, K, K)
        p, h = viterbi(E[s:t+1], Aw, beam_size)
        path.append(p[-1]); hist.append(h[-1])
    return path, torch.stack(hist)

def forward_backward(E: Tensor, A: Tensor):
    T, K = E.shape
    la = torch.full_like(E, MASK_VALUE); lb = torch.zeros_like(E)
    la[0] = E[0]
    for t in range(1, T):
        la[t] = E[t] + torch.logsumexp(la[t-1][:, None] + A[t-1], dim=0)
    for t in range(T-2, -1, -1):
        lb[t] = torch.logsumexp(A[t] + E[t+1][None, :] + lb[t+1][None, :], dim=1)
    z = torch.logsumexp(la[-1], dim=0)
    P = torch.exp(la + lb - z)
    return P / P.sum(dim=1, keepdim=True).clamp_min(1e-12)

def softmax_conf(scores: Tensor, path: list[int], temp: float):
    P = torch.softmax(scores/max(temp,1e-6), dim=-1)
    return [float(P[t,a]) for t,a in enumerate(path)], (-(P*torch.log(P.clamp_min(1e-12))).sum(1)).tolist()

def margin(scores: Tensor, path: list[int]):
    vals=[]
    for t,a in enumerate(path):
        r=scores[t].clone(); b=float(r[a]); r[a]=MASK_VALUE; vals.append(b-float(r.max()))
    return vals

def decode_dataset(ds: list[Episode], p: HMMParams, cfg: dict[str,Any]):
    rows=[]
    trans_mode = str(deep_get(cfg,"decode.transition_mode","soft"))
    dec_mode = str(deep_get(cfg,"decode.mode","offline"))
    conf_mode = str(deep_get(cfg,"decode.confidence_mode","posterior"))
    temp = float(deep_get(cfg,"decode.confidence_temperature",1.0))
    lag = int(deep_get(cfg,"decode.fixed_lag",10))
    beam = deep_get(cfg,"decode.beam_size",None)
    for ep in ds:
        E = emission_scores(ep,p); A = transition_scores(ep,p,trans_mode)
        if dec_mode == "fixed_lag": path, hist = fixed_lag(E,A,lag,beam)
        else: path, hist = viterbi(E,A,beam)
        if conf_mode == "posterior" and A.numel() and dec_mode=="offline":
            P = forward_backward(E,A); conf=[float(P[t,a]) for t,a in enumerate(path)]
            ent = (-(P*torch.log(P.clamp_min(1e-12))).sum(1)).tolist()
        else:
            conf, ent = softmax_conf(hist,path,temp)
        mar = margin(hist,path)
        for t,a in enumerate(path):
            gt=int(ep.gt_candidate_pos[t])
            off = int(p.offroad_enabled and a >= ep.K)
            pred = OFFROAD_EDGE_IDX if off else int(ep.candidate_edge_idx[t,a]) if a < ep.K else -1
            if ep.gt_edge_idx is not None: gt_edge=int(ep.gt_edge_idx[t])
            elif 0 <= gt < ep.K: gt_edge=int(ep.candidate_edge_idx[t,gt])
            else: gt_edge=-1
            legal=True
            if t>0 and not off and path[t-1] < ep.K and a < ep.K and ep.transition_mask.numel():
                legal=bool(ep.transition_mask[t-1,path[t-1],a])
            row = dict(trajectory_id=int(ep.trajectory_id), t=int(t), pred_candidate_pos=int(a),
                       gt_candidate_pos=int(gt), pred_edge_idx=pred, gt_edge_idx=gt_edge,
                       is_offroad_state=off, transition_legal=int(legal), confidence=float(conf[t]),
                       entropy=float(ent[t]), second_best_margin=float(mar[t]),
                       emission_score=float(E[t,a]) if a<E.shape[1] else np.nan,
                       path_score=float(hist[t,a]) if a<hist.shape[1] else np.nan)
            if ep.candidate_proj_xy is not None and not off and a < ep.K:
                row["pred_proj_x"]=float(ep.candidate_proj_xy[t,a,0]); row["pred_proj_y"]=float(ep.candidate_proj_xy[t,a,1])
            else:
                row["pred_proj_x"]=np.nan; row["pred_proj_y"]=np.nan
            if ep.gt_proj_xy is not None:
                row["gt_proj_x"]=float(ep.gt_proj_xy[t,0]); row["gt_proj_y"]=float(ep.gt_proj_xy[t,1])
            rows.append(row)
    return pd.DataFrame(rows)

def edit_distance(a,b):
    if len(a)<len(b): a,b=b,a
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1): cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]

def evaluate(df: pd.DataFrame, cfg: dict[str,Any]):
    labelled=df[df.gt_edge_idx>=0].copy()
    if bool(deep_get(cfg,"evaluation.require_gt_candidate",True)):
        labelled=labelled[labelled.gt_candidate_pos>=0].copy()
    labelled["edge_correct"]=labelled.pred_edge_idx.astype(int)==labelled.gt_edge_idx.astype(int)
    labelled["action_correct"]=labelled.pred_candidate_pos.astype(int)==labelled.gt_candidate_pos.astype(int)
    if {"pred_proj_x","pred_proj_y","gt_proj_x","gt_proj_y"}.issubset(labelled.columns):
        labelled["projection_error_m"]=np.sqrt((labelled.pred_proj_x-labelled.gt_proj_x)**2+(labelled.pred_proj_y-labelled.gt_proj_y)**2)
    else: labelled["projection_error_m"]=np.nan
    th=float(deep_get(cfg,"evaluation.projection_threshold_m",10.0))
    succ=float(deep_get(cfg,"evaluation.trajectory_success_accuracy",0.90))
    for r in [2,5,10]: labelled[f"within_{r}m"]=labelled.projection_error_m<=r
    labelled["projection_success"]=labelled.projection_error_m<=th
    labelled["near_but_wrong_edge"]=(~labelled.edge_correct)&labelled.within_5m
    labelled["severe_error"]=(~labelled.edge_correct)&(~labelled.within_10m)
    labelled["low_confidence"]=labelled.confidence<0.35
    traj=[]; edits=[]
    for tid,g in labelled.groupby("trajectory_id"):
        g=g.sort_values("t"); e=edit_distance(g.pred_edge_idx.astype(int).tolist(), g.gt_edge_idx.astype(int).tolist())
        edits.append(e); acc=float(g.edge_correct.mean())
        traj.append(dict(trajectory_id=int(tid), points=int(len(g)), edge_accuracy=acc,
                         action_accuracy=float(g.action_correct.mean()), mean_projection_error_m=float(g.projection_error_m.mean()),
                         within_5m_rate=float(g.within_5m.mean()), mean_confidence=float(g.confidence.mean()),
                         mean_entropy=float(g.entropy.mean()), path_edit_distance=int(e), success=bool(acc>=succ)))
    traj_df=pd.DataFrame(traj); errors=labelled[~labelled.edge_correct].copy(); trans=labelled[labelled.t>0]
    metrics=dict(
        num_points=int(len(df)), num_labelled_points=int(len(labelled)), num_unlabelled_or_gt_missing_points=int(len(df)-len(labelled)),
        num_trajectories=int(labelled.trajectory_id.nunique()), point_action_accuracy=float(labelled.action_correct.mean()),
        point_edge_accuracy=float(labelled.edge_correct.mean()), mean_projection_error_m=float(labelled.projection_error_m.mean()),
        median_projection_error_m=float(labelled.projection_error_m.median()), p90_projection_error_m=float(labelled.projection_error_m.quantile(.9)),
        within_2m_rate=float(labelled.within_2m.mean()), within_5m_rate=float(labelled.within_5m.mean()), within_10m_rate=float(labelled.within_10m.mean()),
        projection_success_rate=float(labelled.projection_success.mean()), mean_confidence=float(labelled.confidence.mean()),
        mean_entropy=float(labelled.entropy.mean()), mean_second_best_margin=float(labelled.second_best_margin.mean()),
        pred_transition_legal_rate=float(trans.transition_legal.mean()) if len(trans) else 1.0,
        offroad_state_rate=float(labelled.is_offroad_state.mean()) if "is_offroad_state" in labelled else 0.0,
        path_edit_distance_mean=float(np.mean(edits)), path_edit_distance_median=float(np.median(edits)),
        trajectory_success_rate=float(traj_df.success.mean()), num_error_points=int((~labelled.edge_correct).sum()),
        error_near_but_wrong_edge_rate=float(errors.near_but_wrong_edge.mean()) if len(errors) else 0.0,
        error_severe_rate=float(errors.severe_error.mean()) if len(errors) else 0.0,
        low_confidence_error_rate=float(errors.low_confidence.mean()) if len(errors) else 0.0)
    return metrics,traj_df,errors

def run_check(cfg):
    for sp in ["train","val","test"]:
        ds=load_dataset(Path(deep_get(cfg,f"paths.{sp}_dataset"))); print(f"[OK] {sp}: {dataset_summary(ds)}", flush=True)

def run_decode(cfg, split):
    ds=load_dataset(Path(deep_get(cfg,f"paths.{split}_dataset"))); p=params_from_config(cfg)
    df=decode_dataset(ds,p,cfg); out=Path(deep_get(cfg,"paths.match_dir","HMM/outputs/matches"))/f"hmm_matches_{split}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True); df.to_parquet(out,index=False); print(f"[OK] Decoded {split}: {out}", flush=True); return out

def run_evaluate(cfg, split):
    mp=Path(deep_get(cfg,"paths.match_dir","HMM/outputs/matches"))/f"hmm_matches_{split}.parquet"
    if not mp.exists(): raise FileNotFoundError(f"Run decode first: {mp}")
    metrics,traj,errors=evaluate(pd.read_parquet(mp),cfg); md=Path(deep_get(cfg,"paths.metric_dir","HMM/outputs/metrics")); md.mkdir(parents=True,exist_ok=True)
    (md/f"hmm_metrics_{split}.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8",newline="\n")
    traj.to_csv(md/f"hmm_trajectory_metrics_{split}.csv",index=False); errors.to_csv(md/f"hmm_error_cases_{split}.csv",index=False)
    print("[OK] HMM evaluation complete", flush=True)
    for k,v in metrics.items(): print(f"{k}: {v}", flush=True)
    return metrics

def grid_product(grid):
    keys=list(grid); return [dict(zip(keys,c)) for c in itertools.product(*[grid[k] for k in keys])]

def run_tune(cfg):
    split=str(deep_get(cfg,"grid_search.split","val")); ds=load_dataset(Path(deep_get(cfg,f"paths.{split}_dataset")))
    base=asdict(params_from_config(cfg)); gcfg=deep_get(cfg,"grid_search",{})
    grid={k:v for k,v in gcfg.items() if k not in {"split","max_trials"} and isinstance(v,list)}
    trials=grid_product(grid); random.seed(int(deep_get(cfg,"project.seed",42))); random.shuffle(trials)
    mt=deep_get(cfg,"grid_search.max_trials",None); trials=trials[:int(mt)] if mt else trials
    rows=[]; best=None; best_score=-1
    for i,ov in enumerate(trials,1):
        pp=HMMParams(**{**base,**ov}); df=decode_dataset(ds,pp,cfg); metrics,_,_=evaluate(df,cfg); score=float(metrics["point_edge_accuracy"])
        rows.append({"trial":i,**ov,**metrics}); print(f"[grid] {i}/{len(trials)} point_edge_accuracy={score:.6f}", flush=True)
        if score>best_score: best_score=score; best={"params":asdict(pp),"overrides":ov,"metrics":metrics}
    md=Path(deep_get(cfg,"paths.metric_dir","HMM/outputs/metrics")); md.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(md/"hmm_grid_search.csv",index=False); (md/"hmm_best_params.json").write_text(json.dumps(best or {},indent=2),encoding="utf-8",newline="\n")
    print(f"[OK] HMM grid tuning complete. Best point_edge_accuracy={best_score:.6f}", flush=True)

def run_make_tuned(cfg):
    bp=Path(deep_get(cfg,"paths.metric_dir","HMM/outputs/metrics"))/"hmm_best_params.json"
    if not bp.exists(): raise FileNotFoundError(f"Run tune first: {bp}")
    best=json.loads(bp.read_text(encoding="utf-8")); params=best.get("params",{})
    out=dict(cfg); out["emission"]=dict(cfg.get("emission",{})); out["transition"]=dict(cfg.get("transition",{})); out["offroad"]=dict(cfg.get("offroad",{})); out.setdefault("decode",{})["split"]="test"
    em_keys={"emission_scale","distance_weight","log_distance_weight","yaw_weight","rank_weight","speed_consistency_weight","oneway_weight","road_class_prior_weight","candidate_density_weight","yaw_reliability_weight","adaptive_sigma_enabled","sigma_base","sigma_min","sigma_max","sigma_speed_weight","sigma_ambiguity_weight","sigma_yaw_unreliable_weight"}
    tr_keys={"transition_scale","legal_bonus","illegal_penalty","same_edge_bonus","same_osm_way_bonus","same_road_class_bonus","route_distance_weight","route_gps_ratio_weight","route_minus_gps_weight","turn_weight","yaw_change_weight","uturn_penalty","sharp_turn_speed_weight","time_feasible_bonus","time_infeasible_penalty","rank_delta_weight","distance_delta_weight"}
    for k,v in params.items():
        if k in em_keys: out["emission"][k]=v
        elif k in tr_keys: out["transition"][k]=v
    save_yaml(Path("HMM/configs/hmm_tuned.yaml"),out); print("[OK] Wrote HMM/configs/hmm_tuned.yaml", flush=True)

def main():
    ap=argparse.ArgumentParser(description="Single-file SOTA-style HMM/Viterbi workflow for One-Direction.")
    ap.add_argument("stage", nargs="?", default="all", choices=["check","decode","evaluate","all","tune","make-tuned-config"])
    ap.add_argument("--config", type=Path, default=Path("HMM/configs/hmm_default.yaml"))
    ap.add_argument("--split", choices=["train","val","test"], default=None)
    ap.add_argument("--override", nargs="*", default=[])
    args=ap.parse_args(); cfg=apply_overrides(load_yaml(args.config),args.override); split=args.split or str(deep_get(cfg,"decode.split","test"))
    if args.stage=="check": run_check(cfg)
    elif args.stage=="decode": run_decode(cfg,split)
    elif args.stage=="evaluate": run_evaluate(cfg,split)
    elif args.stage=="all": run_check(cfg); run_decode(cfg,split); run_evaluate(cfg,split)
    elif args.stage=="tune": run_tune(cfg)
    elif args.stage=="make-tuned-config": run_make_tuned(cfg)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
