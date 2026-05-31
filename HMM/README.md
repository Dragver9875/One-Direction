# One-Direction HMM

SOTA-style classical HMM/Viterbi baseline for One-Direction map matching.

## Entry point

```powershell
python HMM\scripts\run_hmm.py all --config HMM\configs\hmm_default.yaml --split test
```

## Variants

```powershell
python HMM\scripts\run_hmm.py all --config HMM\configs\hmm_emission_only.yaml --split test
python HMM\scripts\run_hmm.py all --config HMM\configs\hmm_transition_light.yaml --split test
python HMM\scripts\run_hmm.py all --config HMM\configs\hmm_online_fixed_lag.yaml --split test
python HMM\scripts\run_hmm.py all --config HMM\configs\hmm_offroad_debug.yaml --split test
```

## Tune

```powershell
python HMM\scripts\run_hmm.py tune --config HMM\configs\hmm_default.yaml
python HMM\scripts\run_hmm.py make-tuned-config --config HMM\configs\hmm_default.yaml
python HMM\scripts\run_hmm.py all --config HMM\configs\hmm_tuned.yaml --split test
```

## Included upgrades

```text
adaptive distance uncertainty
speed/yaw-reliability-aware yaw scoring
route-distance and route-minus-GPS transition penalties
turn, sharp-turn, and U-turn penalties
transition scaling
forward-backward posterior confidence
entropy and second-best margin diagnostics
fixed-lag online-style decoding
optional off-road/map-error state
candidate-aware evaluation
trajectory path edit distance
near-but-wrong-edge and severe-error taxonomy
grid search and tuned-config generation
```
