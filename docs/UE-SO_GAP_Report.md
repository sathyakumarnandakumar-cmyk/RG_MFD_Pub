# UE–SO Gap: Findings Summary

## 1. MFD production curves

![MFD production curves](figures/mfd_production.png)

- `7_region`: no jam point
- `7_region_new`: jams at 5,536 veh

| | aMFD | bMFD | cMFD | n_critical | n_jam |
|---|---|---|---|---|---|
| 7_region | 4.79e-10 | -1.53e-5 | 0.1257 | 5,572 | none |
| 7_region_new | -1.04e-8 | -8.2e-4 | 4.858 | 2,812 | 5,536 |

## 2. Load curves

![init-acc-scale](figures/load_curve_init_acc.png)
![demand-scale](figures/load_curve_demand.png)

- init-acc-scale: t=0 only
- future-demand-scale: real lever
- fd 0.8 ≈ critical

## 3. MARG cost

`marg = avg × r`, `r = 2 − n·G'/G`
`--marg-weight`

## 4. When MARG cost matters

| % critical | r |
|---|---|
| 2% | 1.01 |
| 22% (old fd50) | 1.16 |
| 67% | 1.58 |
| 112% | 2.19 |

r≈1 below ~50% critical.

Current runs (worst region, peak): antic_fd070 46% critical, antic_fd080 56% critical — both above the dead zone, between the 22% and 67% rows.

## 5. Argmax ceiling

SO needs flow-splitting. Argmax can't split. Needs stochastic policy (PPO/IPPO).
