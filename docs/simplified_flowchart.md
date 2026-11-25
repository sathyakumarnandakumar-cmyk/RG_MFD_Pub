# Math and logic flowchart

The simulation as a sequence of calculations: what is computed, from what, and what is passed on. No
function names appear here. The codebase view of the same loop is in
[`SIMULATION_FLOWCHART.md`](SIMULATION_FLOWCHART.md). Equation numbers refer to the report
(`RG_MFD_Report_2026-11-24.pdf`).

## 1. One event

```mermaid
flowchart TD
    A["Initial state<br/>t = 0, Active = vehicles with t_dep = 0<br/>Waiting = vehicles with t_dep > 0"]
    B{"Active or Waiting<br/>not empty?"}
    C["Next event time (Eq. 4.4)<br/>t' = min( min t_arr_k, min t_dep_k )<br/>dt = t' - t"]
    D["Admit departures<br/>Waiting to Active if t_dep_k = t'"]
    E["Accumulation (Eq. 3.12)<br/>n_i = number of active vehicles in region i"]
    F["Speed (Eq. 3.5)<br/>v_i = max( l_i (d1 n_i^2 + d2 n_i + d3) phi s, v_min )"]
    G["Region travel time (Eq. 3.6)<br/>T_i = n_i / g_i(n_i)"]
    H["Edge cost<br/>C_ij = T_i + T_j if i, j adjacent, otherwise infinity"]
    I["Shortest paths<br/>Theta(i, j) = Dijkstra over C"]
    J["Motion (Eq. 4.5)<br/>L_rem_k = max( 0, L_rem_k - v_r(k) dt )"]
    K{"L_rem_k <= eps<br/>(leg finished)?"}
    L{"r(k) = Dest_k?"}
    M["Feasible set (Eq. 4.10)<br/>F = adjacent, not blocked,<br/>Dest_k still reachable"]
    N{"vehicle<br/>class"}
    O["HDV<br/>h = second region of Theta(r(k), Dest_k)"]
    P["CAV<br/>S = ( 1[curr], 1[Dest], n_i / cap_i, 1[F] )  (Eqs. 4.11, 4.12)<br/>h = argmax over F of Q(S, a)  (Eq. 4.13)"]
    Q["Move to h<br/>r(k) = h<br/>new leg length L_rem_k (Eq. 4.8)"]
    R["Reschedule (Eq. 4.6)<br/>t_arr_k = t' + L_rem_k / max( v_r(k), v_min )"]
    S["Completion (Eq. 4.7)<br/>done_k if leg_k > route length and r(k) = Dest_k<br/>Active to Completed"]
    T["Output<br/>travel time of each completed vehicle,<br/>n_i(t) history"]

    A --> B
    B -->|"yes"| C
    B -->|"no"| T
    C -->|"t', dt"| D
    D -->|"Active"| E
    E -->|"n_i"| F
    E -->|"n_i"| G
    F -->|"v_i"| J
    G -->|"T_i"| H
    H -->|"C_ij"| I
    I -->|"Theta"| O
    J -->|"L_rem_k"| K
    K -->|"no"| R
    K -->|"yes"| L
    L -->|"yes: arrived"| R
    L -->|"no"| M
    M -->|"F"| N
    N -->|"HDV: Theta, F"| O
    N -->|"CAV: n_i, Dest_k, F"| P
    O -->|"h"| Q
    P -->|"h"| Q
    Q -->|"r(k), L_rem_k"| R
    F -->|"v_i"| R
    R -->|"t_arr_k"| S
    S -->|"Active"| B
```

If $|F| = 1$ that region is taken without consulting the policy. If $F$ is empty no region is
chosen, and the vehicle is routed straight toward its destination.

## 2. Equations in order

$$t' = \min\Big(\min_k t^{arr}_k,\ \min_k t^{dep}_k\Big), \qquad \Delta t = t' - t \tag{Eq. 4.4}$$

$$n_i(t) = \big|\{\,k \in \mathit{Active} : r(k) = i\,\}\big| \tag{Eq. 3.12}$$

$$v_i(n) = \max\Big(l_i\left(\delta_1 n^2 + \delta_2 n + \delta_3\right)\varphi(\rho)\,\sigma,\ v_{min}\Big) \tag{Eq. 3.5}$$

$$T_i(n) = \frac{n}{g_i(n)} \tag{Eq. 3.6}$$

$$C_{ij} = T_i + T_j \quad \text{for adjacent } i, j, \qquad C_{ij} = \infty \text{ otherwise}$$

$$L^{rem}_k(t + \Delta t) = \max\big(0,\ L^{rem}_k(t) - v_{r(k)}\,\Delta t\big) \tag{Eq. 4.5}$$

$$F(i, \mathit{Dest}) = \{\,h : h \text{ adjacent to } i \text{ and not blocked, } \mathit{Dest} \text{ reachable from } h\,\} \tag{Eq. 4.10}$$

$$a^* = \arg\max_{a \in F(i, \mathit{Dest})} Q(S, a) \tag{Eq. 4.13}$$

$$t^{arr}_k = t + \frac{L^{rem}_k}{\max(v_{r(k)},\ v_{min})} \tag{Eq. 4.6}$$

$$\text{done}_k \iff \big(\text{leg}_k > |\text{route}_k|\big) \wedge \big(r(k) = \mathit{Dest}_k\big) \tag{Eq. 4.7}$$

The blocked set contains the regions the vehicle has already visited. If that leaves $F$ empty, the
blocked set is dropped for that decision. The reward passed to the agent after each event is
$-\Delta t \cdot N_{active}$, where $N_{active} = |\mathit{Active}|$.

## 3. Variables

| Symbol | Meaning | Units |
|---|---|---|
| $t$, $\Delta t$ | Simulation time, time since the previous event | s |
| $t^{dep}_k$, $t^{arr}_k$ | Departure time of vehicle $k$, time it finishes its current leg | s |
| $r(k)$ | Region vehicle $k$ occupies | region index |
| $\mathit{Dest}_k$ | Destination region of vehicle $k$ | region index |
| $n_i$ | Accumulation of region $i$ | veh |
| $v_i$, $v_{min}$ | Speed in region $i$, speed floor (0.5 m/s) | m/s |
| $l_i$ | Mean trip length in region $i$ | m |
| $\delta_1, \ldots, \delta_6$ | MFD coefficients | none |
| $\varphi(\rho)$, $\sigma$ | CAV-penetration factor, unit scale | none |
| $g_i$ | Trip completion rate of region $i$ | veh/s |
| $T_i$ | Average travel time in region $i$ | s |
| $C_{ij}$ | Cost of moving from region $i$ to adjacent region $j$ | s |
| $\Theta(i, j)$ | Shortest path from $i$ to $j$ over $C$ | regions |
| $L^{rem}_k$ | Distance vehicle $k$ still has to cover on its current leg | m |
| $\varepsilon$ | Arrival tolerance, $10^{-4}$ m | m |
| $F$ | Feasible next regions | set |
| $S$ | CAV state: current region (7), destination (7), $n_i / \mathrm{cap}_i$ (7), mask (7) | none |
| $Q(S, a)$ | Learned value of choosing region $a$ in state $S$ | none |
| $h$ | Next region chosen | region index |
