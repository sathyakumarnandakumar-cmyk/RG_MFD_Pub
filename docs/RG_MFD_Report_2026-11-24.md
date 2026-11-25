---
title: "Event-Triggered Reinforcement Learning for CAV Route Guidance in Mixed Traffic on Trip-Based Macroscopic Fundamental Diagram Networks"
---

```{=latex}
% Front matter set in TeX Gyre Pagella (a Palatino-style serif) instead of the body font
\begingroup
\fontspec{TeX Gyre Pagella}\selectfont
```

# Acknowledgements {-}

I would like to express my sincere gratitude to my thesis supervisor, **Dr. hab. Rafał Kucharski** of Jagiellonian University, for pushing my thinking, challenging my assumptions, and helping shape my early hypotheses into a coherent piece of research. I also thank my supervisor and long-time mentor, **Dr. Paweł Gora**, also of Jagiellonian University, for introducing me to this research area, for valuable insights along the way, and for guidance while drafting this thesis. I am also deeply grateful to my co-supervisor, **Prof. dr hab. Joanna Rączaszek-Leonardi** of the Faculty of Psychology at the University of Warsaw, for her steady support and valuable perspective throughout this work.

This research was carried out at the COeXISTENCE Lab at Jagiellonian University, and I appreciate the access to the laboratory and computational resources that made the experiments possible. In particular, I am grateful to **Dr. hab. Rafał Kucharski** for the financial and logistical support covering my travel and accommodation during my research stays in Kraków.

Finally, I would like to thank my parents and my sister for their unwavering support, patience, and encouragement throughout my Master's programme.

# Declaration on the Use of AI Tools {-}

During the preparation of this thesis, I used large language models, namely Anthropic's Claude Code and Google Gemini, for the following purposes:

1. assisting with preliminary research to identify relevant literature and evaluate thematic research alignment;
2. supporting programming tasks, including code writing, debugging, and generating preliminary pseudocode;
3. formulating, formatting, and verifying mathematical formulas and equations within the algorithms in the thesis draft;
4. performing grammar correction and paraphrasing complex ideas and phrases into simpler structures to improve clarity.

Language models also helped turn the implemented code into the pseudocode algorithms presented in this thesis, such as the main event loop of the simulation, and I checked each algorithm against the code it describes.

All AI-assisted outputs, such as code, mathematical derivations, and analyses, were independently and meticulously reviewed, tested, and edited by me, ensuring that the thesis reflects my own original intellectual contribution.

```{=latex}
\endgroup
```


```{=latex}
\setcounter{chapter}{0}
```

# Introduction

**Abstract**

**Objectives:** Modern urban road networks carry both human-driven vehicles (HDVs) and Connected and Autonomous Vehicles (CAVs). Once the number of vehicles in a region reaches a critical point, \edt{the number of trips the region completes per unit time, its throughput, starts to fall.} Consequently, the spatial distribution of vehicles across regions governs how effectively available road capacity is utilised. Within this mixed-traffic environment, only CAVs can be centrally guided, whereas HDVs independently choose their routes in response to experienced congestion. The number of vehicles on the network also rises and falls as demand changes over the peak period. This thesis investigates whether reinforcement learning (RL) can route the CAVs on a regional simulation network modelled using macroscopic fundamental diagrams (MFDs) better than the standard shortest-path routing. We specifically assess how enriching the state observation vector of the routing policy with engineered features and combining multiple reward terms influence the resulting RL routing strategies.

**Methods:** Each region is treated as a homogeneous MFD zone and individual trips are followed through it. A CAV only consults the routing policy when it reaches a region boundary and will select the next adjacent region rather than a full path. Because the time between boundary crossings is variable, the decision process is semi-Markov. All CAVs use the same policy which is trained via DQN or PPO over masked legal moves. HDVs will recompute a deterministic shortest path at every boundary. \edt{The experiments use a seven-region network in which 60\% of vehicles are CAVs, and vary the demand, the state features and the reward profile.}

**Findings & results:** With regional congestion trends and the demand phase added to the state, training stabilises above light load, and an ablation study confirms that these features, rather than the change in normalisation, are responsible. Reward design strongly affects learnability of the policy: reward profiles that keep a weight on the vehicle's own travel time can be learned, whereas a purely network-wide travel-time reward fails, because one vehicle's action has a negligible effect on the total across thousands of vehicles. PPO is able to cope with noisy rewards and higher levels of demand better than DQN, with PPO remaining stable up to demand 1.30, whereas DQN diverges above 0.90. No configuration outperforms the Dijkstra-only baseline by more than its seed-to-seed variation in either the CAVs against the HDVs in their vicinity or for the network as a whole.

## 1.1 Motivation

\edt{Many drivers, each in their own vehicle, make a number of route choices as they go through the city.} Over thousands of trips in a peak period those route choices determine how the traffic loads the various parts of the network and \edt{how efficiently} the network's capacity is being used. \edt{That capacity is not a fixed amount. A city area carries traffic most efficiently at a certain number of vehicles, and less efficiently once that number is exceeded} \cite{Daganzo2007, Geroliminis2008}. The same level of demand can therefore be served quite well or quite badly depending on how the vehicles are distributed across the network.

\edt{Traffic operators can watch congestion build, but until now they have had no way to steer individual vehicles. Today, CAVs make this possible, as long as a controller can send a route instruction to a vehicle and the vehicle follows it.} \edt{A CAV here is a vehicle that is both connected, as it receives the controller's instructions, and autonomous, as it executes those instructions by itself.}

## 1.2 Problem statement

This thesis looks at whether a single learned routing policy (consulted by each and every CAV only at the point it reaches a region boundary) can achieve a lower travel time than that of shortest-path routing. The problem is difficult for two reasons. First, the control is partial: the HDVs are still free to route themselves and to react to the congestion that the CAV routing decisions create. Second, the population of CAVs that is being controlled is not static: CAVs are constantly entering and leaving the system throughout an episode, \edt{their number can rise from zero to thousands under heavy load and fall back to zero as the network clears, and each one asks the controller for a decision at an irregular time determined by its own progress through the network.}

## 1.3 Prior work and the research gap

\edt{Earlier work on route guidance in regional networks takes one of two approaches. Model-based regional guidance uses a prediction model of the network to decide how aggregate flows should split between regions \cite{yildirimoglu2018hierarchical, Hosseinzadeh2023}. It treats traffic as flows rather than as individual vehicles and assigns each flow a route or a split ratio, so no vehicle chooses its own next region at a boundary.}

\edt{The alternative is learned routing, in which each vehicle makes its own decision using a centralised or a shared policy. Existing work of this kind uses microscopic simulators that model individual road links, and each vehicle picks one whole route before it departs. In that setting, standard multi-agent algorithms rarely do better than human drivers, even after long and expensive training \cite{Akman2025}, and whether they converge depends on the reward \cite{Psarou2025}.}

We are not aware of a study that learns per-vehicle routing at the regional level based on MFD, with decisions made at region boundaries in a mixed-traffic setup, and compares it against shortest-path routing up to the demand at which congestion blocks the traffic in the network. This thesis addresses that research gap.

## 1.4 Approach

![The network used throughout this work: seven regions, six of them peripheral and arranged around a central hub, with dashed lines marking which regions adjoin. The arrows trace one vehicle's route $R_1 \to R_2 \to R_7 \to R_4$.](design/out/fig_network.pdf){width=78%}

The test network consists of seven regions with identical MFD parameters: six outer regions surrounding a central hub which is adjacent to all six. Speeds are derived from the MFD and there are no queues at region boundaries and no link level state so all congestion comes from the function mapping each region's vehicle count to its speed. Trips are tracked individually, as changing the route of one vehicle will change not only the regional counts but also the distance remaining on that trip and the boundaries it will cross next \cite{Batista2021a, Sirmatel2021}.

An **event** is any point at which the network state changes (a vehicle leaves a region, a vehicle completes its crossing of its current region). The simulation goes from one event to the next rather than ticking along a fixed clock. A single pass from the first departure until all vehicles have cleared is an **episode** and a training **run** is a series of episodes. At each boundary the HDV will take the deterministic shortest path according to the current regional costs, whereas the CAV asks the shared policy for the next adjacent region it should go to. Any regions that are unreachable or already visited are masked and the decision is only valid until the next boundary. \edt{An outer region adjoins three others and the central hub adjoins six, so a CAV chooses between at most three or six regions. Turning back into a region it has already visited is generally blocked by the no-revisit rule, unless no other region offers a path to the destination, in which case the rule is relaxed so that the vehicle is not trapped.}

The simulator was written in Python for this project. \edt{Because it is event-driven and skips the time between events, a training episode with more than 20,000 vehicles takes only a few minutes, so a policy can be trained over a thousand episodes in two to three days, without the cost of simulating every road link.} With the config and random seed fixed you can reproduce a run exactly and so individual decisions can be replayed and verified.

## 1.5 \edt{Contributions and hypotheses}

\edt{This thesis adds five things to existing work:}

1. **Formulation** of semi-Markov, boundary-triggered routing of individual CAVs on a regional MFD network in mixed traffic, \edt{with a single policy that every CAV consults, applied to a changing population of vehicles.}
2. **State design** using an augmented state vector which incorporates regional congestion trends and demand phase into the current occupancy state, with an ablation study to separate its effect from that of normalisation.
3. **Reward design finding** \edt{showing which per-vehicle rewards can be learned (those that keep a weight on the vehicle's own travel time)}, tested across behaviour profiles and two learning algorithms (DQN and PPO) on the same states and rewards.
4. **Evaluation** using two complementary measures, CAVs versus the HDVs in the same run and all vehicles against a Dijkstra-only baseline, across all demand levels up to the point at which the baseline breaks down.
5. \edt{\textbf{Hindsight optimisation} to find the system optimum, a simple and effective method that trains a routing policy of the same form, but one that is shown the future states of the network in a fixed scenario and rewarded directly on total system travel time. Its best episode is the closest approach to the system optimum on that scenario.}

Chapter 6 tests five hypotheses:

1. \edt{A learned CAV routing policy can reach lower travel times than the Dijkstra-only baseline as demand rises to and beyond the point where regions become congested.}
2. \edt{Additional feature engineering of the state observation vector makes the routing problem learnable at higher demand.}
3. \edt{Given the same state and reward, a policy-gradient algorithm (PPO) and a value-based algorithm (DQN) differ in how well they learn this routing problem.}
4. \edt{How a vehicle's reward weighs its own travel time against network-wide travel time affects whether the shared routing policy can be learned.}
5. \edt{A policy trained by hindsight optimisation, which sees the future of a fixed scenario and is rewarded on total system travel time, can find routing that is reliably faster than shortest path routing.}

## 1.6 \edt{Thesis outline}

Chapter 2 looks at MFD traffic dynamics, assignment under partial control, semi-Markov decisions, multi-agent RL and prior work on learned CAV routing. Chapter 3 is the MFD equations and notation. Chapter 4 is the event loop and the HDV/CAV routing rules. Chapter 5 is the learning problem, the reward profiles and the training algorithms. Chapter 6 is the results and the answers to the \edt{five} hypotheses.

\edt{Where an idea is complex, such as the design of the simulation environment, it is explained in plain terms, and the findings in Section 6.4 are organised under questions.}


```{=latex}
\setcounter{chapter}{1}
```

# Literature Review

## 2.1 Macroscopic fundamental diagrams (MFDs) and routing feedback

How much traffic a road network can \edt{clear depends on how its vehicles are distributed}, not only on how many there are. In a region with fairly even traffic, network throughput rises with accumulation until a certain point is reached and then starts to fall. A macroscopic fundamental diagram (MFD) takes this into account by relating the accumulation in a region to its production and outflow via a single \edt{well-defined curve for the whole region} rather than having a separate curve for each link \cite{Daganzo2007, Geroliminis2008}.

\edt{Four quantities describe a region in this kind of model. Its \textbf{accumulation} is the number of vehicles inside it at a given moment. Its \textbf{production} is the total distance those vehicles cover per second, and its \textbf{trip completion rate}, or outflow, is the number of vehicles that finish crossing it per second. Its average \textbf{speed} is production divided by accumulation. An MFD region is drawn so that congestion inside it is roughly uniform, and Geroliminis and Daganzo \cite{Geroliminis2008} confirmed with measured data from Yokohama that such a region has a well-defined curve of this kind. The curve is not monotonic\edt{, as the trip completion rate} rises with accumulation while vehicles can still move freely, peaks at the \textbf{critical accumulation}, and then falls as the region congests, until it reaches zero at the \textbf{jam accumulation}, where traffic stops. Speed falls steadily as accumulation grows. Past the critical accumulation, a region therefore holds more vehicles but completes fewer trips, and every extra vehicle lowers its throughput rather than raising it. The average time to cross a region follows from the same curve by Little's law, as accumulation divided by trip completion rate. For example, if 2,000 vehicles are in a region and 2 of them finish every second, each spends on average 1,000 seconds there. Chapter 3 gives the curve used in this thesis and its calibration.}

Controlling traffic through MFD regions \edt{in a traffic network} has practical advantages over controlling it link by link. A region's state is summarised by one number, its accumulation, so a controller \edt{needs only a few regional values, not} the flows and queues on every link. Regional route guidance and perimeter control build on this, steering how flows split between regions rather than between individual links \cite{yildirimoglu2018hierarchical, Sirmatel2021}. A regional model also runs far faster than a microscopic one, which matters for reinforcement learning, where a policy must be trained over many episodes. Akman et al. \cite{Akman2025} report this cost directly: training standard multi-agent algorithms on a microscopic simulator was lengthy and expensive on every network they tested.

\edt{In an MFD network, travel times depend on regional accumulations and on how vehicles are distributed across neighbouring regions, and route choices shift those counts and so change later travel times.} Static shortest-path search on a regional graph misses this feedback \cite{yildirimoglu2015equilibrium, yildirimoglu2018hierarchical}.

## 2.2 User equilibrium and system optimum

\edt{Two classical states frame what routing can achieve, both described by Wardrop \cite{Wardrop1952}. Wardrop's first principle assumes that each driver chooses the route that is quickest for them. Traffic then settles into a \textbf{user equilibrium (UE)}, in which all routes used between an origin and a destination take the same time and no unused route is quicker, so no driver can gain by switching route alone. This is the state drivers reach by themselves, whether with a navigation app or without one. Wardrop's second principle describes the \textbf{system optimum (SO)}, the state a central controller would aim for by routing every vehicle so that the total travel time of all trips is as small as possible.}

\edt{The two states differ because a driver who enters a congested region pays only their own crossing time, the \textbf{average travel time} of the region, but also slows down every vehicle already inside it. The sum of the two effects is the \textbf{marginal travel time}, the increase in the total time spent by all vehicles in the region when one more vehicle enters. The difference between marginal and average travel time is the \textbf{congestion externality}, the delay imposed on others only. A selfish driver compares routes by average travel time, whereas the system optimum is reached when routes are compared by marginal travel time, so that marginal travel times on all used routes are equal. Yildirimoglu et al. \cite{yildirimoglu2015equilibrium} use this condition to characterise the system optimum in MFD networks. Chapter 3 defines all three costs for our model.}

\edt{The difference in total travel time between the two states is the \textbf{UE-SO gap}, and it is the room any guidance scheme has to work in. In MFD networks it vanishes at low flow, where one more vehicle barely slows anyone down and \edt{the two states} coincide, and it grows as demand rises and regions pass the critical accumulation \cite{yildirimoglu2015equilibrium}. It is also bounded in theory: with linear travel costs, selfish routing costs at most 4/3 of the optimum \cite{Roughgarden2002}. Transport economics usually closes the gap by congestion pricing, which charges each driver the externality they cause so that the selfish choice and the system-optimal choice become the same. No tolls are used in this thesis. The gap is instead approached by routing the share of vehicles that accept a route, while the HDVs follow the shortest path on current travel times, which approximates the user equilibrium.}

In mixed traffic, neither classical state applies. The system optimum assumes every vehicle can be told where to go, and the user equilibrium assumes none can. With only CAVs guided, the achievable state lies somewhere between the two, and how close it comes to the system optimum depends on how many vehicles are guided and on how the HDVs respond to the congestion the CAVs create.

## 2.3 Semi-Markov decisions over variable travel times

Boundary decisions do not happen on a fixed clock. Once a CAV picks its next region, it has to cross that region before it decides again. How long that takes depends on the leg length drawn for the region and on speeds that change as other vehicles enter and leave. Because the time between decisions is random and reward builds up across the whole crossing, the problem is a semi-Markov decision process \cite{Bradtke1995}, a setting also covered by work on temporal abstraction in RL \cite{Sutton1999}.

This structure has two practical consequences for training. Rewards must be summed over the full leg rather than read at single points. Discounting should, in principle, follow the actual elapsed time, although a fixed per-leg discount is a reasonable approximation when trips take only a few legs (Chapter 5). A fixed-timestep formulation fits poorly, because decisions happen at boundaries, not on a regular tick while vehicles are mid-leg.

## 2.4 Multi-agent RL and credit assignment

A changing fleet is a further obstacle for multi-agent RL. Vehicles depart and arrive throughout an episode, so the number of active CAVs shifts constantly and can reach thousands under heavy demand. Many multi-agent algorithms assume a fixed list of agents. Value-decomposition methods such as QMIX \cite{Rashid2018} learn one value per agent and combine them into a joint value through a mixing network with one input slot per agent, so the set of agents must be fixed in advance. Centralised critics such as MADDPG \cite{Lowe2017} give each agent a critic that takes the observations and actions of all agents, so the critic's input grows with the number of agents. Neither works when the set of active agents keeps changing.

This work therefore adopts independent learning with a single shared policy: every CAV queries the same network from its own local observation, and the experience of every CAV trains it. Tan \cite{Tan1993} first showed that agents sharing a learned policy can learn faster than agents learning alone, and recent benchmarks find that independent learners remain competitive with centralised methods, with parameter sharing improving results in most cooperative tasks \cite{Papoudakis2020, Yu2021}. Sharing one policy removes the problem of a changing agent count.

It does not remove the problem of credit assignment, which limits what any reward can teach. When thousands of vehicles drive at once, one vehicle's route choice changes network-wide averages by a tiny amount compared with the variation caused by everyone else. Under a shared global reward, policy-gradient variance grows with the number of agents \cite{Lowe2017}, and each agent mostly observes the effect of other agents' choices rather than its own \cite{Sunehag2018}. Two remedies exist in the literature: difference rewards, which pay each agent its own marginal contribution to the system objective \cite{Wolpert2001}, and counterfactual baselines, which remove one agent's action while holding the others fixed \cite{Foerster2018}. Both need either the system outcome with and without each agent's action, or a centralised critic over all agents' actions, and neither is practical at every boundary crossing with thousands of vehicles in flight. A usable reward must therefore depend on something the vehicle's own choice visibly changes. Chapter 5 sets out the per-leg reward designed on this principle.

## 2.5 Learned CAV routing and the gap

\edt{Reinforcement learning has also been applied to CAV route choice directly. Akman et al. \cite{Akman2025} built the Urban Routing Benchmark, which evaluates multi-agent RL algorithms for CAV routing on 29 real-world networks with realistic demand, against human drivers and three baseline methods. Despite lengthy and costly training, the state-of-the-art algorithms rarely outperformed the human drivers, and several failed to converge. The authors conclude that current approaches struggle to scale to large urban networks. Psarou et al. \cite{Psarou2025} start from the earlier finding that selfish learning vehicles may need the equivalent of years of commuting to converge, and could destabilise traffic in the meantime. They add to each vehicle's reward a term built from the marginal cost of its route choice, that is, its effect on total travel time. This term aligns the agents' objectives, preserves the equilibria of the routing game and avoids oscillations. With it, the agents reach the system optimum faster and more reliably than the baseline algorithms, which fail to, on both a toy network and the real network of Saint-Arnoult. Both papers describe a CAV's reward as a combination of its own travel time and its effect on others, the structure our behaviour profiles also use (Chapter 5). Both, however, work at the level of individual road links in a microscopic simulator, with the cost noted in Section 2.1.}

Taken together, the literature leaves one combination untested: per-vehicle learned routing on a regional MFD network, with decisions triggered at boundary crossings, a single policy shared by a changing population of CAVs, and HDVs routing for themselves in the same network. This is the problem this thesis studies. The five hypotheses of Chapter 1 follow from the points above: whether a learned policy can beat shortest-path routing (Section 2.2), whether richer state information makes such a policy learnable (Section 2.1), which learning algorithm copes with the noise of a shared population (Section 2.4), which rewards survive credit assignment (Section 2.4), and whether a policy that sees the future can find a routing reliably faster than the user equilibrium (Section 2.2).


```{=latex}
\setcounter{chapter}{2}
```

# Regional Traffic Dynamics and Notation

All code referred to in this chapter and throughout the document, together with its documentation, is available at <https://github.com/sathyakumarnandakumar-cmyk/RG_MFD_Pub>.

This chapter sets out the traffic physics governing each region and fixes the notation used
\edt{in later chapters. The first part} develops the macroscopic
fundamental diagram itself: how a region's accumulation determines its production, its
trip completion rate, and the speed at which vehicles inside it travel, and how those quantities
yield the average, marginal and external travel-time costs that later chapters use to
evaluate a routing decision. The second defines the network state, meaning the accumulation
variables the simulation tracks in the backend and how the vehicle counts \edt{are} derived from them.

Every relation stated here is the one the simulation computes. The definitions and explanations of the symbols below are not repeated in Chapters 4 and 5, so readers may return to this chapter whenever needed. The notation for the reinforcement learning methodology is introduced in Chapter 5.


## Definitions and units

The following table gives a glossary of every quantity used in the environment description, in simple terms. We ask the reader to refer back to this table whenever
a symbol's meaning or unit is unclear.

| Symbol | Name | Plain-English meaning | Units |
|---|---|---|---|
| $n_i$ | **Accumulation** | A measure of the number of vehicles inside region $i$ | veh (a count) |
| $P_i$ | **Production** | Total distance covered per second by *every* vehicle currently in region $i$ combined. A measure of "distance throughput".  | veh·m/s |
| $g_i$ | **Trip completion rate (outflow)** | How many vehicles *finish crossing* region $i$ per second, $g_i = P_i/l_i$. | veh/s |
| $v_i$ | **Speed** | The average speed a vehicle moves at inside region $i$, given its current accumulation | m/s |
| $\rho_{CAV}$ | **CAV penetration** | Fraction of vehicles on the road that are CAVs (the rest are HDVs). $\rho_{HDV}$ = 1- $\rho_{CAV}$  | dimensionless, $[0,1]$ |
| $\varphi(\rho)$ | **Penetration factor** | A correction multiplier on the MFD polynomial capturing how the CAV/HDV mix changes \edt{trip completion rate} and speed at a fixed accumulation | dimensionless |
| $l_i$ | **Trip length** | The mean distance a vehicle travels while crossing region $i$ | \edt{metres} |
| \edt{$\delta_1 \ldots \delta_6$} | \edt{\textbf{MFD coefficients}} | \edt{The six fitted constants of the region's MFD: $\delta_1,\delta_2,\delta_3$ shape the trip completion rate curve against accumulation, and $\delta_4,\delta_5,\delta_6$ correct it for the CAV/HDV mix.} | \edt{dimensionless} |
| \edt{$T_i$ (also written $t_i$)} | **Average travel time** | \edt{The expected time for one vehicle to cross region $i$ at its current accumulation, $T_i = n_i/g_i$. The two symbols denote the same quantity and are used interchangeably.} | seconds |
| $T_i^{marg}$ (written $c^{marg}$ in Chapter 5) | **Marginal travel time** | The *total extra delay, summed across every vehicle*, caused by one additional vehicle entering region $i$\edt{, which is} bigger than the average because that vehicle also slows everyone else down | seconds |
| $c^{ext}_i$ | **Congestion externality** | Marginal - average: the delay imposed on *other* vehicles only, without factoring in the entering vehicle's own delay | seconds |
| $g_i' = dg_i/dn$ | **Throughput derivative** | How much the trip completion rate changes per one additional vehicle. Positive below critical accumulation (one more vehicle still helps total throughput); negative above it (increase in accumulation creates a drag in throughput) | veh/s per veh, i.e. $1/\text{s}$ |
| $n_{cr}$ | **Critical accumulation** | The accumulation at which trip completion rate peaks\edt{, that is, the region's accumulation at} maximum throughput | veh |
| $n_{max}$ / $n_{jam}$ | **Jam accumulation** | The accumulation at which the region is fully gridlocked and velocity drops to zero| veh |
| $C_{ij}$ | **Edge / route cost** | \edt{The time it is expected to cost to travel from region $i$ to region $j$. The router adds these up along each candidate route and takes the cheapest} | seconds |
| \edt{$\text{active}(t)$} | \edt{\textbf{Active set}} | \edt{The vehicles on the road at time $t$: departed by $t$, not yet arrived. Vehicles still queued to depart, and vehicles that have finished, are excluded} | \edt{veh} |
| \edt{$r(k,t)$} | \edt{\textbf{Vehicle location}} | \edt{The region vehicle $k$ occupies at time $t$} | \edt{a region index} |

: Symbols, names and units used in Chapters 3 to 6.


We will use the following terms at a later stage



## 3.1 The macroscopic fundamental diagram of a region

\edt{A \textbf{macroscopic fundamental diagram (MFD)} denotes the relationship between the number
of vehicles circulating in a region and the rate at which that region completes trips of a
road network. An MFD gives values at the level of a whole neighbourhood rather than a single
road link, where an MFD region is chosen so that congestion inside it is roughly uniform. This
property has been confirmed on measured urban data \cite{Geroliminis2008}. The curve is not
monotonic. Network throughput rises with accumulation only while vehicles can still move, it
then peaks, and then falls as the region congests, so a region can hold more vehicles and
complete fewer trips.}

A region in this model has no internal geometry. It is characterised entirely by how many
vehicles it currently holds, and every other quantity \edt{is derived from this number} through
the relations below\edt{, in a fixed order}: accumulation determines production,
production determines trip completion rate, and trip completion rate determines both the speed vehicles
travel at and the time they take to cross. The final three subsections turn those into the
cost quantities a routing decision is judged against, namely the average travel time a
vehicle experiences, the marginal travel time it imposes, and the difference between them.

![The shape of the relations developed in this section. Trip completion rate rises with accumulation to a peak and then falls away, while speed declines throughout. The two marked accumulations, $n_{cr}$ and $n_{max}$, are \edt{the critical and jam accumulations}.](design/out/fig_mfd.pdf){width=88%}

### 3.1.1 Productivity $P_i$

Productivity $P_i$ is the distance-throughput of region $i$: total distance covered per second by every vehicle currently in region $i$ combined (it can be thought of as accumulation-weighted average
speed, scaled to a distance rate via trip length $l_i$):

$$P_i(n) = l_i\left(\delta_1 n^3 + \delta_2 n^2 + \delta_3 n\right)\varphi(\rho)\,\sigma \tag{Eq. 3.1}$$

Without the penetration factor ($\varphi \equiv 1$):

$$P_i(n)\big|_{\varphi = 1} = l_i\left(\delta_1 n^3 + \delta_2 n^2 + \delta_3 n\right)$$

with the penetration factor $\varphi$ defined next.

### 3.1.2 Penetration factor $\varphi$, the $(\delta_4 + \delta_5\rho + \delta_6\rho^2)$ term of Eq. 3.1

$$\varphi(\rho) = \delta_4 + \delta_5\,\tilde\rho + \delta_6\,\tilde\rho^{\,2}, \qquad \tilde\rho = \max(\rho_{CAV},\ \rho_{min}), \quad \rho_{min} = 0.10 \tag{Eq. 3.2}$$

The lower limit exists because as $\rho \to 0$ the polynomial collapses to
$\varphi \approx \delta_4 \approx 3.6\times10^{-10}$, which drives every region to the speed
floor regardless of accumulation. This is needed for \edt{runs with no CAVs}.

At the default CAV_penetration value, $\rho_{CAV} = 0.6$: $\varphi = 0.0270$.

### 3.1.3 Trip completion rate $g_i$

$$g_i(n) = \max\!\Big(\ \max\big(\left(\delta_1 n^3 + \delta_2 n^2 + \delta_3 n\right)\varphi(\rho)\,\sigma,\ \ 0\big),\ \ 10^{-6}\Big) \tag{Eq. 3.3}$$

The cubic $\delta_1 n^3 + \delta_2 n^2 + \delta_3 n$ is the raw shape of the MFD, fitted to
observed data: it is the curve that best matches how many vehicles clear a region as a function
of how many are in it. At $n=0$ it is $0$; it rises as accumulation increases and more vehicles
are available to finish; then it falls as congestion sets in and vehicles move too slowly to
complete trips even though there are plenty of them.

\edt{The leading coefficient is \textbf{negative} ($\delta_1 = -1.04\times10^{-8}$), which is what
makes that fall happen: the curve turns over at $n_{cr}$ and continues down until it reaches
zero. Trip completion rate therefore cannot grow without bound with accumulation. On this
calibration the positive root of the cubic is $n = 5536$ vehicles, which is \edt{the} jam
accumulation $n_{max}$\edt{, at which the region completes no trips}. The
inner $\max(\cdot, 0)$ is only a lower limit for $n > n_{max}$, where the fitted polynomial
would otherwise return a negative \edt{trip completion rate}.}

\edt{Trip completion rate $g_i(n)$ is \textbf{cubic} in accumulation, while speed $v_i(n)$ is
\textbf{quadratic} (Eq. 3.5). They are not independent: $g_i = n\, v_i / l_i$, so the extra power
of $n$ in $g_i$ is \edt{the number of vehicles travelling}. Speed falls
monotonically from the first vehicle onward; \edt{trip completion rate} does not, because for a while the rising
vehicle count still outweighs the falling speed.} $\varphi(\rho)$ rescales that curve for the current
CAV mix, and $\sigma = 1/3600$ converts the time units from hours to seconds. The two
lower limits, \edt{the} inner $\max(\cdot,0)$ \edt{and the} outer $10^{-6}$, exist to \edt{stop the simulation producing}
negative \edt{trip completion rate} \edt{or dividing} by zero.


**Why there is no $l_i$ here:** 
Productivity is $P_i = l_i(\delta_1 n^3 + \delta_2 n^2 + \delta_3 n)\varphi$ and
trip completion rate is $g_i = P_i / l_i$. **The $l_i$ cancels in that division:**

$$g_i = \frac{P_i}{l_i} = \left(\delta_1 n^3 + \delta_2 n^2 + \delta_3 n\right)\varphi\,\sigma$$

The \edt{model} never needs $P_i$ (productivity)\edt{, because every later use of the MFD depends on the} **trip completion rate** $g_i$.

**Note:** Trip completion rate and speed describe the same physics, so they must
satisfy the equation $g_i = n\,v_i / l_i$ (trip completion rate = accumulation × speed ÷ trip length).

### 3.1.4 Slope of the trip completion rate, $dg_i/dn$

$$g_i'(n) = \frac{dg_i}{dn} = \left(3\delta_1 n^2 + 2\delta_2 n + \delta_3\right)\varphi(\rho)\,\sigma \tag{Eq. 3.4}$$

\edt{This is the ordinary calculus derivative of the trip completion rate cubic: it answers "if one more vehicle is added to this region, how much does the trip completion rate go up (or down) by?" It is a standard power-rule differentiation of $\delta_1 n^3 + \delta_2 n^2 + \delta_3 n$, giving $3\delta_1 n^2 + 2\delta_2 n + \delta_3$; $\varphi(\rho)$ and $\sigma$ carry over unchanged because they do not depend on $n$.}

\edt{The derivative is \textbf{positive below $n_{cr}$}, where one more vehicle still raises throughput because the region has spare capacity, and \textbf{negative above it}, where one more vehicle actively \emph{reduces} throughput because the trip completion rate curve has already turned over. \edt{The critical accumulation $n_{cr}$ is where this derivative changes sign.} It is evaluated at the saturated accumulation $\tilde n = \min(n, n_{clamp})$, defined with the congestion externality.}

### 3.1.5 Speed $v_i$
\edt{Evaluated} once per event per region:

$$v_i(n) = \max\Big(\ l_i\left(\delta_1 n^2 + \delta_2 n + \delta_3\right)\varphi(\rho)\,\sigma,\ \ v_{min}\Big), \qquad v_{min} = 0.5\ \text{m/s} \tag{Eq. 3.5}$$

This follows from $v = P_i/n_i$ (Eq. 3.1 divided by $n$), with the $n$ divided out
analytically. Note this expression carries
$l_i$ and $\sigma$, so it is in m/s directly.

### 3.1.6 Regional travel time $T_i$

\edt{Written} $T_i$:

$$T_i(n) = \frac{n}{g_i(n)} \tag{Eq. 3.6}$$

\edt{If} $g_i$ vehicles get through their crossing every second, and there are $n$ vehicles in the region at the moment, then on average each of those $n$ vehicles must have taken $n/g_i$ seconds to get through\edt{:} accumulation divided by throughput gives a time. This is
\edt{the} trip completion rate turned around: \edt{the trip completion rate} counts how many vehicles finish each
second, while this asks how long one vehicle takes to finish.

### 3.1.7 Marginal (system-optimal) travel time

$$T_i^{marg}(n) = \frac{d}{dn}\Big[n \cdot T_i(n)\Big] = n\left(\frac{2}{g_i} - \frac{n\,g_i'}{g_i^{\,2}}\right) \tag{Eq. 3.7}$$

**What is $n \cdot T_i(n)$?** $T_i(n)$ is the *average* time one vehicle spends
crossing the region. If there are $n$ vehicles in the region, each spending roughly
$T_i(n)$ seconds there, then $n \cdot T_i(n)$ is the **total time being spent by everyone in
the region right now**. It is \edt{the total cost} for the region at this instant, not a per-vehicle \edt{quantity}.

**Why do we differentiate it?** The question we want answered is *if one more vehicle enters this region,
how much does the total time spent there increase?* A derivative with respect to $n$ answers that question: it is the rate at which the region's total time grows as accumulation rises by one
vehicle. That growth includes **two different effects**: the new vehicle's own
travel time, *and* the fact that its arrival makes the region slightly more congested, which
slows down every vehicle already in that particular region. The **average** travel time $T_i(n)$ only
captures the first effect; the derivative of the total captures both\edt{, which} is why it is
called the *marginal* cost, and why it is \edt{never smaller than} the average.

 $n \cdot T_i(n)$ is a product of two
functions of $n$, so:

$$\frac{d}{dn}\big[n \cdot T_i(n)\big] = T_i(n) + n \cdot \frac{dT_i}{dn}$$

\edt{The first term is the average cost; the second is $n$ times the increase in that average per additional vehicle.} 
Substituting
$T_i = n/g_i$ and differentiating that with the quotient rule gives
$\frac{dT_i}{dn} = \frac{1}{g_i} - \frac{n\,g_i'}{g_i^2}$, and combining both terms above
\edt{gives} Eq. 3.7.

The marginal cost is *strictly larger* than the average only when $dT_i/dn > 0$, i.e. when the addition of vehicles is *already* making the average trip slower (congestion setting in). \edt{An additional vehicle therefore delays not only itself but also each of the $n$ vehicles already in the region, and the marginal cost is the average delay plus $n$ times that extra slowdown. This is the \edt{well-known} distinction between user-optimal and system-optimal routing: a selfish driver feels only the average cost $T_i$, whereas a system-optimal planner charges each vehicle its marginal cost $T_i^{marg}$, the total delay its presence adds to the region.}

\edt{\textbf{The UE-SO gap.} These two costs correspond to two states the network can settle into. If every driver takes the route that is cheapest for them, traffic moves toward a \textbf{user equilibrium (UE)}, in which no driver can shorten their own trip by switching route alone: any faster alternative would already have attracted enough traffic to lose its advantage. This is Wardrop's first principle, and it describes drivers who each choose for themselves. If the same demand is routed on \emph{marginal} cost instead, the network reaches the \textbf{system optimum (SO)}, where the total time spent by all vehicles is as small as the network allows. At low load the two states are the same, because one more vehicle hardly slows anyone else down. As regions fill, they drift apart. The \textbf{UE-SO gap} is the difference in total travel time between them: the time lost because each driver minimises their own trip rather than the total.}

\edt{The size of the gap depends on load, through the same $dT_i/dn$ that separates marginal from average cost. In free flow the two costs are equal, an extra trip \edt{does not measurably slow any other vehicle}, and coordination has little to gain. As regions fill, the gap widens, most quickly beyond $n_{cr}$, where each additional vehicle reduces throughput instead of adding to it. In transport economics the gap is usually closed by congestion pricing: if each driver is charged the externality they cause, the selfish choice and the system-optimal choice become the same. Rather than charging each driver for the delay they impose on others, this thesis tries to narrow the gap by guiding the CAVs, the only vehicles that accept a route. This is why $T_i^{marg}$ and $c^{ext}_i$ appear again in Chapter 5, as possible reward terms.}


### 3.1.8 Congestion externality

First the **marginal-to-average ratio**, which is
scale-free:

$$r_i(n) = \frac{T_i^{marg}}{T_i} = 2 - \frac{n\,g_i'(n)}{g_i(n)} \tag{Eq. 3.8}$$

\edt{Dividing the marginal cost by the average cost and cancelling a factor of $n/g_i$ gives this ratio. The constant 2 comes from the total delay $n \cdot T_i(n)$. When $T_i$ grows roughly in proportion to $n$, the total delay grows like $n^2$, and its derivative is $2n$. The $g_i'$ term corrects for the change in the trip completion rate $g_i$ with $n$.}

\edt{At very low accumulation ($n \to 0$), $r_i \to 1$. One more vehicle barely slows anyone down, so the marginal and average costs are equal and there is no externality. As accumulation approaches $n_{cr}$ and beyond, $g_i'$ shrinks and then turns negative, and $r_i$ rises above 1 and keeps rising, so the externality grows with congestion. A ratio $r_i = 1$ means that a vehicle imposes no cost beyond its own delay; $r_i = 2$ means that its presence costs the network, in total, twice what it costs the vehicle itself.}

\edt{Multiplying the excess of the ratio over 1 by the region's travel time gives the externality in seconds:}

$$c^{ext}_i = \max\big(r_i(\tilde n) - 1,\ 0\big)\cdot \frac{l_i}{v_i(\tilde n)}, \qquad \tilde n = \min\big(n,\ n_{clamp}\big) \tag{Eq. 3.9}$$

**Saturation.** \edt{Above a certain accumulation, the MFD speed falls below the floor $v_{min}$, and every vehicle in the region moves at $v_{min}$. This accumulation, $n_{clamp}$, is the largest at which the MFD speed is still above the floor. It has no closed form, so the simulation finds it by bisection:}

$$n_{clamp} = \sup\Big\{\,n : l_i(\delta_1 n^2 + \delta_2 n + \delta_3)\varphi\sigma > v_{min}\Big\} \tag{Eq. 3.10}$$

\edt{Beyond $n_{clamp}$, speed is held at the floor rather than read from the curve, so the derivative of the polynomial no longer describes the region. Evaluating the externality there would divide by the floor speed and give very large values, so the externality is held constant above $n_{clamp}$. The same limit applies to the throughput derivative.}

\edt{The externality $c^{ext} = T^{marg} - T$ is therefore the delay imposed on other vehicles. The vehicle's own travel time is left out so that it is not counted twice alongside the $a \cdot t_{own}$ term of the reward.}

### 3.1.9 Critical and jam accumulation $n_{cr}$, $n_{max}$

\edt{Critical accumulation is where the trip completion rate curve stops rising, so it is found by
setting the throughput derivative to zero and keeping the positive root:}

$$\edt{\frac{dg_i}{dn} \;=\; 3\delta_1 n^2 + 2\delta_2 n + \delta_3 \;=\; 0,
\qquad n_{cr} = \min\{\,n > 0 \;:\; 3\delta_1 n^2 + 2\delta_2 n + \delta_3 = 0\,\}}
\tag{Eq. 3.11}$$

\edt{The quantity set to zero is the throughput derivative $g_i'(n)$, with
$\varphi\sigma$ factored out since it is always positive and does not move the zero crossing.
So $n_{cr}$ is, by definition, the accumulation at which that derivative becomes zero: the peak of
the trip completion rate curve, the tipping point past which one more vehicle lowers throughput
rather than raising it.}

\edt{Being a quadratic, this has two roots, and in principle one could be a maximum and the
other a minimum. \edt{On this calibration only one of the two roots is valid.} The two roots are
$n = -55376$ and $n = 2811.9$, and the first is a negative accumulation, which has no physical
meaning: a region cannot hold fewer than zero vehicles. Restricting to $n > 0$ therefore leaves
exactly one candidate, and it is the maximum.}

\edt{Taking the value the source quotes rounded, $n_{cr} = 2812$ vehicles. Jam accumulation is
$n_{max} = 5536$ vehicles (\texttt{njamMFD}), read from the MFD profile inputs; it is the declared
point beyond which the region is treated as fully gridlocked and instantaneous speed is zero.
That figure is not independent of the polynomial, however: $5536$ is also where
the fitted cubic itself reaches zero, so the declared jam point and the curve's own root agree.}



### 3.1.10 MFD coefficients of the 7-region network

\edt{These coefficients were not fitted for this thesis. They were taken from a fit to this MFD network already made by senior researchers working in the same lab.} The following MFD coefficients model `7_region_new` (`networks/7_region_new/mfd.json`):

| | $\delta_1$ | $\delta_2$ | $\delta_3$ | $\delta_4$ | $\delta_5$ | $\delta_6$ | $l$ | $n_{max}$ |
|---|---|---|---|---|---|---|---|---|
| value | $-1.04\!\times\!10^{-8}$ | $-8.2\!\times\!10^{-4}$ | $4.8582$ | $0.9137$ | $0.2903$ | $1.1084$ | $4910$ m | $5536$ |

: MFD coefficients of the 7-region network.

$\delta_1,\delta_2,\delta_3$ shape the trip completion rate curve vs. accumulation;
$\delta_4,\delta_5,\delta_6$ correct that curve for the CAV/HDV mix.

Other constants: $v_{min} = 0.5$ m/s, $\rho_{min} = 0.10$, $\sigma = 1/3600$ (on `7_region_new`;
the earlier `7_region` calibration instead used $\sigma = 1.0$), $\beta = 0.5$, $\varepsilon_{arr} = 10^{-4}$. \edt{The logit sensitivity $\beta$ is listed for completeness only: in every experiment reported here, HDVs route by deterministic shortest path (Dijkstra), so $\beta$ plays no role. Deterministic routing was chosen deliberately, because the experiments test whether a learned policy can beat shortest-path routing, which approximates the user equilibrium.}


## 3.2 Network state

\edt{The MFD physics} treats a region's accumulation as given. This section defines \edt{how that quantity is obtained} in the simulation, how it is initialised, and how it decomposes by vehicle class.
Two kinds of accumulation are in play: the true count $n_i$, used both for HDV routing and
for all reward-side quantities, and the per-class counts that let costs be attributed to one
fleet or another.

\edt{The network these accumulations are counted over is the seven-region topology of Figure 1.1: six peripheral regions around a central hub, each peripheral region adjoining the hub and its two neighbours.}

### 3.2.1 Accumulation $n_i$

\edt{The accumulation is derived by counting the number of vehicles inside region $i$:}

$$n_i(t) = \big|\{\,k \in \text{active}(t)\ :\ r(k,t) = i\,\}\big| \tag{Eq. 3.12}$$

\edt{Here $\text{active}(t)$ is the set of vehicles on the road at time $t$: those that have
departed by $t$ and have not yet reached their destination. $r(k,t)$ is the region vehicle $k$
occupies at that instant. So \edt{$n_i(t)$} counts the vehicles currently on the road whose region is $i$.}

\edt{Initialised from the steady-state initial accumulation $x^{init}_{ij}$ supplied with the network, the number of vehicles in region $i$ bound for region $j$ at $t = 0$, scaled by $s_{init}$ ($0.3$ in all experiments reported here):}

$$n_i(0) = s_{init} \cdot \sum_j x^{init}_{ij} \tag{Eq. 3.13}$$

For region $i$, the total accumulation is $n_i(t) = n_i^{\rm{CAV}}(t) + n_i^{\rm{HDV}}(t)$,
where $n_i^{\rm{CAV}}(t)$ and $n_i^{\rm{HDV}}(t)$ are the numbers of CAVs and HDVs circulating
in region \edt{$i$} at time $t$.

### 3.2.2 Vehicle counts by class $n_{CAV}$, $n_{HDV}$

\edt{Counted} per region and network-wide, across three classes \edt{(}two CAV
fleets and HDVs\edt{)}:

$$n^{\mathrm{fleet1}}_i = \big|\{\,k \in \text{active}(t) : \mathrm{fleet}_k = 1,\ r(k,t) = i\,\}\big|, \qquad n^{\mathrm{fleet2}}_i,\ \ n^{\mathrm{hdv}}_i \ \text{likewise} \tag{Eq. 3.14}$$

$$N^{\mathrm{fleet1}} = \sum_i n^{\mathrm{fleet1}}_i, \qquad N^{\mathrm{fleet2}} = \sum_i n^{\mathrm{fleet2}}_i, \qquad N^{\mathrm{hdv}} = \sum_i n^{\mathrm{hdv}}_i \tag{Eq. 3.15}$$

> \edt{\textbf{These counts do not feed the penetration factor.} The factor $\varphi(\rho)$ uses the \edt{CAV penetration rate} $\rho_{CAV}$ set in the experiment configuration, which is the same in every region and fixed for the whole run, not the CAV share $n^{\mathrm{fleet1}}_i / n_i$ measured in each region. The class counts are used to compute costs by class in the reward.}



```{=latex}
\setcounter{chapter}{3}
```

# The Event Cycle

*The following section is an overview of how our simulation model works. It details how the simulation advances time, advances a vehicle along time, and which stage each vehicle is in.*

\edt{*This mechanism is the \textbf{event loop}: it jumps the clock to the next
moment something happens, moves every vehicle, and asks any vehicle at a region boundary
where it goes next.*}


This chapter does not cover how the CAV policy is trained. This is an overview of how **HDVs and CAVs are routed while the simulation runs, or during inference when a trained policy is available**. Chapter 5 details how the policy is trained. 

All parameters in this section are explained with the MFD coefficients of the region `networks/7_region_new`,
 Its critical accumulation is $n_{cr} = 2812$ veh, its
jam accumulation $n_{max} = 5536$ veh, and its mean trip length $l_i = 4910$ m.


## 4.1 Vehicle data: where do the vehicles come from?

\edt{The event loop, which advances simulation time, needs a population of
vehicles before it can start.} They arrive in three separate cohorts
that differ **only in when they enter**. Once running, all three are handled by \edt{the same code}.

| Cohort | Departure time | What it represents |
|---|---|---|
| Initial accumulation | $t^{dep} = 0$ | Traffic already on the road when the episode begins |
| Internal trips | $t^{dep} > 0$ | Trips beginning and ending in the same region |
| External trips | $t^{dep} > 0$ | Trips crossing at least one region boundary |

: The three vehicle cohorts.

\edt{Intra-region (internal) trips never cross a region boundary; they travel down their remaining distance and arrive without a routing decision, but they still add to accumulation and therefore to congestion. Initial-accumulation vehicles are already on the road at $t = 0$ and are mostly destined for other regions, so at each boundary they are routed in the same way as external trips.}

New demand is released over a loading window $T_q = 3600$ s, shaped by a trapezoid
$\pi(\tau)$ that ramps up, plateaus, and ramps down, and scaled globally by $\kappa$:

$$q^{OD}_{new}(\tau_s) = q^{OD}_{max} \cdot \pi(\tau_s) \cdot \kappa \tag{Eq. 4.1}$$

\edt{Both $q$ terms are \textbf{demand rates}, in vehicles per second, for one origin-destination
pair $OD$. $q^{OD}_{max}$ is the peak rate that pair ever reaches, read from the demand matrix
of the network file and fixed for the whole run. $q^{OD}_{new}(\tau_s)$ is the rate
released during time bin $s$, obtained from the peak by applying the trapezoid $\pi(\tau_s)$,
which is $0$ before loading begins, rises to $1$ across the plateau and falls back to $0$, and
then the global scale $\kappa$, the parameter that sets how congested each experiment in
Chapter 6 is.}

\edt{The loading window is divided into equal bins of width $\Delta\tau = 300$ s, indexed by
$s = 0, 1, 2, \ldots$, with $\tau_s$ the start time of bin $s$. Each bin is then converted into
a whole number of vehicles, each given a departure time drawn uniformly inside its own bin:}

$$N^{OD}_s = \Big\lceil \Delta\tau \cdot q^{OD}_{new}(\tau_s) \Big\rceil, \qquad t^{dep} \sim \mathcal{U}\big(\tau_s,\ \tau_s + \Delta\tau\big), \quad \Delta\tau = 300\ \text{s} \tag{Eq. 4.2}$$

![\edt{The demand loading profile. The dashed line is the trapezoid $\pi(\tau)$ the model defines: demand ramps linearly to its peak over the first quarter of the loading window, holds at $q^{OD}_{max}$ through the middle half, then ramps back to zero. The stepped line is what the simulator uses, since $\pi$ is evaluated once per 300-second bin at that bin's midpoint (marked), giving one constant release rate per bin rather than a continuously varying one. The vertical axis is the fraction of each pair's peak rate, so the same shape applies to every OD pair and the global scale $\kappa$ multiplies it uniformly.}](design/out/fig_demand_profile.pdf){width=95%}

\edt{Reading Eq. 4.2: a rate in vehicles per second multiplied by a bin width in seconds gives
the vehicle count for that bin, rounded up to a whole vehicle; each of those vehicles is then
given a departure time drawn uniformly between the start of the bin and its end, so departures
are spread through the bin rather than released together at its start.}

\edt{The bin width is written $\Delta\tau$, not $\Delta t$. $\Delta t$ denotes the interval
between two consecutive simulation events, variable and set by the traffic;
$\Delta\tau$ is a fixed property of how demand is discretised before the run begins.}

\edt{The upper limit is applied to each bin after scaling}, so \edt{changing $\kappa$ does not change the}
\edt{vehicle count in exact proportion}. The rounding-up error accumulates across bins, and the realised count ends up
slightly above $\kappa \times$ the nominal total.

Finally, every vehicle is assigned a class, HDV or CAV, **once, at creation, randomly, with
counts set by $\rho_{CAV}$**\edt{, and keeps it for the whole episode}:

$$N_{CAV} = \mathrm{round}(N \rho_{CAV}), \quad N_{fleet1} = \mathrm{round}(N_{CAV}\, \phi_1), \quad N_{fleet2} = N_{CAV} - N_{fleet1}, \quad N_{HDV} = N - N_{CAV} \tag{Eq. 4.3}$$

There is also an optional feature to split the CAV population into two fleets, whose sizes
are set by the fleet-1/fleet-2 ratio $\phi_1$. \edt{All experiments in Chapter 6 use a single CAV fleet.}

This single assignment is what \edt{the HDV and CAV route choices} branch on for the rest of the vehicle's life.


## 4.2 The event loop

### 4.2.1 Why the simulation is event-driven

\edt{A fixed-timestep simulation advances the clock by a constant increment and re-evaluates
everything at each tick, whether or not anything has changed. An event-driven simulation
instead finds when the next event happens and jumps directly there.}

\edt{An \textbf{event} is any moment at which the state of the network can change. Everything
between events is already determined, so these are the only moments the simulation needs to
visit.} \edt{In this model an event is one of two occurrences:}

1. **A vehicle departs**, entering the network \edt{from the pending-departure list, the set of
   vehicles whose departure time has not yet arrived}.
2. **A vehicle finishes crossing a region** \edt{and must be instructed} where to go next. This includes the inter-region decision when a vehicle finishes one region and enters the next. 

\edt{An \textbf{episode} is one complete pass of this loop, from the first departure until the
network has cleared; a \textbf{run} is the sequence of episodes over which a policy is trained.}

The network state does not change between those two kinds of event. Vehicle positions
in the interim are completely determined: a vehicle is travelling at its region's current speed, and that
speed only changes when the region's population changes, which only happens on an event.
There is nothing to compute at intermediate times, so the simulator does not stop
there. In our 7-region MFD network, a vehicle when routed through a trained policy traverses about 3-4 regions or 2-3 boundary crossings between origin and destination. 

![A trip as the simulation represents it: a sequence of legs separated by boundary crossings, with a routing decision taken at each crossing.](design/out/fig_route_cost.pdf){width=92%}

\edt{The interval $\Delta t$ between two consecutive events is therefore not constant. It ranges from} microseconds when thousands of vehicles
are in flight \edt{to} minutes when the network is nearly drained. 

\edt{Any quantity accumulated per event therefore covers a variable amount of simulated time, so anything that should be proportional to time must be weighted by $\Delta t$.}

\edt{Decisions are also irregularly spaced, because a vehicle chooses its next region only when it finishes a leg, and legs take different amounts of time.}

\edt{Each decision is a single choice of which adjoining region to enter next. It sets no speed, departure time or route beyond the next boundary. An outer region adjoins three others and the central hub adjoins six, so a vehicle has three to six candidates, minus any from which the destination cannot be reached and, when the revisit rule is enabled, any already visited. Turning back the way it came is therefore allowed unless that rule \edt{blocks} it.} 

Therefore, this model makes the
control problem a *semi-Markov* process- the time interval
between decisions is itself a random variable. That distinction is stressed again for the policy learning
process in the Reinforcement learning algorithm.

### 4.2.2 What the simulation remembers between events

**Per-vehicle data.** \edt{The state of all vehicles is held as one vector per
attribute, each of length equal to the number of vehicles, so that vehicle $k$'s current region,
remaining distance and next arrival time are entry $k$ of three separate vectors rather than
three fields of one record. The fields are listed below. Holding them this way lets the loop
advance every vehicle in a single arithmetic operation per attribute, which is what makes an
episode of tens of thousands of vehicles run in seconds.}

| Field in Codebase | Symbol | Meaning |
|---|---|---|
| `current_sub` | $r(k)$ | Which region the vehicle occupies right now |
| `remain` | $L^{rem}_k$ | Metres still to cover on the current leg |
| `t_next_arr` | $t^{arr}_k$ | The time this vehicle will finish its current leg |
| `curr_leg_idx` | $j_k$ | Which leg of its route it is currently on |
| `dynamic_sub_seq` | $\Theta_k$ | The remaining planned sequence of regions |
| `is_cav`, `is_fleet1` | $c_k, \mathrm{fleet}_k \in \{0,1\}$ | Class flags, fixed at creation |

: Per-vehicle state carried between events.

**Per-network data.**

| Field | Symbol | Meaning |
|---|---|---|
| region counts | $n_i$ | Accumulation, i.e. how many vehicles are in region $i$ |
| region speeds | $v_i$ | Speed inside region $i$ at that accumulation |
| departure queues | — | Not-yet-departed vehicles, sorted by $t^{dep}$ |

: Network state carried between events.

Note on what is **not** stored: there is no queue at region boundaries, no per-link state, and
no explicit flow variable\edt{, as used in accumulation-based MFD models that advance on a fixed time step}. The entire congestion mechanism is the single mapping from
$n_i \rightarrow v_i$.

### 4.2.3 The full event cycle

\edt{Algorithm 4.1 is the pseudocode of the whole simulation. We provide the explanation right after it.}

```pseudocode
caption: The main event loop
label: alg:event-loop
---
Input: vehicle set $\mathcal{K}$ with departure times $\{t_k^{dep}\}$, region adjacency $\mathrm{Adj}$

Notation: $r(k)$ is the region vehicle $k$ currently occupies; $\mathit{Dest}_k$ its destination

Initialize:
    $t \gets 0$
    $\mathit{Active} \gets \{k : t_k^{dep} = 0\}$    \Comment{already on the road at $t = 0$}
    $\mathit{Waiting} \gets \{k : t_k^{dep} > 0\}$    \Comment{awaiting departure}

Loop while $\mathit{Active} \cup \mathit{Waiting} \neq \emptyset$:

    ## Step 1: find the next event
    $t' \gets \min\big( \min_{k \in \mathit{Active}} t_k^{arr},\ \min_{k \in \mathit{Waiting}} t_k^{dep} \big)$
    if $t' = \infty$ then exit    \Comment{nothing left to do}
    $\Delta t \gets t' - t$;\quad $t \gets t'$    \Comment{Eq. 4.4}

    ## Step 2: admit vehicles whose departure time has come
    move every $k \in \mathit{Waiting}$ with $t_k^{dep} = t$ into $\mathit{Active}$

    ## Step 3: measure the network
    for each region $i$:
        $n_i \gets \big|\{k \in \mathit{Active} : r(k) = i\}\big|$    \Comment{Eq. 3.12}
        $v_i \gets v_i(n_i)$    \Comment{Eq. 3.5, floored at $v_{min}$}

    ## Step 4: move everyone, re-route whoever arrived
    for each $k \in \mathit{Active}$:
        $L_k^{rem} \gets \max\big( 0,\ L_k^{rem} - v_{r(k)}\,\Delta t \big)$    \Comment{Eq. 4.5}

    for each $k \in \mathit{Active}$ with $L_k^{rem} \leq \varepsilon_{arr}$:    \Comment{its leg just ended}
        if $r(k) = \mathit{Dest}_k$:
            mark $k$ arrived
        else:
            $F \gets$ the feasible set: regions adjacent to $r(k)$ from which
                $\mathit{Dest}_k$ is still reachable    \Comment{Eq. 4.10}
            if $k$ is an HDV:
                $\Theta \gets$ shortest path $r(k) \rightsquigarrow \mathit{Dest}_k$ over $C$,
                    written as the sequence of regions it visits
                $h \gets \Theta_2$    \Comment{the second region along $\Theta$}
            else:
                $h \gets \arg\max_{a \in F}\, Q(S_k, a)$    \Comment{Eq. 4.13}
            $r(k) \gets h$;\quad append $h$ to $k$'s route
            $L_k^{rem} \gets$ a fresh leg length drawn in $h$    \Comment{Eq. 4.8}
            record the boundary crossing in $k$'s history

    for each $k \in \mathit{Active}$:    \Comment{reschedule everyone}
        $t_k^{arr} \gets t + L_k^{rem} \big/ \max\big( v_{r(k)},\ v_{min} \big)$    \Comment{Eq. 4.6}

    ## Step 5: remove finished vehicles
    move every $k \in \mathit{Active}$ past its final leg into $\mathit{Completed}$    \Comment{Eq. 4.7}
    record $n$ and $t$ for this event

return $\mathit{Completed}$, accumulation history
```

\edt{The subsections below explain Steps 1, 3, 4 and 5; Step 2 only admits the vehicles whose departure time has come.}

### 4.2.4 Step 1: choosing the next event time

This is how we choose the time for the next event to jump to; this single line is what makes the simulation event-driven rather than time-driven:

$$t \leftarrow \min\Big(\ \min_k t^{arr}_k,\ \ \min t^{dep}_{int},\ \ \min t^{dep}_{ext}\ \Big) \tag{Eq. 4.4}$$

Time jumps to whichever comes first across all three candidate sources: someone finishing a
leg, or someone starting a trip from \edt{either cohort, the} intra-regional \edt{one or the} inter-regional \edt{(external) one} (trips crossing boundaries).


### 4.2.5 Step 3: measuring the network

\edt{Unlike a traditional MFD model,} no conservation equation ($\dot n_i$ = inflow − outflow) is integrated. The simulation counts how many vehicles
are currently in region $i$:

$$n_i(t) = \big|\{\,\text{active }k : \text{region}_k(t) = i\,\}\big| \qquad \text{(Eq. 3.12)}$$

A textbook MFD model treats
$n_i$ as a state variable and integrates $\dot n_i = \text{inflow} - \text{outflow}$. Here
$n_i$ \edt{is derived by counting the vehicles the simulation places in region $i$}. \edt{Because each vehicle is either in one region or has left the network, the count conserves vehicles exactly, with no integration error and no fractional vehicles.}

That count then sets the region's speed through the MFD equation:

$$v_i(n) = \max\Big(\ l_i\left(\delta_1 n^2 + \delta_2 n + \delta_3\right)\varphi(\rho)\,\sigma,\ \ v_{min}\Big), \qquad v_{min} = 0.5\ \text{m/s} \qquad \text{(Eq. 3.5)}$$

Each vehicle stores only the speed of the region it currently occupies, and that speed depends on how many
others are there. Every congestion effect in this simulation flows through that single
scalar mapping $n_i \rightarrow v_i$: every delay thereafter, every externality, and every incentive
the algorithm later tries to exploit.

### 4.2.6 Step 4: motion and rescheduling

Each vehicle's remaining distance is decremented by the distance it covered:

$$L^{rem}_k(t + \Delta t) = \max\Big(0,\ L^{rem}_k(t) - v_{i}\cdot \Delta t\Big) \tag{Eq. 4.5}$$

A leg is finished once $L^{rem}_k \le \varepsilon_{arr} = 10^{-4}$ m. Since the event time
was chosen as the *earliest* arrival, **at least one vehicle satisfies this condition on
every event**
Every active vehicle's next arrival is then re-estimated at the new speed:

$$t^{arr}_k = t + \frac{L^{rem}_k}{\max(v_{i},\ v_{min})} \tag{Eq. 4.6}$$

\edt{Eq. 4.6 is recomputed for every active vehicle at every event, not only for the vehicle}
\edt{that just arrived or for vehicles that changed region.} It is how congestion
propagates through the model. When a region gains vehicles, the instantaneous speed of that region drops, and on the next event *every vehicle inside it* has its predicted arrival pushed farther. 

### 4.2.7 Step 5: completion

A vehicle retires or exits the simulation when it has run past the last leg of its route while standing on its
destination:

$$\text{done}_k \iff \big(\text{leg}_k > |\text{route}_k|\big) \ \wedge\ \big(\text{region}_k = \mathit{Dest}_k\big) \tag{Eq. 4.7}$$

Both conditions are required. The leg counter alone will not track this, because the route is
*rebuilt* every time the vehicle is re-routed. A vehicle can be on leg 5 of a
three-leg plan that has since been replaced by a seven-leg one.

### 4.2.8 Leg structure and realised travel time

A route is a sequence of **legs**, one per region traversed. Leg lengths are drawn
randomly rather than fixed:

$$L^{leg}_k = \begin{cases} 2\,l_i\,U, & U\sim\mathcal U(0,1), \ \text{first or last leg (partial crossing)} \\ l_i, & \text{interior leg (full crossing)}\end{cases} \tag{Eq. 4.8}$$

This comes from simple geometry: a vehicle whose trip *begins* somewhere inside region $i$
only has to cover some part of it and not the whole region before reaching the boundary, and a vehicle whose trip *ends* inside
region $j$ stops partway through. A vehicle passing through covers the region's full
mean trip length $l_i$. Drawing $2 l_i U$ for the partial cases gives a mean of $l_i$, so the
average survives while individual vehicles have some variation.

\edt{The travel time of a leg is the sum of the event intervals the vehicle spends on it, which is approximately:}

$$\tau_k \approx \frac{L_k}{v_i(n_i)} \tag{Eq. 4.9}$$

with $L_k$ the vehicle's **own** drawn length, not $l_i$.


### 4.2.9 A worked event

\edt{Let us go through the event cycle with a worked example.} Take `7_region_new` with region 3
holding $n_3 = 1406$ vehicles, half its critical accumulation.

```pseudocode
caption: One event, traced end to end
label: alg:worked-event
---
Given: region 3 holds $n = 1406$ veh    \Comment{50\% of $n_{cr} = 2812$}
$\quad$ $l = 4910$ m,  $\sigma = 1/3600$

## Step 3: what speed does that accumulation imply?
$v_3 \gets v^{\text{MFD}}(1406) = 7.47$ m/s    \Comment{Eq. 3.5}
$g_3 \gets$ trip completion rate at $n = 1406$, $= 2.14$ veh/s
$T_3 \gets n / g_3 = 657.1$ s    \Comment{Eq. 3.6}

## Step 4: vehicle $k$ is 3000 m into region 3, and the next event is $\Delta t = 12.0$ s away
$L_k^{rem} \gets 3000 - 7.47 \times 12.0 = 2910.4$ m    \Comment{Eq. 4.5}
$\quad$ $L_k^{rem} > \varepsilon_{arr}$, so $k$ does not route
$t_k^{arr} \gets t + 2910.4 / 7.47 = t + 389.6$ s    \Comment{Eq. 4.6}

## Now suppose region 3 fills up: 50 more vehicles arrive
$v_3 \gets v^{\text{MFD}}(1456) = 7.39$ m/s    \Comment{slower}
$t_k^{arr} \gets t' + 2910.4 / 7.39 = t' + 393.8$ s    \Comment{about 4 s later}

```


**Speed over an interval.** A vehicle moves across the interval $[t_{prev}, t]$ at the speed given by the accumulation at time $t$, at the end of the interval. Its motion over each interval therefore reflects the congestion at the end of that interval. Because intervals between events are usually short, the effect on travel times is small.

**Minimum speed.** Speed never falls below $v_{min} = 0.5$ m/s. The bound exists for numerical reasons: speed appears in the denominator of Eq. 4.6 and Eq. 4.9, and the bound keeps the travel time through a fully jammed region large but finite.

\edt{In short, the} accumulation of each region is counted directly from the vehicle records at every event, and that count sets the region's speed. Each vehicle's predicted arrival time is recalculated at every event, which is how congestion spreads through the network. When a vehicle finishes a leg away from its destination, it makes a routing decision: an HDV follows the shortest path, and a CAV asks the learned policy.

## 4.3 Routing of human-driven vehicles (HDVs)

\edt{HDVs route by Dijkstra's shortest path.} They route through the shortest path over a cost matrix rebuilt from the
live network state, and step to the first region on the path. \edt{HDVs do not use the stochastic multinomial logit choice that the model supports; their routing is deterministic. Every HDV applies this rule at every boundary crossing. We made this choice deliberately, to test whether a learned policy can beat shortest-path routing, which approximates the user equilibrium; the HDVs and the Dijkstra-only baseline must therefore follow that rule.}


### 4.3.1 When an HDV makes a decision

We do not have a decision made at every arrival. Vehicle $k$ is asked to choose a next region if and
only if all three of these hold at once, which is the `else` branch of Step 4 in the main
loop:

1. $k$ has just finished a leg, $L_k^{rem} \leq \varepsilon_{arr}$.
2. $k$ is not standing on its destination, $r(k) \neq \mathit{Dest}_k$.
3. The trip crosses at least one region boundary, $O_k \neq \mathit{Dest}_k$.

\edt{The third condition excludes the internal-trip cohort of Table 4.1, whose vehicles start and end in the same region and so never make a routing decision.}

\edt{When a decision is triggered, the vehicle commits only as far as the next region boundary and is then asked again, so no route is chosen at the start of the trip. This is one reason why the HDVs' travel time is hard to beat. An HDV making a shortest-path, or selfish, choice still recomputes that path at every boundary, navigating against the network state as it is at that moment rather than as it was at its last decision.}

### 4.3.2 From network state to next region

HDV routing is made in two phases. The cost tables are rebuilt from the live network state once
per event, before any vehicle is asked anything; each vehicle then does a lookup against them.

```pseudocode
caption: HDV route choice, from network state to next region
label: alg:hdv
---
## Phase 1: rebuild the cost tables, once per event
Input: accumulation $n$, region adjacency $\mathrm{Adj}$

for each region $i$:
    $T_i \gets n_i \big/ g_i(n_i)$    \Comment{Eq. 3.6}

for each ordered pair $(i, j)$:
    if $\mathrm{Adj}_{ij} > 0$:  $C_{ij} \gets T_i + T_j$
    else:  $C_{ij} \gets \infty$

for each origin $i$:
    run Dijkstra over $C$ from $i$
    store, for each destination $j$, the shortest path $i \rightsquigarrow j$

## Phase 2: one vehicle chooses, at a boundary
Input: vehicle $k$ at region $i$, destination $\mathit{Dest}$, feasible set $F$

if $F = \emptyset$:  return nothing
if $|F| = 1$:  return the single element of $F$    \Comment{forced; no choice made}

$\Theta \gets$ the stored shortest path $i \rightsquigarrow \mathit{Dest}$,
    as the sequence of regions it visits
$h \gets \Theta_2$    \Comment{the second region along $\Theta$}

if $h \in F$:  return $h$
else:  return $\arg\min_{h' \in F}\big( C_{i h'} + \text{cost of } h' \rightsquigarrow \mathit{Dest} \big)$
```

Phase 1 builds the cost matrix from the true accumulation $n_i$ of every region.

**The per-region cost is the average travel time of Eq. 3.6**, accumulation divided by trip completion rate (Little's law), in real seconds.

\edt{The cost of moving from region $i$ to region $j$ is $C_{ij} = T_i + T_j$, the travel time of the region being left plus that of the region being entered. Summed along a path, this counts every interior region twice and each end region once. It therefore penalises the number of regions on a route beyond the true travel time and biases route search toward routes through fewer regions, which on this ring-plus-hub network means routes through the hub.}

### 4.3.3 Taking the next region from the shortest path

\edt{An HDV uses only the second region of its shortest path, the region it enters next. It does not need the rest of the route, because it computes a new path at the next boundary. If that region is not in the feasible set, the last line of Algorithm 4.3 instead picks the cheapest of the legal next regions, using the same cost matrix.}


## 4.4 Routing of connected and autonomous vehicles (CAVs)

**A note on terminology.** What the policy is given at a decision point is called the
observation elsewhere in this chapter, but in reinforcement learning the same observation state vector is
usually called the **state vector**. We use that term here, and write it $S$, since this is
the one section where we start with the RL framing. \edt{Similarly, the action $a$ chosen by the policy is the next region the vehicle enters: each action maps one-to-one onto a neighbouring region $h$, so $a = h$. We write $a$ for the action and keep $h$ for a region.}

### 4.4.1 When a CAV makes a decision

A CAV is asked to choose at the same moment as an HDV\edt{, and the HDV's three conditions apply unchanged}. Only the routing branch taken afterwards differs. \edt{If the vehicle is a CAV}, it routes to the learned policy, and everything else
falls through to \edt{the HDV route choice}.

### 4.4.2 The feasible set and the action mask

Before the policy is consulted, the set of feasible next regions is computed from
topology alone:

$$\begin{aligned} F(i, \mathit{Dest}) = \big\{\, h :\ & h \text{ is adjacent to } i \text{ and not blocked}, \\ & \mathit{Dest} \text{ can still be reached from } h \text{ without entering a blocked region} \,\big\} \end{aligned} \tag{Eq. 4.10}$$

To explain, a candidate region $h$ must satisfy three things: it must be **adjacent** to the
vehicle's current region, it must not be **blocked**, and choosing it must still leave a
route to the destination. \edt{The second and third conditions are what prevent} the policy from steering a
vehicle into \edt{an} infinite loop or a dead end during early exploration.

The blocked set $\mathit{Blocked}$ is where the **\edt{no-revisit rule}** acts.

```pseudocode
caption: The feasible set and the revisit rule
label: alg:feasible
---
$\textsc{ChooseFeasibleSet}$ (vehicle $k$ at region $i$, destination $\mathit{Dest}$)

Input: region adjacency $\mathrm{Adj}$, visited history $H_k$

Initialize: $\mathit{Blocked} \gets \emptyset$

if the region-revisit rule is enabled:
    $\mathit{Blocked} \gets H_k \setminus \{i, \mathit{Dest}\}$    \Comment{every region $k$ has
        visited, except where it stands and where it is going}

$F \gets \big\{ h : \mathrm{Adj}_{ih} > 0,\ h \notin \mathit{Blocked},$
    $\quad$ a path $h \rightsquigarrow \mathit{Dest}$ exists that avoids $\mathit{Blocked} \big\}$

if $F = \emptyset$:
    $\mathit{Blocked} \gets \emptyset$
    $F \gets \big\{ h : \mathrm{Adj}_{ih} > 0,\ h \notin \mathit{Blocked},$
        $\quad$ a path $h \rightsquigarrow \mathit{Dest}$ exists that avoids $\mathit{Blocked} \big\}$
        \Comment{relax the rule rather than trap the vehicle with no legal move}

return $F$
```

Reachability is searched over the
region graph with the blocked set removed.

When the no-revisit rule is enabled, every region a vehicle has already visited stays blocked for the rest of its trip\edt{, unless blocking would leave no legal move (the fallback in Algorithm 4.4)}. If this option is **disabled**, `Blocked` set stays empty and feasibility
depends on topology alone: on this network every outer region then offers three options and
the hub six.

Enabling the rule stops a vehicle from revisiting a region, which rules out loops while the policy is still untrained. As a result, the action set **shrinks as a trip lengthens**, since each region visited is one fewer choice
available later. Two consequences:

- A vehicle deep into a trip may be left with a single legal move, in which case the policy
  is never consulted\edt{ and the move is forced}.
- Feasibility becomes **history-dependent**, while the state vector encodes only the
  current network state. Two vehicles with identical state vectors can face different legal
  moves.

The last line of the algorithm is a safety valve. If the history has blocked every route,
the rule is abandoned for that decision and the CAV chooses among all adjacent regions rather than leaving the vehicle stranded.

Whichever way the rule is set, the resulting set is converted into a boolean vector the
network can consume:

$$\text{mask} = \mathbb{1}[h \in F] \in \{0,1\}^{7}$$


### 4.4.3 The state vector

\edt{The state vector $S$ is the only input to the policy, which keeps no memory of}
\edt{previous decisions.} 28 dimensions:

$$S = \Big[\ \underbrace{\mathbb{1}[i = \text{curr}]}_{7}\ \Big\|\ \underbrace{\mathbb{1}[i = D]}_{7}\ \Big\|\ \underbrace{\hat n_i}_{7}\ \Big\|\ \underbrace{\mathbb{1}[i \in F]}_{7}\ \Big] \tag{Eq. 4.11}$$

| Feature | Symbol | Width | What it tells the policy |
|---|---|---|---|
| One-hot current region | $\mathbb{1}[i = \text{curr}]$ | 7 | Current region |
| One-hot destination | $\mathbb{1}[i = D]$ | 7 | Destination region |
| Density | $\hat n_i$ | 7 | Current load of every region |
| Action mask | $\mathbb{1}[i \in F]$ | 7 | Legal next moves |

: The four features of the CAV state vector.

Density is normalised by a capacity, either jam or critical accumulation:

$$\hat n_i = \frac{n_i}{\text{cap}_i}, \qquad \text{cap}_i = \begin{cases} n_{max,i}, & \text{jam normalisation (default)} \\ n_{cr,i}, & \text{critical normalisation}\end{cases} \tag{Eq. 4.12}$$

On `7_region_new`, $n_{max} = 5536$ against $n_{cr} = 2812$, and regions typically run between
1000 and 3000 vehicles, so the $n_{max}$-normalised feature spans only about 0.18 to
0.54. Half its range goes unused, and $n_{cr}$, the one value that governs the dynamics, sits
at an unremarkable 0.51. Normalising by $n_{cr}$ instead puts the decision-relevant regime at
1.0, where a network can learn a threshold more easily.

\edt{A CAV sees only the network occupancy values in its state vector. It has to learn the relationship between occupancy and expected delay on its own, as Chapter 5 describes.}

### 4.4.4 Choosing the next region

```pseudocode
caption: CAV next-region selection
label: alg:cav-choice
---
$\textsc{ChooseNextRegion-CAV}$ (vehicle $k$ at region $i$, destination
$\mathit{Dest}$, feasible set $F$, policy $Q$)

    if $F = \emptyset$:  return nothing
    if $|F| = 1$:  return the single element of $F$    \Comment{policy is not consulted}

    $m \gets$ mask over actions, $m_a = 1$ if $a \in F$ else $0$
    $S \gets$ state vector for $k$    \Comment{Eq. 4.11}

    ## ask the policy
    $q_a \gets Q(S, a)$ for every action $a$    \Comment{one forward pass}
    $q_a \gets -\infty$ for every $a$ with $m_a = 0$    \Comment{illegal moves cannot win}
    $a \gets \arg\max_a q_a$    \Comment{Eq. 4.13}

    during training only, with probability $\epsilon$:
        $a \gets$ a uniform choice over $F$    \Comment{exploration}

    return $a$
```

$Q$ is a **Q-network**: a function, parametrised by weights learned during training, that
takes a state vector and an action and returns a single number, the action's estimated value.
No table of values is stored anywhere; the network is queried fresh at every decision. How
those weights are learned is the subject of the next chapter. Given $Q$, the routing decision
is a masked greedy choice over learned action values:

$$a^* = \arg\max_{a \in F(i,\mathit{Dest})} Q(S, a) \tag{Eq. 4.13}$$

\edt{$Q(S, a)$} is the network's learned estimate of how good it is to
\edt{take action $a$, that is, step to region $h = a$,} given state vector $S$, where higher is better. The policy evaluates all
seven candidates in one forward pass, then takes the best one that is legal. \edt{This chapter illustrates inference with a value-based policy $Q$; under the policy-gradient alternative of the next chapter, the masked argmax is replaced by sampling from a learned action distribution $\pi(\cdot \mid S)$.}

**How masking is implemented:** Illegal actions are set to $-\infty$
*before* the argmax is applied, rather than filtered afterwards. \edt{An illegal region then cannot be selected, whatever value the Q-network assigns to it, so even an untrained network cannot choose an unreachable region.}

### 4.4.5 Greedy evaluation

\edt{The $\epsilon$-greedy branch in Algorithm \ref{alg:cav-choice} is used only during training.} \edt{When we} route CAVs during testing, \edt{that is, in evaluation runs} ($\epsilon = 0$), we take the pure
argmax. \edt{The exploration schedule is described in Chapter 5.}


## 4.5 Comparing HDV and CAV routing

\edt{A CAV routes at the same trigger as an HDV but through a learned policy. Illegal next regions are masked to $-\infty$ before the argmax, so they can never be chosen.}


```{=latex}
\setcounter{chapter}{4}
```

# \edt{Reinforcement Learning for CAV Routing}

\edt{In the previous chapter we introduced how CAVs make their routing decisions: a CAV that reaches a region boundary selects the neighbouring region with the highest learned Q-value, that is, the action $a^* = \arg\max_a Q(S,a)$. This chapter explains what that Q-function is and how it is learned, and introduces a policy-gradient alternative.}

\edt{Reinforcement learning trains a policy $\pi : \mathcal{S} \longrightarrow \mathcal{A}$, the decision rule that maps the state vector a vehicle observes at a boundary to the region it enters next. The policy is implemented as a single neural network shared by every controlled vehicle, so each update to its weights improves the entire fleet simultaneously.} The agent observes a state, selects an action, and receives a reward together
with the state that follows. Applied
here for our routing problem, the state is the prevailing congestion picture of the regions from the perspective of the vehicle, the action is the choice of which
neighbouring region to enter, and the reward is negative travel time, so that maximising
reward is equivalent to minimising the time vehicles spend on the road.

\edt{Section 5.2 defines what a single training example is, given that one decision covers a whole leg. As a reminder, a leg here refers to a single traversal through a region, running from one boundary decision to the next. Section 5.3 explains how the network-wide travel-time objective is translated into a reward that can be credited to an individual vehicle's decision. Sections 5.4 and 5.5 describe how the policy is trained on those examples.}


## 5.1 Problem formulation

The environment is the regional network of Chapter 1, simulated as Chapter 4 describes.

**The objective.** \edt{The network-level objective is to minimise} the **total system travel time
\edt{(TSTT)}** of the whole network: time spent on the road, summed over every vehicle, both controlled and
uncontrolled, over the simulation horizon $[0, T]$.

$$
\mathrm{TSTT} \;=\; \int_{0}^{T} \sum_{i \in \mathcal{R}} n_i(t)\, \mathrm{d}t
\tag{Eq. 5.1}
$$

where $n_i(t)$ is the accumulation of region $i$ at time $t$ and $\mathcal{R}$ is the set of
regions. Since $\sum_i n_i(t)$ counts the vehicles in flight at time $t$, the integral is the
aggregate time the network's users spend travelling. The learning problem is posed as
maximisation of $-\mathrm{TSTT}$\edt{, in vehicle-seconds}. 

**The learning objective.** Total system travel time is \edt{the metric} the *network* is evaluated on at the end of our training, but
\edt{it cannot be used directly as the training objective, as we will see in \emph{Action dependence and credit assignment}}. 
\edt{In standard reinforcement learning, the objective is defined as the expected discounted return earned by the agent following the policy $\pi$:}

$$J(\pi) \;=\; \mathbb{E}_{\zeta \sim \pi}\!\left[\ \sum_{t=0}^{T} \gamma^{\,t} r_t \right] \tag{Eq. 5.2}$$

where \edt{$\zeta$ is a \textbf{trajectory} generated by $\pi$: the ordered sequence of states,
actions and rewards the agent produces from its first step to the end of the episode, $\zeta = (S_0, a_0, r_0, S_1, a_1, r_1, \ldots)$. Writing $\zeta \sim \pi$ means
the expectation is taken over the trajectories the policy generates, since the policy decides the
actions and the actions decide which states follow.} $r_t$ \edt{is} the reward received at step $t$, and
$\gamma \in (0,1]$ the discount weighting later reward against sooner.

\edt{To apply this to our routing problem, we make two substitutions.} First, a *step* here is not a unit of
time but one leg, so the index runs over a vehicle's successive boundary decisions $j$ rather
than over seconds. Second, $r_j$ is the per-leg reward of Eq. \edt{5.7}, in seconds and built from
negative delay, so maximising the sum is minimising time spent on the road. \edt{Note that this expectation is over a single vehicle's journey, not the whole fleet. The sum below tracks one vehicle across its own legs, and every controlled vehicle feeds its own trajectory into training the same shared policy.}

$$J(\pi) \;=\; \mathbb{E}_{\pi}\!\left[\ \sum_{\text{leg } j=0}^{\text{final leg}} \gamma^{\,j} \underbrace{r_j}_{\text{reward on leg } j} \right] \tag{Eq. 5.3}$$

\edt{For a single vehicle, this adds up the reward from every leg between its first decision ($j = 0$) and arrival, weighting each later leg by an extra factor of $\gamma$.}

\edt{Four properties are unique to our event-driven routing problem, and they make reinforcement learning challenging to apply here. The design choices we made to resolve them are covered in later sections.}

\edt{\textbf{A decision covers a whole leg -- is this still Markovian?} --} In a standard RL problem, an agent acts, the model
advances one step, and a reward arrives. Here a vehicle \edt{starts at its departure time}, chooses a region and then *drives
through it*: hundreds of simulated seconds, spanning many simulation events that concern other vehicles, before it reaches the boundary and makes the next decision, and so on until it arrives at \edt{its} destination. The interval between one vehicle's consecutive decisions is a random
variable rather than a constant fixed time, which makes this a **semi-Markov** decision process. The reward for a single action has to be gathered across an
interval whose duration is not known in advance\edt{,} which makes our problem statement challenging.

\edt{\textbf{Thousands of vehicles share one policy} --} Every controlled vehicle in the network consults
the same network weights, which is the simple RL model we have started with. There is no per-vehicle policy. Each vehicle acts on its own state
vector, with no communication with other agents and no shared trip plan, but we train only a single central
policy rather than one learning algorithm per vehicle, which is why the two algorithms are \edt{referred to here as deep Q-network (DQN) and Proximal Policy Optimization (PPO).}

\edt{\textbf{Non-stationarity as every agent is acting simultaneously} --} Since all controlled vehicles share the policy,
improving it changes how every other vehicle behaves at once. That changes the congestion pattern,
which changes what counts as a good decision. \edt{Each vehicle faces} an environment that is constantly shifting
underneath it for reasons its own state vector never captures. Non-stationarity is the usual difficulty of
multi-agent learning, and it is why training has more variance than the same algorithm
run on a stationary environment.

\edt{\textbf{One vehicle is a tiny fraction of the network} --} With tens of thousands of vehicles in flight, a single routing choice moves any network-wide quantity by far less than the noise
everyone else's simultaneous choices generate. \edt{For reward design, this means that a reward measuring a global outcome carries almost no signal about which action caused it.}


## 5.2 Constructing a training example from a leg

### 5.2.1 One transition per leg

A training example is a tuple \edt{$(S, a, r, S')$}\edt{, whose elements are defined in our simulation as follows:}

| Element | Meaning here |
|---|---|
| $S$ | The state vector the vehicle sees as it reaches a boundary and is asked to choose an action |
| \edt{$a$} | The action chosen, the region the vehicle enters next |
| $r$ | The reward or cost it accumulated while it drove through that region |
| $S'$ | The state vector at its **next** boundary |

So a transition spans **one leg**, running from one decision to the same vehicle's
next decision, and covers many simulation events concerning other agents along the way. It does not correspond to a
simulation step.


```pseudocode
caption: The life-cycle of one transition
label: alg:transition
---

Initialise, at the start of the episode:
    $R_{sys}^{ep} \gets 0$    \Comment{episode system reward, reported only}
    $C_X \gets 0$ for every category $X$    \Comment{running integrals}

At every simulation event, of duration $\Delta t$:
    $R_{sys}^{ep} \gets R_{sys}^{ep} + R_{sys}$    \Comment{Eq. 5.5, network-wide, never trained on}
    for every category $X$:
        $C_X \gets C_X + \Delta t \cdot N_X$    \Comment{cost charged to $X$ over this event}
    

When vehicle $k$ is asked to choose at time $t$:

    if $k$ has an open transition from an earlier decision:
        $\tau_k \gets t - t_k^{open}$    \Comment{how long that leg took}
        $r \gets \mathrm{LegReward}(k, \tau_k)$    \Comment{reward design}
        emit the training example $(S_k^{open}, a_k^{open}, r, S)$
            \Comment{$S$ is the state vector right now: the next state of the closed leg}

    $a \gets$ the region chosen now    \Comment{from the active policy}

    open a new transition for $k$, recording:
        $S_k^{open} \gets S$
        $a_k^{open} \gets a$
        $t_k^{open} \gets t$
        a snapshot of every running integral, $C_X(t)$

At the end of the episode:
    for each vehicle $k$ with an open transition:
        discard it    \Comment{leg cut short; see Section 5.2.3}
    report $R_{sys}^{ep}$    \Comment{for evaluation only; never used in training}
```

\edt{Each boundary state vector closes the preceding transition as $S'$ and opens the next one as $S$.}

### \edt{5.2.2 Computing category-wise costs over a leg}

\edt{The simulation reports costs per event across the network, whereas a vehicle's leg spans many events. While a vehicle's own leg duration $t_{own}$ is the elapsed time $t_1 - t_0$, computing the collective terms of the reward requires knowing the total cost accrued by other vehicles during that same interval $[t_0, t_1]$.}

\edt{To obtain this, we maintain a running total $C_X(t)$ for each vehicle category $X$, updated at every event. When a leg opens at time $t_0$, the current totals are recorded; when the leg closes at time $t_1$, they are subtracted to give the category-wise cost accrued during that leg:}

$$\Delta C_X(\text{leg}) \;=\; C_X(t_1) \;-\; C_X(t_0) \tag{Eq. 5.4}$$

\edt{Here, $C_X(t)$ is the running total, in vehicle-seconds, of all cost charged to category $X$ from the start of the episode up to time $t$. The subscript $X$ selects one of three categories: the vehicle's own fleet (\textit{my}), all other vehicles (\textit{other}), or the whole network (\textit{all}). The times $t_0$ and $t_1$ are when the leg opened and closed, in seconds.}

\edt{Each category total $\Delta C_X(\text{leg})$ is then divided by the corresponding vehicle headcount to obtain the category-wise average delays ($\bar{t}_{my}$, $\bar{t}_{other}$ and $\bar{t}_{all}$). These averages are then combined with the vehicle's own leg duration $t_{own}$ and optional shaping terms to produce the final per-vehicle, per-leg reward $r$ (Eq. \edt{5.7}).}

\edt{This method costs two lookups and one subtraction per leg, no matter how many events occurred in between. Tracking each vehicle individually through its leg would be the obvious alternative, but with tens of thousands of vehicles in flight it is too slow to run.}

### \edt{5.2.3 Handling arrivals and truncated trajectories}

\edt{When a vehicle reaches its destination, its final transition closes as a terminal state: there is no next state, the value of what follows is zero, and the transition is emitted with a terminal flag.}

\edt{When an episode is cut short, however, vehicles still driving have an open leg that stopped mid-journey. This is not a terminal state, because the consequence of the last choice was never observed, so these legs are discarded. When training on return-based targets, the whole trajectory is discarded, since a missing remainder corrupts every leg in it.}


## 5.3 Reward design

The simulation and the training algorithm require different quantities. The simulation tracks a network-wide **system reward** (Eq. 5.5) per event for evaluation, whereas training optimises a per-vehicle, per-leg **vehicle reward** (Eq. 5.7) derived from its components via the differencing of Eq. 5.4. Keeping these two levels separate follows from the mixed-class assignment and credit-assignment constraints set out in Chapter 2 (Sections 2.2, 2.4, and 2.5).

### 5.3.1 The system reward

At every event of duration $\Delta t$, the network accrues cost in proportion to the number of active vehicles:

$$R_{sys} = -\underbrace{\Delta t \cdot N_{active}}_{\text{Total Time Spent}} \;-\; w\,\Delta t \sum_{i} \left(\frac{\max(n_i - n_{cr,i},\ 0)}{n_{cr,i}}\right)^{2} \tag{Eq. 5.5}$$

The first term is Total Time Spent (TTS): every active vehicle accrues $\Delta t$ seconds of cost, negated so that lower travel time yields higher reward. Summing TTS over an episode and dividing by the total number of vehicles gives the **average system travel time** (seconds per vehicle) reported throughout Chapter 6. The second term is an optional overload penalty above critical accumulation $n_{cr,i}$; its default weight is $w = 0$ and it is unused in the experiments reported here. Evaluating the first term separately by vehicle class yields the per-category running integrals $C_X$ used in Eq. 5.4.

### 5.3.2 Per-leg averages

To provide a per-vehicle delay rather than a fleet total, each category's leg cost $\Delta C_X(\text{leg})$ (Eq. 5.4) is divided by the **distinct vehicle count ($M_X$)** active at any point during the leg:

$$\bar t_X = \frac{\Delta C_X(\text{leg})}{M_X}, \qquad M_X = N^X_{active}(t_1) + \big[\text{arrived}_X(t_1) - \text{arrived}_X(t_0)\big] \tag{Eq. 5.6}$$

Here $M_X$ counts vehicles still active at $t_1$ plus those that completed their trips during $[t_0, t_1]$, giving $\bar t_X$ in seconds per vehicle. Using a headcount rather than time-averaged occupancy is essential: a time-averaged denominator cancels the vehicle count in the numerator and collapses every category average onto the vehicle's own leg duration $t_{own}$\edt{, since $\bar N = \Delta C_X / t_{own}$ and therefore $\Delta C_X / \bar N = t_{own}$ for every category}.

### 5.3.3 The vehicle reward, behaviour profiles, and shaping

The per-leg reward used for training combines the vehicle's own travel time with category-wise average delays, weighted by a **behaviour profile**, plus optional shaping terms:

$$r = \frac{1}{1000}\Big(\underbrace{a\,t_{own} + b\,\bar t_{my} + c\,\bar t_{other} + d\,\bar t_{all}}_{\text{behaviour profile}} \underbrace{- \lambda_{marg}\,c^{marg} - \lambda_{ext}\,c^{ext}}_{\text{optional shaping}}\Big) \tag{Eq. 5.7}$$

```{=latex}
% plain tabular instead of a pipe table, so the five rows cannot split across a page
\begin{center}
\begin{tabular}{ll}
\toprule
Term & Meaning \\
\midrule
$t_{own}$ & Duration of this vehicle's own leg \\
$\bar t_{my}$ & Average delay across the vehicle's own fleet \\
$\bar t_{other}$ & Average delay across all other vehicles \\
$\bar t_{all}$ & Average delay across the whole network \\
\bottomrule
\end{tabular}
\end{center}
```

All terms are in seconds; the factor $1/1000$ rescales rewards to order one for neural-network stability without altering action rankings.

A **behaviour profile** fixes the weights $(a, b, c, d)$, \edt{following} the Urban Routing Benchmark \cite{Akman2025} (Section \edt{2.5}) \edt{but extended to four terms,} to span objectives from purely self-interested to purely network-oriented without tuning coefficients between runs:

| Profile | $a$ | $b$ | $c$ | $d$ | What it optimises |
|---|---|---|---|---|---|
| `selfish` | −1.0 | 0 | 0 | 0 | Own travel time only |
| `collaborative` | −0.5 | −0.5 | 0 | 0 | Own time and own fleet's, equally |
| `social` | −0.5 | 0 | 0 | −0.5 | Own time and whole network's, equally |
| `altruistic` | 0 | 0 | 0 | −1.0 | Whole network's time only |

: The behaviour profiles used in the experiments of Chapter 6.

The **altruistic** reward is not the system reward of Eq. 5.5: $\bar t_{all}$ covers only the vehicle's own leg $[t_0, t_1]$ and is divided by the vehicles present during it, so it is the per-vehicle share of $R_{sys}$ over that leg.

Because delays are positive and rewards are maximised, negative weights penalise delay. With a single CAV fleet, $b$ weights that fleet and $c$ weights HDVs; in a two-fleet configuration, $c$ spans HDVs and the second CAV fleet.

**Optional shaping terms.** Two additional cost terms, marginal travel time $c^{marg}$ ($= T_i^{marg}$, Section 3.1.7) and congestion externality $c^{ext}$ (Section 3.1.8), can be enabled via $\lambda_{marg}$ and $\lambda_{ext}$, both defaulting to zero. Both are measured in seconds on the entered region and subtracted so a vehicle can price the delay it imposes on others.

### 5.3.4 Action dependence and credit assignment

As noted in Section \edt{2.4}, a reward term teaches only to the extent that the vehicle's own choice measurably changes it. In Eq. 5.7, $\bar t_{all}$ averages over thousands of concurrent vehicles, so a single CAV's routing decision shifts it far less than background traffic variation; training solely on it (**altruistic**, $d = -1$) therefore suffers severe gradient variance. Targeting a network-level objective without losing action attribution requires terms evaluated at the chosen region itself, such as $t_{own}$, $c^{marg}$, or $c^{ext}$, which vary directly across candidate next regions.


## 5.4 \edt{Value-based training (DQN)}

A **value-based** training algorithm does not \edt{store a policy directly}. Instead, it learns \edt{a} state-action value function, \edt{$Q(S, a)$}, the expected discounted \edt{return} of \edt{taking action $a$ (entering that region)} from state $S$ and \edt{acting greedily thereafter}, and \edt{derives the policy directly as $\arg\max_a Q(S,a)$}. Improving the policy therefore means improving an estimate, not adjusting a separate set of policy parameters.

The algorithm we use is \edt{the deep Q-network} (DQN)\edt{, introduced by Mnih et al. \cite{Mnih2015},} which approximates \edt{$Q(S, a)$} with a neural network rather than a lookup table\edt{,} since the state observation vector is continuous. A single network is shared across the entire controlled fleet, meaning every CAV queries it at every decision point and contributes its experience to the training of \edt{the} same network parameters. There is no inter-vehicle communication or joint action space. Because DQN is \edt{an} off-policy algorithm, past transitions can be stored and reused across updates even though they were generated by earlier policies. Here, we start with DQN as a benchmark for congestion-dependent routing, against which we compare the \edt{policy-gradient} method later.

### \edt{5.4.1} The replay buffer and target network

\edt{Two components introduced with DQN by Mnih et al. \cite{Mnih2015} keep this training stable: a replay buffer and a target network.}

**The replay buffer.** Training directly on the transitions \edt{$(S, a, r, S')$} as they happen is unstable because \edt{consecutive} decisions from the same vehicle in an episode are strongly correlated. Each step would be \edt{a high-variance} estimate of the true gradient\edt{,} and successive steps along the trajectory with the same policy would amplify the same bias. Storing transitions in a replay buffer and sampling uniformly breaks this correlation, lowers update variance, and lets each transition be reused across multiple updates. This also \edt{keeps} the training \edt{sample-efficient, because transitions} are computationally expensive to generate.

**The target network.** The target \edt{$y = r + \gamma \max_{a'} \bar{Q}(S', a')$} is computed using $\bar{Q}$, a slowly-updated copy, rather than $Q$ itself, \edt{because updating $Q$ would otherwise also shift the regression target, amplifying estimation errors.} This instability in reinforcement learning is well-known and is commonly referred to as \edt{the} *deadly triad* \edt{\cite{SuttonBarto2018}}, \edt{which arises whenever function approximation, bootstrapping and off-policy training are combined. This is the setting we have in our problem. Added to this, the environment is itself non-stationary: since every controlled vehicle shares the policy, improving it changes the congestion pattern the values are being fitted to.} Blending $\bar{Q}$ slowly toward $Q$\edt{, the soft update introduced by Lillicrap et al. \cite{Lillicrap2016},} keeps the target approximately fixed over any single update\edt{.}

### \edt{5.4.2} Targets and discounting

The standard DQN target is bootstrapped: it estimates the value of the remaining journey using the target network's own prediction, \edt{$\max_{a'} \bar{Q}(S', a')$}. This is efficient because updates can occur as soon as a leg closes, but approximation errors in $\bar{Q}$ propagate into subsequent estimates.

As an alternative, \edt{bootstrapping can be dropped} in favour of realised (Monte-Carlo) returns. Every leg of a vehicle's journey is held until arrival, and the target is computed backwards from the final leg:

$$G_j = r_j + \gamma G_{j+1}, \qquad G_{\text{final}} = r_{\text{final}} \tag{Eq. 5.8}$$

Each transition is then flagged as terminal so that the target collapses to $G_j$, regressing $Q$ directly onto a measured return rather than a predicted one. This is \edt{the} standard bias-variance trade-off \edt{of} RL theory. Bootstrapped targets are biased by current network errors but have lower variance and are immediately available, whereas realised returns are unbiased but have higher variance, require waiting until arrival\edt{, which takes longer when the} network is congested, and \edt{lose} the entire trajectory if an episode is truncated.

\edt{\textbf{Discounting variable-length legs.}} Because each decision commits a vehicle to crossing an entire region, the holding time $\tau_k$ between consecutive decisions is a random variable that depends on regional congestion. A strictly semi-Markov formulation would discount by elapsed time, $\gamma^{\tau_k}$, discounting longer legs more heavily than shorter ones. Here, we instead apply a uniform per-decision factor $\gamma$ regardless of $\tau_k$. This approximation is tolerable because the effective horizon in steps is very short: a typical journey spans three to four legs, so at $\gamma = 0.99$:

$$\gamma^3 = 0.99^3 \approx 0.97$$

Over four legs, the final leg is only discounted by \edt{3\%}, so the policy still optimises nearly the full journey cost. Keeping $\gamma < 1$ also keeps value estimates bounded when a vehicle gets stuck in heavy congestion\edt{.}

### \edt{5.4.3} Exploration by the shared policy during DQN training

Exploration applies only to the decisions of CAVs, since HDVs always take the shortest path. Each time a CAV reaches a boundary during training, the shared policy either explores, picking a random feasible region with probability $\epsilon$, or exploits, picking the region with the highest Q-value. At the start of the training run, our freshly initialised policy does not make good decisions\edt{,} and we need to \edt{encourage} the policy to explore before greedily querying it for \edt{the} best decisions. \edt{Therefore, we use $\epsilon$-greedy exploration}: with probability $\epsilon$ the vehicle picks uniformly among its *legal* moves rather than the highest-valued one.

$\epsilon$ starts high and anneals toward a small lower bound\edt{, such as} 0.05, \edt{with optional periodic resets to a higher value from which it anneals again,} so the policy explores broadly early and exploits what it has learned at later stages. \edt{Conventionally, some exploration is kept until the end}, because the environment is non-stationary and a region that was a poor choice early can become a good one once other vehicles change how they behave. \edt{Exploration is used} only at **training time**. Evaluation or testing after the training \edt{acts} greedily, \edt{i.e.\ with $\epsilon = 0$.}

\edt{Putting the replay buffer, the target network, the bootstrapped target and $\epsilon$-greedy exploration together gives the whole procedure:}

```pseudocode
caption: Value-based training (deep Q-learning)
label: alg:iql
---

Algorithm parameters: discount $\gamma$, learning rate $\alpha$, batch size $B$, update interval $K$,
    target-update rate $\tau_{soft}$, exploration rate $\epsilon$

Initialise:
    $Q \gets$ action-value network, random weights
    $\bar{Q} \gets$ target network, a copy of $Q$
    $D \gets$ replay buffer, empty

Loop for each episode:

    Run one episode of the simulation; for each boundary decision $(S, F)$ at decision count $k$:

        ## anneal, then draw
        $\epsilon \gets \max(\epsilon_{end},\ \epsilon_{start} \cdot \text{decay}^{\,k})$

        if training and $\mathrm{Uniform}(0,1) < \epsilon$:
            $a \gets$ a region drawn uniformly from $F$    \Comment{explore, legal moves only}
        else:
            $q_a \gets Q(S, a)$ for every action $a$
            for every $a \notin F$:  $q_a \gets -\infty$
            $a \gets \arg\max_a q_a$    \Comment{evaluation always takes this branch}

        On leg close (Algorithm 5.1): append $(S, a, r, S')$ to $D$

        Every $K$ decisions, if $|D| \geq B$:
            Sample a minibatch of $B$ examples uniformly at random from $D$
            for each example:
                if the example is terminal:
                    $y \gets r$
                else:
                    $y \gets r + \gamma \cdot \max_{a'} \bar{Q}(S', a')$
            Update $Q$ by one gradient step, learning rate $\alpha$, on $\mathrm{mean}\big((Q(S,a) - y)^2\big)$
            $\bar{Q} \gets \tau_{soft}\cdot Q + (1 - \tau_{soft})\cdot \bar{Q}$    \Comment{slow blend toward $Q$}
```


## 5.5 \edt{Policy-gradient training (PPO)}

\edt{As an alternative to DQN, we train Proximal Policy Optimization (PPO), designed by Schulman et al. \cite{Schulman2017}, an on-policy actor-critic method that learns a stochastic policy directly rather than acting greedily on learned action values. PPO is chosen here because its state-value baseline reduces gradient variance under noisy multi-agent rewards, while its clipped objective keeps policy updates stable without requiring a replay buffer.}

\edt{The leg-based transition structure, running-integral differencing, arrival and truncation handling, and the entire reward design (including all behaviour profiles) remain identical to those used for DQN, so any difference in outcome is attributable to the learning algorithm rather than the objective.}


### \edt{5.5.1} Actor-critic instead of action values

The network has one shared body and two heads\edt{.} The **policy head** emits one score per region, which after masking and normalisation defines a
probability distribution over the next region. The **value head** emits a single number, an
estimate of the expected return from the current state irrespective of the action taken. The
shared body has the same width as the \edt{DQN} network, so that any difference in
results is not attributable to a difference in model capacity.

Action masking works as before: illegal moves are set to $-\infty$ before normalisation, forcing \edt{the probability} of choosing \edt{an} illegal region to zero.

Exploration is handled differently as a result. The value-based \edt{method} acts greedily with
probability $1-\epsilon$ and at random otherwise, following a schedule fixed in advance. Here, \edt{exploration} is built into the stochastic policy itself\edt{: rather than taking an $\epsilon$-greedy argmax,} the vehicle samples its next region from the resulting action distribution, \edt{narrowing} naturally as the policy becomes more confident without needing an external annealing schedule.

### \edt{5.5.2} On-policy constraints

PPO is **on-policy**: its update is only valid for experience generated by the policy being updated. There are three primary differences from DQN in PPO:

| \edt{Difference} from DQN | Reason |
|---|---|
| Updates run between episodes, not during them | The policy must stay fixed throughout a rollout so the update compares the new policy against a single data-generating policy. |
| Ordered per-vehicle trajectories, not a replay buffer | Advantage estimation recurses backwards along each vehicle's sequence of legs; shuffling destroys this ordering. |
| Action probability and state value stored at sampling time | Both must be evaluated under the policy that chose the action, not recomputed after weights change. |


Unlike DQN, which updates continuously from a replay buffer, PPO collects a full episode under fixed weights, runs a small number of passes over that rollout, and discards it.

### \edt{5.5.3} Advantage estimation


The policy is updated in the direction of the advantage, meaning how much better an action turned out than the value head expected. The per-leg temporal-difference error is $\delta_j = r_j + \gamma V(S_{j+1}) - V(S_j)$, combined backwards along each vehicle's trajectory\edt{ by generalised advantage estimation (GAE), proposed by Schulman et al. \cite{Schulman2016}}:   

$$\hat{A}_j = \delta_j + \gamma\lambda\,\hat{A}_{j+1}, \qquad \hat{A}_{\text{final}} = \delta_{\text{final}} \tag{Eq. 5.9}$$
\edt{with $V(S_{\text{final}+1}) = 0$ on arrival.}

The parameter $\lambda$ here controls the bias-variance trade-off\edt{. At} $\lambda = 0$\edt{,} the recursion
collapses to $\hat{A}_j = \delta_j$, a single bootstrapped step \edt{(}low variance, biased by \edt{the error in the value head $V$)}. At $\lambda = 1$ \edt{the intermediate $V$ terms cancel, leaving} the realised return less
$V(S_j)$ (unbiased, but \edt{with the} higher variance of the whole journey).

### \edt{5.5.4} The clipped objective

To prevent a single update from pushing the policy too far from the one that collected the data, PPO tracks the probability ratio $\rho_j = \pi_{\text{new}}(a_j \mid S_j) / \pi_{\text{old}}(a_j \mid S_j)$ and clips it within a bound\edt{, the clipped surrogate objective of Schulman et al. \cite{Schulman2017}}:

$$L^{policy} = -\,\mathbb{E}\Big[\ \min\big(\rho_j \hat{A}_j,\ \ \mathrm{clip}(\rho_j,\ 1-\epsilon_{clip},\ 1+\epsilon_{clip})\,\hat{A}_j\big)\Big] \tag{Eq. 5.10}$$

The unclipped term $\rho_j \hat{A}_j$ is the ordinary policy-gradient
objective: raise the probability of actions whose advantage was positive. The clipped term is
the same thing with the ratio confined to a band around 1. \edt{During optimisation, minimising the above objective means maximising the $\min(\cdot, \cdot)$ term inside the expectation:}

- \edt{$\hat{A}_j > 0$ (good action): the optimiser pushes $\rho_j$ up; past $1 + \epsilon_{clip}$, the clip caps the value at $(1 + \epsilon_{clip})\hat{A}_j$, zeroing the gradient so the policy stops moving further.}
- \edt{$\hat{A}_j < 0$ (bad action): the optimiser pushes $\rho_j$ down; below $1 - \epsilon_{clip}$, the clip caps it at $(1 - \epsilon_{clip})\hat{A}_j$, again zeroing the gradient.}

\edt{\textbf{Why take the minimum rather than clipping alone?} If $\rho_j$ moves in the wrong direction (e.g., $\hat{A}_j > 0$ but $\rho_j < 1 - \epsilon_{clip}$), $\min$ picks the unclipped $\rho_j \hat{A}_j$ so the gradient stays alive to pull it back. It only stops updates that overshoot in the good direction.}

The full loss adds a value-head \edt{regression term} and an entropy bonus\edt{:}

$$L = L^{policy} \;+\; c_v\,\underbrace{\mathbb{E}\big[(V(S_j) - G_j)^2\big]}_{\text{value head}} \;-\; c_e\,\underbrace{\mathbb{E}\big[H(\pi(\cdot \mid S_j))\big]}_{\text{entropy bonus}}$$

The value term trains the critic used in \edt{Eq. \edt{5.9}}. \edt{The entropy}, $H(\pi(\cdot \mid S_j)) = -\sum_a \pi(a \mid S_j)\log \pi(a \mid S_j)$, measures how spread out the action probabilities are\edt{:} high when the policy is near-uniform, zero when it is deterministic. Because $-c_e \mathbb{E}[H]$ is subtracted from the loss, minimising the loss pushes entropy up, penalising the policy for collapsing onto a single action too early\edt{, so it encourages exploration.}

\edt{Putting the on-policy constraints, advantage estimation and the clipped objective together} gives the whole procedure:

```pseudocode
caption: Policy-gradient training (PPO)
label: alg:ippo
---

Algorithm parameters: discount $\gamma$, advantage decay $\lambda$, clip width $\epsilon_{clip}$,
    passes per rollout $E$, value weight $c_v$, entropy weight $c_e$

Initialise:
    $\pi \gets$ policy head, $V \gets$ value head, sharing one body; random weights

Loop for each episode:

    ## Collect, with the policy held fixed
    Run one episode with fixed weights:
        at each boundary decision $(S, F)$:
            sample $a \sim \pi(\cdot \mid S)$, masked to $F$
            store $(S, a, \pi_{old}(a \mid S), V(S))$ in that vehicle's trajectory, in order
        on leg close (Algorithm 5.1): attach the leg reward $r$ to that entry
    Discard truncated trajectories.

    ## Estimate advantages, per vehicle
    for each completed trajectory:
        $\hat{A}_{\text{final}+1} \gets 0$
        for $j$ from the last leg backwards to the first:
            $\delta_j \gets r_j + \gamma \cdot V(S_{j+1}) - V(S_j)$    \Comment{$V$ after arrival is 0}
            $\hat{A}_j \gets \delta_j + \gamma \cdot \lambda \cdot \hat{A}_{j+1}$    \Comment{Eq. 5.9}
            $G_j \gets \hat{A}_j + V(S_j)$    \Comment{target for the value head}

    Normalise the advantages across the whole rollout.

    ## Update, several passes over the same data
    Repeat $E$ times:
        for each minibatch of legs:
            $\rho_j \gets \pi(a_j \mid S_j) / \pi_{old}(a_j \mid S_j)$
            $L_{policy} \gets -\mathrm{mean}\big(\min(\rho_j \hat{A}_j,\ \mathrm{clip}(\rho_j, 1-\epsilon_{clip}, 1+\epsilon_{clip}) \hat{A}_j)\big)$    \Comment{Eq. 5.10}
            $L_{value} \gets \mathrm{mean}\big((V(S_j) - G_j)^2\big)$
            $L_{entropy} \gets \mathrm{mean}\big(H(\pi(\cdot \mid S_j))\big)$
            Update $\pi, V$ by one gradient step on $L_{policy} + c_v L_{value} - c_e L_{entropy}$

    Discard the rollout.
```


## 5.6 Summary

Both DQN and PPO are trained on the same leg-based transitions we \edt{designed} earlier \edt{and the same} reward profiles, so any performance gap we \edt{find in the experiments chapter} comes from the learning algorithm rather than the objective.

One expectation here is that the value-based target regresses
on a bootstrapped estimate, so variance in the reward enters the regression target directly,
whereas the policy-gradient update subtracts a state-dependent value baseline before the step
and removes variance the vehicle's own action did not cause. Where a reward profile carries a
weak action-attributed signal, the latter should prove the more stable of the two. \edt{Benchmarks of cooperative multi-agent learning find independent PPO competitive with, or better than, off-policy value-based methods \cite{Yu2021, Papoudakis2020}, and Papoudakis et al. \cite{Papoudakis2020} point out that independent learners trained from a replay buffer suffer because the stored transitions were generated by policies that have since changed. The value-baseline explanation above is our own hypothesis.} Chapter 6 tests this. 


```{=latex}
\setcounter{chapter}{5}
```

# Experiments and Results

This chapter reports the experiments in the order they were run. We first characterise the \edt{\textbf{\underline{unguided routing baseline}}}\edt{, in which all vehicles route by shortest path with no learned guidance,} to find the demand level at which it collapses. We then train DQN on the 28-dimensional state vector of Chapter 4, which trains reliably only under light load \edt{and deteriorates under heavier congestion}. That failure motivates adding trend and demand-phase features, which restores stable training across higher demands. Finally, we introduce PPO on the same setup and show where it separates from DQN: under reward profiles \edt{where a vehicle's own choice has the least effect on its reward,} and at higher traffic loads. \edt{Section 6.4 then explains the findings in plain terms, with its subsection headings deliberately phrased as questions for clarity.}


## 6.1 Experimental setup

### 6.1.1 Network architectures

Both algorithms use the same \edt{hidden layers} so that model capacity is held constant across comparisons.

| Component | DQN | PPO |
| :--- | :--- | :--- |
| **\edt{Hidden layers}** | \edt{Two layers, 64 units each, ReLU} | \edt{Same, shared by the policy and value outputs} |
| **Output** | One value per region (7) | Policy logits (7) + one state value |
| **Input width** | 28 (or 36 with anticipation) | Same |

: Network architectures for DQN and PPO.

### 6.1.2 Hyperparameters and reproducibility

Table 6.2 lists the hyperparameters held fixed across all experiments\edt{.} PPO uses a lower learning rate, as is standard for on-policy actor-critic updates with larger minibatches. For DQN, $\epsilon$ anneals from $1.0$ to $0.05$ with \edt{optional} periodic resets to maintain exploration under non-stationary traffic.

| Parameter | DQN | PPO |
| :--- | :--- | :--- |
| **Learning rate** | $10^{-3}$ | $3 \times 10^{-4}$ |
| **Discount $\gamma$** | $0.99$ | $0.99$ |
| **Batch / minibatch** | $64$ | $256$ |
| **Replay capacity** | $10{,}000$ | Rollout only |
| **Target update** | Soft ($\tau_{\text{soft}} = 0.005$) | — |
| **Passes per update** | $1$ | $4$ |
| **GAE $\lambda$** | — | $0.95$ |
| **Clip $\epsilon_{clip}$** | — | $0.2$ |
| **Value / entropy weight** | — | $0.5$ / $0.01$ |
| **Gradient clip** | $10.0$ | $0.5$ |
| **Exploration** | $\epsilon: 1.0 \to 0.05$ (\edt{optional} periodic reset) | Entropy bonus |

: Fixed hyperparameters for DQN and PPO.

All runs are seeded for reproducibility, and training scripts, checkpointing utilities, and command-line flags are documented in the project repository.


## 6.2 Evaluation metrics

System reward (network-wide, per event) and vehicle reward (per vehicle, per leg) measure different quantities and cannot be compared on the same axis. A policy can improve its training reward while leaving network-wide travel time worse. Throughout this chapter, we therefore evaluate performance using travel time rather than training reward.

Every result in \edt{this chapter} is reported \edt{with two metrics.}

\edt{The first} is the **travel-time ratio**:
$$\text{ratio} = \frac{\bar{t}_{\text{CAV}}}{\bar{t}_{\text{HDV}}}$$

the mean journey time of the \edt{CAVs} divided by the mean journey time of the \edt{HDVs} in the same run. Because both \edt{classes} experience the same demand realisation and congestion, this ratio isolates the effect of the routing policy from the load. \edt{A ratio of $1.00$ means CAVs match the HDVs' average travel time; values below $1.00$ mean CAVs are faster, and values above $1.00$ mean they are slower. We refer to a ratio close to $1.00$ as \textbf{travel-time parity}, or simply parity, meaning the two journey times are close to one another.}

Because this is an internal comparison within a run, it does not by itself show whether guidance improved conditions network-wide\edt{:} \edt{CAVs} that beat HDVs can still leave the overall network slower.

\edt{The second metric answers that question: whether the learned policy beats shortest-path routing for the network as a whole. It is the mean travel time of all vehicles, CAVs and HDVs together, divided by the mean travel time of the Dijkstra-only baseline at the same demand and random seed, recorded during greedy evaluation:}

$$\text{ratio}_{\text{all}} = \frac{\bar{t}_{\text{all}}}{\bar{t}_{\text{Dijkstra-only}}}$$

\edt{Values below $1.00$ mean the network as a whole is faster than under shortest-path routing, and values above $1.00$ mean it is slower.}

\edt{Reported} ratios are measured at the final completed training episode, with exploration still active for DQN. \edt{At the end of training, we also measure the journey times with exploration turned off ($\epsilon = 0$), so every action is deterministic and reproducible for a given random seed. These results are labelled \textbf{greedy evaluation}.}

## 6.3 The sequence of experiments

### 6.3.1 The unguided baseline

Every experiment below is measured against an unguided baseline in which all vehicles\edt{, CAVs and HDVs alike, route} via Dijkstra's shortest-path algorithm, recomputed at every simulation event \edt{using the current state of the network. We call it the Dijkstra-only baseline, or the unguided baseline for short. It is distinct from the HDVs that drive alongside the CAVs in a trained run, which serve only as the reference of the CAV/HDV ratio and are never called a baseline.} Demand is raised in steps, with five independent random seeds at each level, until the network \edt{fails}.

\edt{Note that this is a demanding baseline rather than one produced by a naive algorithm.} Because shortest paths are rebuilt at every event from \edt{current} regional accumulations, the unguided router works with undelayed network state and zero computation cost\edt{;} the only thing it lacks is \edt{a prediction of the network's future state}. Because it prices current cost, it can route vehicles into a region that is cheap at decision time but congested by the time they arrive\edt{. This is the small margin a learned policy can exploit.}

The demand sweep produces an abrupt threshold rather than a gradual decline (Figures 6.1–6.2). The network \edt{stays below the jam threshold} through a demand scale of $1.30$ and fails on all five seeds at $1.35$, with peak accumulation jumping by roughly $40\%$ for a demand increase of under $4\%$. \edt{A run is counted as failed when its peak accumulation crosses} the jam threshold $n_{\max} = 5536$ vehicles (with critical accumulation at $n_{cr} = 2812$)\edt{. Vehicles still reach their destinations in a failed run, as the completion rate in Figure 6.1 shows, but the most loaded region is jammed for part of the episode and travel times rise sharply.} \edt{These results set the demand levels used in the rest of the chapter:} scales of $0.70$ to $0.90$ sit well inside the stable regime, whereas $1.30$ and above test policies at or past the unguided baseline's limit.

![The demand sweep in full. Faint markers are individual seeds, the line their mean. \edt{The completion-rate plot (bottom right) shows the share of vehicles that reach their destination at each demand. It is 100\% throughout, so the means are free of \textbf{survivorship bias}: no average is taken only over the vehicles that finished while leaving out those still stuck in the network.}](results/dijkstra_baseline/demand_sweep_60cav/plots/hdv_baseline_sweep.png){width=90%}

![The transition at higher resolution. Each point is one seed's peak accumulation as a percentage of jam accumulation; colour marks whether all seeds survived at that demand, some, or none. \edt{Runs begin to fail where peak accumulation crosses the jam threshold.}](results/dijkstra_baseline/demand_sweep_60cav/plots/hdv_baseline_transition.png){width=80%}

### 6.3.2 Training with the 28-dimensional state vector

The first set of DQN runs uses the 28-dimensional state vector of Eq. 4.11 (current region, destination, per-region density, and action mask), varying demand while holding everything else fixed.

| Demand | Ratio, episode 1 | Ratio, last episode | Episodes |
| :---: | :---: | :---: | :---: |
| 0.70 | 1.97 | 1.06 | 767 |
| 0.80 | 1.97 | 2.18 | 381 |
| 0.90 | 2.03 | 2.83 | 138 |
| 1.00 | 2.57 | 4.57 | 119 |

: DQN training with the 28-dimensional state vector of Chapter 4. Ratios are travel-time ratios at the first and last completed training episode.

Only the lightest demand ($0.70$) stabilises. At $0.80$ and above, the ratio worsens with training, ending worse than the untrained network\edt{, and at demand $1.00$ the training loss explodes}. The two heaviest \edt{demand} runs oscillate between roughly $2.0$ and $4.6$ without trend and \edt{were therefore} stopped early (Table 6.3\edt{; Figure 6.7 shows the 28-dimensional state vector diverging at demand $0.80$}).

The likely cause is that \edt{the static picture of regional accumulation available to this policy} stops being sufficient near $n_{cr}$. Because a vehicle commits to a leg lasting several hundred seconds, what matters is not only which region is fuller now, but which region is filling\edt{, which is} information absent from Eq. 4.11.

### 6.3.3 Augmenting the state vector with trend and demand phase

To give the policy information about how congestion is evolving over a leg, we \edt{engineer two additional features of the state observation vector that help the policy anticipate congestion, and append them} to Eq. 4.11.

**Trend.** \edt{This feature models the rate of change of each region's accumulation. It is} estimated as a strictly backward difference over a window $W = 120\text{ s}$ and projected forward over a horizon $H = 500\text{ s}$ (roughly one leg):

$$t_0 = \text{oldest sample} \ge t - W \tag{Eq. 6.1}$$

$$\frac{dn_i}{dt} = \frac{n_i(t) - n_i(t_0)}{t - t_0}, \qquad \hat{\Delta n}_i = \text{clip}\!\left(\frac{dn_i}{dt} \cdot \frac{H}{n_{cr,i}},\; -2,\; +2\right) \tag{Eq. 6.2}$$

Using a backward difference prevents leaking future state into the observation. Scaling by $H / n_{cr,i}$ puts the projection in the same units as the critical-normalised density feature (Eq. 4.12), so $\hat{n}_i + \hat{\Delta n}_i$ estimates density one leg ahead. The finite window $W$ smooths event-to-event noise, and clipping bounds abrupt spikes.

**Demand phase.** \edt{This feature tracks progress through the demand loading window as a single scalar:}

$$\phi = \frac{t}{T_q} \tag{Eq. 6.3}$$

where $T_q = 3600$ s is the length of the demand loading window (Section 4.1). $\phi$ crosses $1.0$ where the network shifts from filling to draining and is left unclipped (episodes run to $\phi \approx 1.9$)\edt{, so the policy knows whether the network as a whole is still loading or draining; the local derivative $dn_i/dt$ only describes each region's recent change.}

Appending the seven trend values and $\phi$ extends Eq. 4.11 from 28 to 36 dimensions:

$$S^+ = [S \,\|\, \hat{\Delta n}_i \,\|\, \phi] \tag{Eq. 6.4}$$

| Demand | Ratio, episode 1 | Ratio, last episode | Episodes | \edt{28-dim, last episode} |
| :---: | :---: | :---: | :---: | :---: |
| 0.70 | 1.98 | 1.03 | 1000 | 1.06 |
| 0.80 | 1.97 | 1.02 | 1000 | 2.18 |
| 0.90 | 2.01 | 1.04 | 1000 | 2.83 |

: DQN training with the 36-dimensional state vector, compared against the \edt{28-dimensional state vector}.

The divergence seen \edt{with the 28-dimensional state vector} disappears: every run settles near parity at demands where the 28-dimensional configuration previously deteriorated (Figure \edt{6.3}). A ratio near $1.00$ represents parity with shortest-path routing rather than an improvement over it, showing that the augmented state vector makes the problem learnable at higher loads.

```{=latex}
\begin{figure}[H]
\centering
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{results/antic_fd070/plots/avg_travel_times.png}
\caption*{DQN, selfish, demand 0.70}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{results/antic_fd090/plots/avg_travel_times.png}
\caption*{DQN, selfish, demand 0.90}
\end{minipage}
\caption{Average travel time per episode with the trend- and phase-augmented state vector, at
\edt{demand levels where the 28-dimensional state vector diverges or barely stabilises (Table 6.3)}. Both now converge
toward parity.}
\end{figure}
```

### 6.3.4 Behaviour profiles under DQN

The four reward profiles of Section 5.3.3 were next trained with DQN on the 36-dimensional state vector at demands $0.80$ and $0.90$.

| Profile | Weight on $t_{\text{own}}$ | Demand 0.80 | Demand 0.90 |
| :--- | :---: | :---: | :---: |
| `selfish` | full | 1.024 | 1.036 |
| `social` | half | 1.026 | 2.708, diverging |
| `collaborative` | half | 1.007 | 1.005 |
| `altruistic` | none | 1.984, diverging | Divergent |

: DQN travel-time ratios at the last completed training episode. Runs marked *diverging* were worsening with training when stopped.

Every profile that retains weight on the vehicle's own travel time stabilises at demand $0.80$, whereas **altruistic**\edt{, which} drops $t_{\text{own}}$ entirely\edt{,} diverges outright. This confirms the credit-assignment argument of Section 5.3.4: a single vehicle's choice barely moves the network-wide average $\bar{t}_{\text{all}}$ across thousands of vehicles \edt{\cite{Lowe2017}}. Once a profile diverges at a lower demand, it is not run at higher loads, since heavier congestion only adds noise to an already uninformative signal.

### 6.3.5 Policy-gradient training (PPO)

PPO was trained on the same 36-dimensional state vector, reward profiles, and environment so that any difference isolates the learning algorithm.

| Profile | Demand 0.80 | Demand 0.90 | Demand 1.30 |
| :--- | :---: | :---: | :---: |
| `selfish` | 0.99 | 1.00 | 0.99 |
| `social` | 1.01 | 1.01 | 1.00 |
| `collaborative` | 1.00 | 1.00 | 1.00 |
| `altruistic` | 1.11, stable | 1.16, stable | 1.35, worsening |

: PPO travel-time ratios at the last completed training episode (compare DQN in Table \edt{6.5}).

Where DQN diverged on **altruistic**, PPO holds a stable plateau above parity at demands $0.80$ and $0.90$ before worsening at $1.30$. To classify these outcomes cleanly in Table \edt{6.7}, we distinguish three states: **converged** (stable at or below parity), **weak** (a stable plateau above parity), and **divergent** (worsening with training). **altruistic** is the only profile that passes through all three as demand rises. \edt{Figure 6.4 shows the first two side by side at demand $0.90$.}

```{=latex}
\begin{figure}[H]
\centering
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/ch6/ppo_profiles_fd090_cav_hdv_ratio.png}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/ch6/ppo_profiles_fd090_all_vs_dijkstra.png}
\end{minipage}
\caption{\edt{PPO training at demand $0.90$, one line per reward profile. Left: CAV/HDV. Right: all vehicles against the Dijkstra-only mean, not paired by seed; the grey band is the seed-to-seed spread of that baseline.}}
\end{figure}
```

The demand sweep also extends past the unguided baseline's collapse point: **selfish** holds parity at $1.00$, $1.20$, $1.30$, and $1.35$\edt{,} the load at which every seed of the unguided baseline failed (Figure \edt{6.5}). At $1.30$, peak accumulation in the worst-hit region still reaches roughly $1.6\,n_{cr}$ (on the falling branch of $g(n)$), but remains at $0.83\,n_{\max}$, well short of gridlock. The policy does not prevent regions from entering the falling branch, but it holds parity through recovery.

The separation between DQN and PPO appears specifically on **altruistic** (Figure \edt{6.6}), because PPO's state-dependent value baseline removes variance the vehicle's own action did not cause, whereas DQN's bootstrapped target absorbs that variance directly \edt{\cite{Papoudakis2020, Yu2021}}.

```{=latex}
\begin{figure}[H]
\centering
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{results/ippo_selfish_fd080/plots/avg_travel_times.png}
\caption*{PPO, selfish, demand 0.80 (well inside the stable regime)}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{results/ippo_selfish_fd135/plots/avg_travel_times.png}
\caption*{PPO, selfish, demand 1.35 (the \edt{unguided baseline's} collapse point)}
\end{minipage}
\caption{Average travel time per episode, well below and exactly at the load where the
\edt{unguided} baseline fails on every seed. The trained policy converges to parity at
both.}
\end{figure}
```

```{=latex}
\begin{figure}[H]
\centering
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{results/antic_fd080_altruistic/plots/avg_travel_times.png}
\caption*{DQN, altruistic, demand 0.80 (diverges)}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{results/ippo_altruistic_fd080/plots/avg_travel_times.png}
\caption*{PPO, altruistic, demand 0.80 (weak but stable)}
\end{minipage}
\caption{The same profile, demand and state vector, trained by each algorithm in turn. \edt{DQN}
diverges on the zero-own-time reward; \edt{PPO} holds a
stable plateau above parity instead.}
\end{figure}
```

### 6.3.6 Summary of experimental coverage

Table \edt{6.7} collects all combinations of demand, behaviour profile, and algorithm tested on the seven-region network at $\rho_{\text{CAV}} = 0.6$. The first column uses the 28-dimensional state vector (Eq. 4.11, jam-normalised); all other columns use the 36-dimensional state vector (Eq. \edt{6.4}, critical-normalised with trend and demand phase). \edt{Table 6.8 turns the same evaluations into two ratios: CAV against HDV within each run, and the mean travel time of all vehicles against the Dijkstra-only baseline at the same seeds.}

```{=latex}
\begin{landscape}
```

| Demand | \edt{Dijkstra-only} | DQN `selfish` (28-dim) | DQN `selfish` (36-dim) | DQN `social` (36-dim) | DQN `collab.` (36-dim) | DQN `altruistic` (36-dim) | PPO `selfish` (36-dim) | PPO `social` (36-dim) | PPO `collab.` (36-dim) | PPO `altruistic` (36-dim) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.80** | 1120 | Divergent | 1122:1119 | 1120:1120 | 1121:1120 | Divergent | 1121:1120 | 1121:1120 | 1119:1120 | 1265:1135 |
| **0.90** | 1159 | Divergent | 1162:1161 | Divergent | 1161:1160 | Divergent | 1159:1160 | 1159:1159 | 1159:1160 | 1353:1183 |
| **1.00** | 1203 | Divergent | Divergent | Divergent | Divergent | Divergent | 1203:1203 | 1203:1203 | 1203:1204 | 1327:1217 |
| **1.20** | 1327 | Divergent | Divergent | Divergent | Divergent | Divergent | 1329:1326 | 1327:1326 | 1328:1326 | 1448:1354 |
| **1.30** | 1476 | Divergent | Divergent | Divergent | Divergent | Divergent | 1475:1481 | 1490:1495 | 1474:1479 | Divergent |

: \edt{Full experimental grid at $\rho_{\text{CAV}} = 0.6$. Each cell gives the mean travel time (s) of CAVs : HDVs under greedy evaluation, averaged over seeds 5000 to 5004. \emph{Dijkstra-only}: mean travel time of all vehicles when every vehicle routes by Dijkstra, same seeds. \emph{Divergent}: worsened with training, or not run because it diverged at a lower demand. —: not attempted.}

```{=latex}
\begin{table}[H]
\centering
\caption{\edt{The evaluations of Table 6.7 as two ratios, each averaged over the five seeds. \emph{CAV/HDV}: mean CAV travel time over mean HDV travel time in the same run. \emph{All/Dijk.}: mean travel time of all vehicles over the Dijkstra-only mean at the same seed. Below 1.000 means faster.}}
\footnotesize
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{c|cc|cc|cc|cc|cc|cc|cc|cc|cc}
\toprule
 & \multicolumn{2}{c|}{\shortstack{DQN\\\texttt{selfish}\\28-dim}} & \multicolumn{2}{c|}{\shortstack{DQN\\\texttt{selfish}\\36-dim}} & \multicolumn{2}{c|}{\shortstack{DQN\\\texttt{social}\\36-dim}} & \multicolumn{2}{c|}{\shortstack{DQN\\\texttt{collab.}\\36-dim}} & \multicolumn{2}{c|}{\shortstack{DQN\\\texttt{altruistic}\\36-dim}} & \multicolumn{2}{c|}{\shortstack{PPO\\\texttt{selfish}\\36-dim}} & \multicolumn{2}{c|}{\shortstack{PPO\\\texttt{social}\\36-dim}} & \multicolumn{2}{c|}{\shortstack{PPO\\\texttt{collab.}\\36-dim}} & \multicolumn{2}{c}{\shortstack{PPO\\\texttt{altruistic}\\36-dim}} \\
Demand & \shortstack{CAV/\\HDV} & \shortstack{All/\\Dijk.} & \shortstack{CAV/\\HDV} & \shortstack{All/\\Dijk.} & \shortstack{CAV/\\HDV} & \shortstack{All/\\Dijk.} & \shortstack{CAV/\\HDV} & \shortstack{All/\\Dijk.} & \shortstack{CAV/\\HDV} & \shortstack{All/\\Dijk.} & \shortstack{CAV/\\HDV} & \shortstack{All/\\Dijk.} & \shortstack{CAV/\\HDV} & \shortstack{All/\\Dijk.} & \shortstack{CAV/\\HDV} & \shortstack{All/\\Dijk.} & \shortstack{CAV/\\HDV} & \shortstack{All/\\Dijk.} \\
\midrule
\textbf{0.80} & \multicolumn{2}{c|}{Divergent} & 1.002 & 1.001 & 1.000 & 1.000 & 1.001 & 1.001 & \multicolumn{2}{c|}{Divergent} & 1.001 & 1.000 & 1.001 & 1.000 & 0.999 & 1.000 & 1.114 & 1.083 \\
\textbf{0.90} & \multicolumn{2}{c|}{Divergent} & 1.000 & 1.002 & \multicolumn{2}{c|}{Divergent} & 1.001 & 1.001 & \multicolumn{2}{c|}{Divergent} & 0.999 & 1.000 & 0.999 & 1.000 & 1.000 & 1.000 & 1.143 & 1.108 \\
\textbf{1.00} & \multicolumn{2}{c|}{Divergent} & \multicolumn{2}{c|}{Divergent} & \multicolumn{2}{c|}{Divergent} & \multicolumn{2}{c|}{Divergent} & \multicolumn{2}{c|}{Divergent} & 1.000 & 1.000 & 1.000 & 1.000 & 1.000 & 1.000 & 1.091 & 1.067 \\
\textbf{1.20} & \multicolumn{2}{c|}{Divergent} & \multicolumn{2}{c|}{Divergent} & \multicolumn{2}{c|}{Divergent} & \multicolumn{2}{c|}{Divergent} & \multicolumn{2}{c|}{Divergent} & 1.003 & 1.001 & 1.001 & 0.999 & 1.002 & 1.000 & 1.070 & 1.063 \\
\textbf{1.30} & \multicolumn{2}{c|}{Divergent} & \multicolumn{2}{c|}{Divergent} & \multicolumn{2}{c|}{Divergent} & \multicolumn{2}{c|}{Divergent} & \multicolumn{2}{c|}{Divergent} & 0.996 & 1.001 & 0.997 & 1.011 & 0.997 & 1.000 & \multicolumn{2}{c}{Divergent} \\
\bottomrule
\end{tabular}}
\end{table}
\end{landscape}
```

Even with the 36-dimensional state vector, DQN breaks down at higher loads: **collaborative** diverges by demand $1.00$, and **selfish** diverges between $0.90$ and $1.20$. PPO does not hit this ceiling, holding parity through demand $1.30$ across **selfish**, **social**, and **collaborative**.


## 6.4 Findings and analysis

### A. How does feature engineering help this problem? {-}

The 28-dimensional state vector of Eq. 4.11 gives each vehicle only a snapshot of current regional occupancy. Because a routing decision commits a vehicle to a leg lasting several hundred seconds during high loads, that snapshot is outdated by the time the vehicle crosses the region. Above light load (demand $0.70$), DQN trained on Eq. 4.11 deteriorates rather than stabilises. Appending regional accumulation trends and demand phase (Eq. 6.4) supplies the missing forward-looking context. With it, DQN holds parity up to demand $0.90$ but diverges at every higher demand, and diverges further as demand rises; PPO holds parity up to demand $1.30$, the highest demand in Table 6.7.

Two things changed between the 28-dimensional and 36-dimensional state vectors: the two anticipation features were added, and density was switched from jam to critical normalisation. A single ablation run at demand $0.80$, on the same seven-region network, separates the two (Figure \edt{6.7}). Using critical normalisation without the trend and phase features, DQN still diverges: its CAV/HDV ratio averages $2.39$ over the last ten training episodes and $1.29$ under greedy evaluation, against $1.002$ for the full 36-dimensional vector. The gain therefore comes from the anticipation features, not from normalisation.

```{=latex}
\begin{figure}[H]
\centering
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/ch6/ablation_antic_fd080_cav_hdv_ratio.png}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/ch6/ablation_antic_fd080_all_vs_dijkstra.png}
\end{minipage}
\caption{\edt{DQN at demand $0.80$, first 100 training episodes, for the three state vectors. Left: CAV/HDV. Right: all vehicles against the Dijkstra-only mean, not paired by seed (grey band as in Figure \edt{6.4}).}}
\end{figure}
```

### B. Why does PPO outperform DQN? {-}

Both algorithms were trained on the same state vector, reward profiles, and simulator, so the performance gap comes mainly from the learning method. At demand $0.80$, DQN and PPO separate only on **altruistic** (Figure \edt{6.6})\edt{, the one profile whose reward leaves out the vehicle's own travel time; why that profile is hard to learn is the subject of the next question}. From demand $0.90$ upwards, they separate on every profile: DQN diverges on **social** at $0.90$, on **collaborative** from $1.00$, and on **selfish** at $1.20$ and $1.30$, while PPO holds parity on all three (Table 6.7). \edt{Figure 6.8 shows the \textbf{selfish} case at demand $1.20$, with both algorithms on a common scale; by episode $100$ DQN reaches a CAV/HDV ratio of $2.34$ (10-episode mean), while PPO holds parity.} PPO's tolerance is not unlimited: it also breaks down on **altruistic** at $1.30$ (Figure \edt{6.9}\edt{; CAV/HDV $1.354$ at episode $100$}), as heavier congestion adds more noise to an already weak reward signal.

```{=latex}
\begin{figure}[H]
\centering
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/ch6/dqn_vs_ppo_fd120_cav_hdv_ratio.png}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/ch6/dqn_vs_ppo_fd120_all_vs_dijkstra.png}
\end{minipage}
\caption{\edt{DQN and PPO at demand $1.20$ on the same profile (\textbf{selfish}) and state vector, up to 100 episodes. Left: CAV/HDV. Right: all vehicles against the Dijkstra-only mean, not paired by seed (grey band as in Figure 6.4).}}
\end{figure}
```

Two mechanisms could explain this gap: PPO's advantage baseline, or replay-buffer staleness in DQN. Before updating its policy, PPO subtracts a state-dependent value estimate, which strips out variance that the vehicle's own action did not cause; DQN's bootstrapped target passes that variance directly into its value regression. Alternatively, DQN's replay buffer can go stale as thousands of vehicles update their shared policy together, whereas PPO discards each rollout after a few passes (Section 5.5.2). Our results are more consistent with the value-baseline explanation. Each episode generates more than $15{,}000$ decisions, so a $10{,}000$-transition buffer is replaced more than once within every episode, which keeps staleness bounded. More importantly, the gap appears first at demand $0.80$ on the single profile with a weak credit signal, and spreads to the remaining profiles only as rising demand adds reward noise, a pattern expected from target variance rather than uniform buffer staleness.

This pattern refines existing multi-agent benchmarks. Prior studies report PPO as matching or outperforming DQN across cooperative tasks in general \cite{Papoudakis2020, Yu2021}. Here, both algorithms perform identically when the load is light and the reward carries a clear local signal (demand $0.80$ on **selfish**, **social**, and **collaborative**). PPO's practical advantage is resilience to a weak or noisy credit signal, whether that weakness comes from the reward definition or from heavy congestion.

### C. How does reward shaping affect learning? {-}

\edt{The algorithm comparison showed that the gap between DQN and PPO opens first on \textbf{altruistic}. This question looks at why that profile is so much harder to learn than the others.} Under PPO, every reward profile that includes the vehicle's own travel time settles at parity with HDVs across all demands tested, and all three perform almost identically once settled (Table 6.7). Removing own travel time (**altruistic**) prevents both algorithms from reaching parity. DQN diverges outright at demand $0.80$. PPO holds a stable plateau above parity from $0.80$ to $1.20$ and diverges by demand $1.30$ (Figure \edt{6.9}).

That plateau degrades the whole network, not just the CAVs. At demand $0.80$, PPO **altruistic** CAVs average $1265\text{ s}$ and HDVs $1135\text{ s}$, against $1120\text{ s}$ when every vehicle routes by Dijkstra (Table 6.7). Across all vehicles, mean travel time sits $6\%$ to $11\%$ above the Dijkstra-only baseline from demand $0.80$ to $1.20$ (Table 6.8), and $89\%$ above it at $1.30$.

```{=latex}
\begin{figure}[H]
\centering
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/ch6/ppo_profiles_fd130_cav_hdv_ratio.png}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/ch6/ppo_profiles_fd130_all_vs_dijkstra.png}
\end{minipage}
\caption{\edt{PPO training at demand $1.30$, one line per reward profile, \edt{up to 100 episodes}. Left: CAV/HDV. Right: all vehicles against the Dijkstra-only mean, not paired by seed (grey band as in Figure \edt{6.4}).}}
\end{figure}
```

This failure is most likely a structural credit-assignment problem rather than a tuning issue, since the same settings succeed for every profile that includes own travel time. One vehicle barely shifts a network-wide average travel time, so the policy cannot separate the effect of its own choice from the simultaneous choices of thousands of other vehicles. \edt{Lowe et al. \cite{Lowe2017} show that the variance of policy-gradient estimates grows with the number of agents, which drowns out the local learning signal.} A reward term therefore teaches a policy only if the vehicle's own action measurably changes it, regardless of how accurately that term expresses the system-level goal.

\edt{Similar failures have been reported for learned routing. In the Urban Routing Benchmark, Akman et al. \cite{Akman2025} find that independent Q-learning and MAPPO fail to converge and end close to random routing. Our DQN fails in a similar way, ending worse than the untrained policy it started from, but PPO settles on a stable policy that performs clearly better than the one it started from.}

\edt{The optional shaping terms of Section 5.3.3, marginal cost and externality pricing, are not reported in this chapter, because on this network they do not produce a meaningful UE-SO gap to close.}

### D. Does any configuration beat the Dijkstra-only baseline, and if not, why? {-}

Under five-seed deterministic greedy evaluation (Section 4.4.5), no configuration beats the HDVs within its own run by more than the seed-to-seed variation of the unguided baseline. The closest, PPO **selfish** at demand $1.30$, has a CAV/HDV ratio of $0.996$: a mean difference of $5.7\text{ s}$ on a trip of almost 25 minutes, against a seed-to-seed standard deviation of $34\text{ s}$ in the unguided baseline at that demand. \edt{Henderson et al. \cite{Henderson2018} show that deep reinforcement learning results can shift substantially between random seeds, and Agarwal et al. \cite{Agarwal2021} argue that with few runs, comparisons should rest on the spread of results rather than the mean alone. This is why the seed-to-seed variation serves as the threshold here.}

Comparing all vehicles against the Dijkstra-only baseline at the same seeds (Table 6.8) gives the same result. At demand $1.30$, the ratio of the mean travel time of all vehicles to the mean travel time when every vehicle routes by Dijkstra is $1.001$ for **selfish**, $1.000$ for **collaborative**, and $1.011$ for **social**. Every settled configuration therefore lands at parity with the unguided baseline overall.

Three factors explain why beating this baseline is difficult. First, the unguided baseline recomputes shortest paths at every event from the current, undelayed network state, leaving only a narrow foresight margin right before collapse. Second, shortest-path routing approximates a user equilibrium, and **selfish** optimises the same individual travel-time metric, while **social** and **collaborative** retain enough weight on own travel time to stay close to it. Third, on this network the shortest-path route and the system-optimal route coincide: routing by marginal rather than average travel time selects the same path for all 42 origin-destination pairs, so there is little room for better routing for any profile to find.

### E. Is there a reliable system optimum on this network? {-}

Every routing method tested in this chapter ties shortest-path routing on total travel time. All of them act only on the present state of the network. That result admits two explanations. Either there is little to gain on this network, because the system optimum lies close to the user equilibrium, or a gain exists but the routers are too short-sighted to find it. To separate the two, we trained a policy by hindsight optimisation, referred to here as the **oracle policy**. It is allowed to see the future of one fixed scenario, and it is trained directly on total system travel time.

**Freezing the scenario.** A scenario has a fixed future only if the same random seed produces the same departures, destinations and trip lengths whatever the CAVs do. In the standard simulator, some random draws are made during the run, such as the trip length remaining in the destination region, and a change of route alters their order, so every later vehicle sees a different future. For the oracle experiments, these random values are therefore tied to each individual vehicle, so every vehicle receives the same trip length and other draws no matter how it or any other vehicle is routed. Replaying shortest-path routing on a frozen scenario reproduced the original total travel time with no measurable difference.

**What the oracle sees.** At each boundary decision, the oracle policy receives the 28-dimensional state vector of Eq. 4.11, extended by the accumulation each region *will* reach $\tau$ seconds later, where $\tau$ takes each value in the set of lookahead horizons $\mathcal{T} = \{120, 300, 600, 1200\}$ s, normalised by critical accumulation, together with the demand phase $\phi$, as in the 36-dimensional state vector of Eq. 6.4. Eq. 6.5 shows the result.

$$S^{\text{oracle}} = \Big[\ S\ \Big\|\ \frac{n_i^{\text{ref}}(t+\tau)}{n_{cr,i}}\Big|_{i = 1..7,\ \tau \in \mathcal{T}}\ \Big\|\ \phi\ \Big] \tag{Eq. 6.5}$$

The oracle state vector therefore has 57 dimensions, made up of the 28 of Eq. 4.11, 28 lookahead values (seven regions at four horizons) and the phase. The future accumulation $n^{\text{ref}}$ cannot be the current episode's own future, which is unknown while the episode runs. It is a reference trajectory, initialised with the shortest-path episode and updated after every episode $e$ towards the trajectory just realised, as in Eq. 6.6.

$$n^{\text{ref}}_{e+1}(t) = 0.5\, n^{\text{ref}}_{e}(t) + 0.5\, n_{e}(t) \tag{Eq. 6.6}$$

The reference is therefore an estimate of the future, not the future itself. Because the scenario is frozen, only routing differs between episodes, so the estimate is accurate to the extent that consecutive episodes route alike. It would equal true hindsight only if the policy stopped changing. A policy that still samples its actions never quite reaches that point, so the lookahead remains an approximation.

**What the oracle optimises.** Unlike the per-vehicle rewards of Chapter 5, the oracle is trained on total system travel time itself. With $C(t)$ the vehicle-seconds accumulated by all vehicles up to time $t$, each decision $d$ taken at time $t_d$ receives the system travel time accrued after it, undiscounted and scaled by the baseline total $TTS_B$, as in Eq. 6.7.

$$G_d = -\,\frac{C_{\text{end}} - C(t_d)}{TTS_B} \tag{Eq. 6.7}$$

Here $C_{\text{end}}$ is the total at the end of the episode, so $C_{\text{end}} - C(t_d)$ is the travel time all vehicles accrue after the decision, and $TTS_B$ is the total travel time of the shortest-path episode in vehicle-seconds, which only rescales $G_d$ to lie roughly between $-1$ and $0$. Time accrued before the decision is left out, because the decision cannot have caused it.

The oracle policy is trained with PPO. The learning rate ($3 \times 10^{-4}$), the value weight ($0.5$) and the four passes over each batch are the same as in the PPO runs of Table 6.2\edt{; the other settings differ}. The clip is $\epsilon_{clip} = 0.1$ rather than $0.2$ and the entropy weight is $0.003$ rather than $0.01$, so each update moves the policy only a little away from shortest-path routing. We set $\gamma = 1$ because the oracle is scored on the total travel time of the whole episode, which is not discounted. Each update uses four episodes in minibatches of $4{,}096$ decisions, and the network has two hidden layers of 128 units.

**Why the result bounds the gap.** Every episode the oracle produces is a feasible routing of the frozen scenario. The best episode therefore shows at least how far the system optimum lies below shortest-path routing on that scenario, measured as the relative gap $\Delta$ of Eq. 6.8.

$$\Delta = \frac{TTS_{B} - TTS_{\text{oracle}}}{TTS_{B}} \times 100\% \tag{Eq. 6.8}$$

Here $TTS_B$ is the total travel time, summed over all vehicles, when every vehicle routes by shortest path, and $TTS_{\text{oracle}}$ is the total travel time of an oracle episode on the same frozen scenario. Positive values of $\Delta$ mean the oracle policy found a faster routing. Table 6.9 also reports the same comparison as a ratio, $TTS_{\text{oracle}} / TTS_B$, averaged over the two seeds, where values below $1.000$ mean faster than shortest-path routing, as in Table 6.8. \edt{Demand $1.35$ is omitted because the network exceeded jam accumulation in every episode on both seeds.}

| Demand | Best episode, seed 42 | Best episode, seed 1234 | Trained policy, seed 42 | Trained policy, seed 1234 | Oracle / Dijkstra-only, best episode (mean) | Oracle / Dijkstra-only, trained policy (mean) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.70** | $+0.01\%$ | $+0.01\%$ | $-0.02\%$ | $-0.07\%$ | $1.000$ | $1.001$ |
| **0.80** | $+0.01\%$ | $+0.01\%$ | $-0.16\%$ | $-0.03\%$ | $1.000$ | $1.001$ |
| **0.90** | $+0.01\%$ | $+0.02\%$ | $-0.27\%$ | $-0.05\%$ | $1.000$ | $1.002$ |
| **1.00** | $-0.06\%$ | $+0.01\%$ | $-0.95\%$ | $-0.03\%$ | $1.000$ | $1.005$ |
| **1.20** | $+0.03\%$ | $+0.03\%$ | $-0.21\%$ | $-0.07\%$ | $1.000$ | $1.001$ |
| **1.30** | $+4.12\%$ | $+0.06\%$ | $+1.24\%$ | $-0.24\%$ | $0.979$ | $0.995$ |

: Oracle policy against shortest-path routing on frozen scenarios. \edt{Percentage columns: $\Delta$ of Eq. 6.8 per seed (positive is faster). Last two columns: $TTS_{\text{oracle}} / TTS_B$ averaged over both seeds (below $1.000$ is faster). \emph{Best episode}: lowest total travel time in 120 training episodes. \emph{Trained policy}: mean of the last three deterministic evaluations.}

\edt{Up to demand 1.20, the oracle policy finds no routing faster than shortest-path routing.} On both seeds its best episode lies within $0.03\%$ of shortest-path routing, and the trained policy ends slightly slower. A policy that sees the future and optimises total travel time directly cannot find a better routing, because none exists at these loads. Below critical accumulation, the user equilibrium and the system optimum coincide on this network. This matches the path comparison in D, in which routing by marginal and by average travel time selects the same path for all 42 origin-destination pairs.

At demand 1.30, a gain appears on one of the two scenarios but not the other, and even there the largest gain comes from a sampled episode of a policy that knows the future. A system optimum distinct from the user equilibrium can therefore appear near breakdown, but it depends on the particular scenario.

No routing method therefore reliably beats shortest-path routing by a significant margin on this network. Below critical accumulation there is no gap to exploit, and near breakdown the gap depends on the scenario and is found only with hindsight. Table 6.8 raises the question of whether the parity of the learned policies is a failure of learning. It is not, because there is little evidence of a significant margin between selfish routing and the system optimum for any real-time routing policy, learned or otherwise, to exploit.

## 6.5 Conclusion

The augmented state vector made the routing problem learnable above light load. Adding accumulation trends and demand phase to the occupancy snapshot pushed the onset of DQN divergence from demand $0.80$, where the 28-dimensional state vector already diverges, to above $0.90$, and an ablation at demand $0.80$ confirms that these features, not critical normalisation, drive the fix. With them, DQN holds parity with shortest-path routing up to demand $0.90$ and diverges above it, while PPO holds parity up to demand $1.30$, the highest demand in Table 6.7.

Reward design determined whether a profile could be learned. Under PPO, every profile that kept a weight on the vehicle's own travel time settled at parity, whereas dropping it (**altruistic**) diverged under DQN and slowed all vehicles by $6\%$ to $11\%$ under PPO before diverging at demand $1.30$. A reward only teaches what a single vehicle's action measurably changes. Between the two algorithms, DQN and PPO matched only at light load with a clear local signal; they separated on **altruistic** at $0.80$ and on every profile from $0.90$ upwards, a pattern consistent with PPO's value baseline filtering out variance that DQN's bootstrapped target absorbs directly.

No configuration beat the unguided baseline. At every demand tested, every converged configuration stayed within the seed-to-seed variation of the Dijkstra-only baseline, on both the CAV/HDV ratio and the travel time of all vehicles.

\edt{The oracle policy explains this result. Although it knew the future and was rewarded directly on total travel time, it found no better routing than shortest path below demand $1.30$, and at $1.30$ it gained on only one of the two scenarios. On this network there is little room between the user equilibrium and the system optimum for any real-time routing policy to use.}


## 6.6 Current work in progress and future work planned

\edt{We are currently repeating these experiments on a larger, 19-region network, with heterogeneous traffic conditions designed to produce a larger UE-SO gap.}

\edt{Two curriculum schedules that build on the externality term are also under way, one for each algorithm. Both keep the vehicle's own travel time $t_{own}$ at full weight, since removing it was shown to fail, and add the shaping term gradually instead of from the first episode. The DQN curriculum raises the externality weight linearly from zero to its target over a fixed number of episodes while holding the own-time weight constant. The PPO curriculum ramps the own-time and network-average weights together with the externality weight, but never lets the own-time weight reach zero, which would recreate the altruistic profile. Neither curriculum had produced results at the time of writing.}

<!-- Working note, not rendered: pandoc drops HTML comments.

```{=latex}
\newpage
```

## Run ID map (working note, remove before submission) {-}

Where each cell of the full experimental grid of Section 6.3.6 comes from in the local `results/` folder. Training ratios are read from `results/<run>/metrics/episode_metrics.csv`, and evaluation ratios from `results/greedy_eval/<run>/aggregate_*_summary.json`. Cells marked *Divergent* above a configuration's first divergent demand, and cells marked `—`, have no run. Two runs appear only in Tables 6.3 and 6.4, not in the grid: `crit_fd070` and `antic_fd070` (demand 0.70).

| Table 6.7 column | Demand | Training run (`results/`) | Greedy evaluation (`results/greedy_eval/`) |
| :--- | :---: | :--- | :--- |
| DQN `selfish` (28-dim) | 0.80 | `crit_fd080` | — |
| DQN `selfish` (28-dim) | 0.90 | `crit_fd090` | — |
| DQN `selfish` (28-dim) | 1.00 | `crit_fd100` | — |
| DQN `selfish` (36-dim) | 0.80 | `antic_fd080` | `antic_fd080` |
| DQN `selfish` (36-dim) | 0.90 | `antic_fd090` | `antic_fd090` |
| DQN `selfish` (36-dim) | 1.20 | `iql_selfish_fd120` | `iql_selfish_fd120` |
| DQN `selfish` (36-dim) | 1.30 | `iql_selfish_fd130` | `iql_selfish_fd130` |
| DQN `social` (36-dim) | 0.80 | `antic_fd080_social` | `antic_fd080_social` |
| DQN `social` (36-dim) | 0.90 | `antic_fd090_social` | `antic_fd090_social` |
| DQN `collab.` (36-dim) | 0.80 | `antic_fd080_collaborative` | `antic_fd080_collaborative` |
| DQN `collab.` (36-dim) | 0.90 | `iql_collaborative_fd090` | `iql_collaborative_fd090` |
| DQN `collab.` (36-dim) | 1.00 | `iql_collaborative_fd100` | `iql_collaborative_fd100` |
| DQN `collab.` (36-dim) | 1.20 | `iql_collaborative_fd120` | `iql_collaborative_fd120` |
| DQN `collab.` (36-dim) | 1.30 | `iql_collaborative_fd130` | `iql_collaborative_fd130` |
| DQN `altruistic` (36-dim) | 0.80 | `antic_fd080_altruistic` | `antic_fd080_altruistic` |
| PPO `selfish` (36-dim) | 0.80 | `ippo_selfish_fd080` | `ippo_selfish_fd080` |
| PPO `selfish` (36-dim) | 0.90 | `ippo_selfish_fd090` | `ippo_selfish_fd090` |
| PPO `selfish` (36-dim) | 1.00 | `ippo_selfish_fd100` | `ippo_selfish_fd100` |
| PPO `selfish` (36-dim) | 1.20 | `ippo_selfish_fd120` | `ippo_selfish_fd120` |
| PPO `selfish` (36-dim) | 1.30 | `ippo_selfish_fd130` | `ippo_selfish_fd130` |
| PPO `social` (36-dim) | 0.80 | `ippo_social_fd080` | `ippo_social_fd080` |
| PPO `social` (36-dim) | 0.90 | `ippo_social_fd090` | `ippo_social_fd090` |
| PPO `social` (36-dim) | 1.00 | `ippo_social_fd100` | `ippo_social_fd100` |
| PPO `social` (36-dim) | 1.20 | `ippo_social_fd120` | `ippo_social_fd120` |
| PPO `social` (36-dim) | 1.30 | `ippo_social_fd130` | `ippo_social_fd130` |
| PPO `collab.` (36-dim) | 0.80 | `ippo_collaborative_fd080` | `ippo_collaborative_fd080` |
| PPO `collab.` (36-dim) | 0.90 | `ippo_collaborative_fd090` | `ippo_collaborative_fd090` |
| PPO `collab.` (36-dim) | 1.00 | `ippo_collaborative_fd100` | `ippo_collaborative_fd100` |
| PPO `collab.` (36-dim) | 1.20 | `ippo_collaborative_fd120` | `ippo_collaborative_fd120` |
| PPO `collab.` (36-dim) | 1.30 | `ippo_collaborative_fd130` | `ippo_collaborative_fd130` |
| PPO `altruistic` (36-dim) | 0.80 | `ippo_altruistic_fd080` | `ippo_altruistic_fd080` |
| PPO `altruistic` (36-dim) | 0.90 | `ippo_altruistic_fd090` | `ippo_altruistic_fd090` |
| PPO `altruistic` (36-dim) | 1.00 | `ippo_altruistic_fd100` | `ippo_altruistic_fd100` |
| PPO `altruistic` (36-dim) | 1.20 | `ippo_altruistic_fd120` | `ippo_altruistic_fd120` |
| PPO `altruistic` (36-dim) | 1.30 | `ippo_altruistic_fd130` | `ippo_altruistic_fd130` |

-->


