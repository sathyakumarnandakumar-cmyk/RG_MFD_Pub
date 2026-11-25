function [main_list, guidance_opt, theta_HDV_out, theta_CAV_out] = update_main_list_latest(main_list, previous_time, previous_speed, current_time, current_speed, region_data, guidance_Init, tHp, Hp, Hc, tfinq, Q_S_sub, current_sub_n, N_mat_in)
%%%%% 终极绝对连通兼容引擎：直连寻路+柔性诱导折扣+原生拓扑安全兜底版
global nb_region nb_subregion flag_No_Control flag_MPC reg_DGraph sub2reg CAV_penetration sub_DGraph subregion_data route_guidance_indices control_number

% 1. 动态计算各大区实时积累量
n_total_macro = zeros(nb_region, 1);
for reg = 1:nb_region
    n_total_macro(reg) = sum(current_sub_n(sub2reg == reg));
end

% 2. 微观需求向上积分为宏观需求矩阵
time_steps = size(Q_S_sub, 2);
Q_S_macro = zeros(nb_region * nb_region, time_steps);
for r = 1:nb_subregion
    ori_m = sub2reg(r);
    for h = 1:nb_subregion
        des_m = sub2reg(h);
        macro_row = (ori_m - 1) * nb_region + des_m;
        sub_row   = (r - 1) * nb_subregion + h;
        Q_S_macro(macro_row, :) = Q_S_macro(macro_row, :) + Q_S_sub(sub_row, :);
    end
end

% 实时起终点初始状态构建
if nargin < 14 || isempty(N_mat_in)
    N_mat_init = zeros(nb_region, nb_region);
    for k = 1:length(main_list)
        c_r = main_list(k).current_location_reg;
        d_r = main_list(k).Destination_reg;
        if c_r > 0 && d_r > 0
            N_mat_init(c_r, d_r) = N_mat_init(c_r, d_r) + 1;
        end
    end
else
    N_mat_init = N_mat_in;
end

% 3. 上层路由推导与控制求解
n_HDV_macro = (1 - CAV_penetration) * n_total_macro;
[theta_HDV_macro, shortest_paths_ij, prob_path_ij, ~] = route_choice(reg_DGraph, n_HDV_macro, region_data, subregion_data);

Tc = 60; 
if flag_MPC == 1 && flag_No_Control == 0 && any(n_total_macro > region_data(1).ncriticalMFD)
    guidance_opt = MPC_CAV_routing(tHp, Hp, Hc, Tc, tfinq, Q_S_macro, guidance_Init, N_mat_init, current_time, region_data, theta_HDV_macro, shortest_paths_ij, route_guidance_indices);
else
    guidance_opt = guidance_Init;
end

theta_HDV = prob_path_ij;
theta_CAV_macro = cell(nb_region, nb_region);
for i = 1:nb_region
    for j = 1:nb_region
        if i ~= j
            if flag_No_Control == 1
                theta_CAV_macro{i,j} = theta_HDV{i,j};
            else
                if exist('route_guidance_indices', 'var') && ~isempty(route_guidance_indices{i,j})
                    idx = route_guidance_indices{i,j}; parsed_theta = max(0, guidance_opt(idx));
                    if sum(parsed_theta) > 0, theta_CAV_macro{i,j} = parsed_theta / sum(parsed_theta); else, theta_CAV_macro{i,j} = ones(1, length(parsed_theta)) / length(parsed_theta); end
                else, theta_CAV_macro{i,j} = theta_HDV{i,j}; end
            end
        end
    end
end

theta_HDV_out = theta_HDV_macro;
theta_CAV_out = zeros(nb_region, nb_region, nb_region);
for i = 1:nb_region
    for j = 1:nb_region
        if i ~= j
            probs_c = theta_CAV_macro{i,j};
            paths_c = shortest_paths_ij{i,j};
            for k = 1:length(probs_c)
                if k <= length(paths_c) && length(paths_c{k}) >= 2
                    next_h = paths_c{k}(2);
                    theta_CAV_out(i, j, next_h) = theta_CAV_out(i, j, next_h) + probs_c(k);
                end
            end
        end
    end
end

% 动态重构跳数矩阵，添加尺寸校验，彻底打破旧版 persistent 内存死锁陷阱
persistent sub_dist_mat
if isempty(sub_dist_mat) || size(sub_dist_mat, 1) ~= nb_subregion
    sub_dist_mat = inf(nb_subregion, nb_subregion);
    for r = 1:nb_subregion
        sub_dist_mat(r, r) = 0; queue = r; head = 1;
        while head <= length(queue)
            u = queue(head); head = head + 1;
            nbs = find(sub_DGraph(u, :) > 0);
            for nb = nbs
                if sub_dist_mat(r, nb) == inf, sub_dist_mat(r, nb) = sub_dist_mat(r, u) + 1; queue(end+1) = nb; end
            end
        end
    end
end

dt = max(0, current_time - previous_time); distance_traveled_sub = dt * previous_speed(:);
N_veh = length(main_list);
if N_veh == 0, return; end

if ~isfield(main_list, 'dynamic_sub_seq')
    for i = 1:N_veh
        main_list(i).dynamic_sub_seq = main_list(i).subregions_thru_expected;
    end
end

curr_subs = [main_list.current_location_sub]; 
idx_currs = [main_list.curr_leg_idx];

distance_traveled_all = dt * previous_speed(curr_subs)';
active_remains = zeros(1, N_veh);
valid_mask = false(1, N_veh);

for i = 1:N_veh
    if idx_currs(i) <= length(main_list(i).dynamic_sub_seq)
        active_remains(i) = main_list(i).trip_L_remain(idx_currs(i));
        valid_mask(i) = true;
    end
end

active_remains(valid_mask) = max(0, active_remains(valid_mask) - distance_traveled_all(valid_mask));
crossed_mask = valid_mask & (active_remains <= 1e-4);
non_crossed_mask = valid_mask & ~crossed_mask;

non_crossed_indices = find(non_crossed_mask);
for k = 1:length(non_crossed_indices)
    idx = non_crossed_indices(k);
    main_list(idx).trip_L_remain(idx_currs(idx)) = active_remains(idx);
end

crossed_indices = find(crossed_mask);

% 提取全网实时子区基础通行预期耗时
sub_pass_time = zeros(1, nb_subregion);
for r = 1:nb_subregion
    sub_pass_time(r) = subregion_data(r).ave_trip_length / max(0.5, current_speed(r));
end

for k = 1:length(crossed_indices)
    i = crossed_indices(k);
    veh = main_list(i);
    idx_curr = idx_currs(i);
    
    veh.trip_L_remain(idx_curr) = 0;
    curr_sub = curr_subs(i);
    dest_sub = veh.Destination_sub;
    
    if curr_sub == dest_sub
        veh.current_location_sub = dest_sub; 
        veh.current_location_reg = veh.Destination_reg;
        veh.curr_leg_idx = idx_curr + 1; 
    else
        curr_reg = veh.current_location_reg; 
        dest_reg = veh.Destination_reg;
        is_cav = isfield(veh, 'is_cav') && veh.is_cav;
        
        % 提取走过的黑名单足迹
        history_nodes = veh.actual_path_recorded;
        
        % 目标大区识别
        target_reg = dest_reg;
        if veh.Origin_reg ~= dest_reg && curr_reg ~= dest_reg
            chosen_m_path = select_macro_p(curr_reg, dest_reg, is_cav, theta_CAV_macro, prob_path_ij, shortest_paths_ij);
            if length(chosen_m_path) >= 2
                target_reg = chosen_m_path(2); 
            end
        end
        
        % =================================================================
        % 🚀 核心革新：从 curr_sub 启动带阻抗引导与黑名单屏蔽的单源直连寻路
        % =================================================================
        cost_matrix = inf(nb_subregion, nb_subregion);
        for r = 1:nb_subregion
            % 屏蔽已走过的节点作为中转枢纽，绝对禁止走回头路
            if ismember(r, history_nodes) && r ~= curr_sub
                continue; 
            end
            
            valid_edges = find(sub_DGraph(r, :) > 0);
            for e = valid_edges
                % 目标终点放行，其余历史踩踏节点严格阻断
                if ~ismember(e, history_nodes) || e == dest_sub
                    base_cost = sub_pass_time(e);
                    
                    % 🚀 柔性诱导折扣机制：若当前处于路口决策起点，且邻居直接通往目标大区，给予通行权值极力倾斜
                    if r == curr_sub && sub2reg(e) == target_reg
                        cost_matrix(r, e) = base_cost * 0.01; % 99% 的成本折扣吸引车流
                    else
                        cost_matrix(r, e) = base_cost;
                    end
                end
            end
        end
        
        % 执行严密直连 Dijkstra 寻路推演
        dist = inf(1, nb_subregion); dist(curr_sub) = 0;
        prev = zeros(1, nb_subregion); visited = false(1, nb_subregion);
        
        for step = 1:nb_subregion
            [min_d, u] = min(dist + visited * inf);
            if isinf(min_d) || u == dest_sub, break; end
            visited(u) = true;
            for v = find(cost_matrix(u, :) < inf)
                alt = dist(u) + cost_matrix(u, v);
                if alt < dist(v)
                    dist(v) = alt; prev(v) = u;
                end
            end
        end
        
        % 路径尾链提取：天生保证从 curr_sub 的真实物理邻居严格延展至 dest_sub
        if prev(dest_sub) ~= 0
            full_tail = [];
            curr_n = dest_sub;
            while curr_n ~= curr_sub
                full_tail = [curr_n, full_tail];
                curr_n = prev(curr_n);
            end
            rem_chain = full_tail;
        else
            % 🚀 绝对安全兜底防线：无视所有黑名单约束，强制顺拓扑图生成物理连通分段
            rem_chain = get_safe_bfs_chain(curr_sub, dest_sub, sub_DGraph);
        end
        
        next_sub = rem_chain(1); % 提取实际迈出的下游第一个子区域
        reg_after = sub2reg(next_sub);
        
        % 1. 游标安全拼接：当前节点之前的所有分段完整保留，后续完整置换为最新连通链
        veh.dynamic_sub_seq = [veh.dynamic_sub_seq(1:idx_curr), rem_chain];
        
        % 2. 剩余长度严格对齐分配
        new_rem_L = zeros(1, length(rem_chain));
        for m = 1:length(rem_chain)
            s_id = rem_chain(m);
            if s_id == dest_sub
                new_rem_L(m) = subregion_data(s_id).ave_trip_length * rand;
            else
                new_rem_L(m) = subregion_data(s_id).ave_trip_length;
            end
        end
        veh.trip_L_remain = [veh.trip_L_remain(1:idx_curr), new_rem_L];
        
        % 状态与足迹精准闭环同步
        veh.current_location_sub = next_sub; 
        veh.current_location_reg = reg_after;
        
        veh.actual_path_recorded(end+1) = next_sub; 
        veh.sub_entry_times(end+1) = current_time;
        
        if length(veh.sub_entry_times) >= 2, veh.sub_durations(end+1) = current_time - veh.sub_entry_times(end-1); else, veh.sub_durations(end+1) = dt; end
        if curr_reg ~= reg_after, veh.actual_region_recorded(end+1) = reg_after; veh.reg_boundary_times(end+1) = current_time; end
        
        veh.curr_leg_idx = idx_curr + 1;
    end
    main_list(i) = veh;
end

% 动态时钟积分对齐更新
for i = 1:N_veh
    veh = main_list(i);
    idx_active = veh.curr_leg_idx;
    num_seq = length(veh.dynamic_sub_seq);
    
    if idx_active > num_seq
        veh.t_arr_subreg(:) = current_time; 
        veh.t_arr = current_time;
        veh.t_next_arr = inf;
    else
        t_accum = current_time;
        veh.t_arr_subreg = zeros(1, num_seq);
        veh.t_arr_subreg(1:idx_active-1) = current_time;
        
        for k = idx_active:num_seq
            v_val = max(0.5, current_speed(veh.dynamic_sub_seq(k)));
            t_accum = t_accum + veh.trip_L_remain(k) / v_val;
            veh.t_arr_subreg(k) = t_accum;
        end
        veh.t_arr = veh.t_arr_subreg(end);
        veh.t_next_arr = veh.t_arr_subreg(idx_active);
    end
    main_list(i) = veh;
end
end

% =========================================================================
% 🚀 底部原生局部辅助函数：保证极端环境下的纯净物理跳数首尾连通链生成
% =========================================================================
function chain = get_safe_bfs_chain(start_node, end_node, adj_mat)
    num_nodes = size(adj_mat, 1);
    parent = zeros(1, num_nodes);
    visited = false(1, num_nodes);
    queue = start_node;
    visited(start_node) = true;
    head = 1; found = false;
    
    while head <= length(queue)
        u = queue(head); head = head + 1;
        if u == end_node
            found = true; break;
        end
        for nb = find(adj_mat(u, :) > 0)
            if ~visited(nb)
                visited(nb) = true;
                parent(nb) = u;
                queue(end+1) = nb;
            end
        end
    end
    
    chain = [];
    if found
        curr = end_node;
        while curr ~= start_node
            chain = [curr, chain];
            curr = parent(curr);
        end
    else
        chain = end_node; % 极端极速备份保障
    end
end