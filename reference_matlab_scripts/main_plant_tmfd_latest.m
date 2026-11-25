%% main.m - 宏微观跨尺度分层控制架构主控程序 
clc, clear;
clear global; 

global flag_change_travelers flag_No_Control flag_MPC sub2reg nb_region nb_subregion Q_S t_plant tfinq

flag_change_travelers = 1; % 0=极速读取既有缓存，1=重新打乱生成需求流
flag_No_Control       = 1;
flag_MPC              = 0;

% 载入网络时变需求与空间状态矩阵
load('demands_value_19subreg.mat'); % 此时 Q_S 安全挂载为全局变量: 361 x 12
load('xInit_ss_19.mat');            % xInit_ss_19: 19 x 19
xInit_ss_19 = 0.6 * xInit_ss_19;
% Q_S = 0.8 * Q_S;

inputs_values; % 载入大区连接图与微观运动拟合参数

guidance_Init = (guidance_lb+guidance_ub)/K_shortestpaths * ones(control_number*Hc, 1);
t_plant = 0; 
current_time = t_init; 
accumulation_save = []; 
time_plot = [];

% 针对 19 子区架构计算初始拥堵速率
n_sub_initial = sum(xInit_ss_19, 2); 
V_n_sub = zeros(nb_subregion, 1);
for i = 1:nb_subregion
    n_i = n_sub_initial(i);
    V_n_sub(i,1) = subregion_data(i,1).ave_trip_length * ...
        (subregion_data(i,1).aMFD * n_i^2 + subregion_data(i,1).bMFD * n_i + subregion_data(i,1).cMFD) * ...
        (subregion_data(i,1).dMFD + subregion_data(i,1).eMFD * CAV_penetration + subregion_data(i,1).fMFD * CAV_penetration^2);
end

traveler_info_matrix; % 生成或加载多尺度精细化车辆序列库
traveler_info_main = traveler_info_initacc_Ri_all;
previous_sub_speed = V_n_sub; 
save_veh_data_cell = {};

% 准备通用绘图留档容器，支持无损承接输出
theta_save_HDV_cell = {}; 
theta_save_CAV_cell = {}; 

tic;
count = 0; 
while ~isempty(traveler_info_main) || ~isempty(traveler_info_demand_Rii_all) || ~isempty(traveler_info_demand_Rij_all)
    
    eps_time = 1e-6; 
    soonest_dep_Rii = inf; 
    soonest_dep_Rij = inf; 
    soonest_event_main = inf;
    
    if ~isempty(traveler_info_demand_Rii_all)
        t_deps = [traveler_info_demand_Rii_all.t_dep]; soonest_dep_Rii = min(t_deps(t_deps >= current_time - eps_time));
    end
    if ~isempty(traveler_info_demand_Rij_all)
        t_deps = [traveler_info_demand_Rij_all.t_dep]; soonest_dep_Rij = min(t_deps(t_deps >= current_time - eps_time));
    end
    
    if ~isempty(traveler_info_main)
        t_nexts = [traveler_info_main.t_next_arr];
        valid_t = t_nexts(t_nexts > current_time + eps_time);
        if ~isempty(valid_t)
            soonest_event_main = min(valid_t);
        end
    end
    
    soonest_time = min([soonest_dep_Rii, soonest_dep_Rij, soonest_event_main]);
    if isinf(soonest_time), break; end
    previous_time = current_time; 
    current_time = soonest_time;
    
    if abs(current_time - soonest_dep_Rii) < 1e-5
        idx = find(abs([traveler_info_demand_Rii_all.t_dep] - current_time) < 1e-5);
        traveler_info_main = [traveler_info_main, traveler_info_demand_Rii_all(idx)];
        traveler_info_demand_Rii_all(idx) = [];
    end
    if abs(current_time - soonest_dep_Rij) < 1e-5
        idx = find(abs([traveler_info_demand_Rij_all.t_dep] - current_time) < 1e-5);
        traveler_info_main = [traveler_info_main, traveler_info_demand_Rij_all(idx)];
        traveler_info_demand_Rij_all(idx) = [];
    end
    
    % 提取空间分布累积计数
    if ~isempty(traveler_info_main)
        loc_s = [traveler_info_main.current_location_sub];
        current_sub_n = accumarray(loc_s', 1, [nb_subregion, 1]);
    else
        current_sub_n = zeros(nb_subregion, 1);
    end
    
    current_reg_n = zeros(nb_region, 1);
    for i = 1:nb_region, current_reg_n(i) = sum(current_sub_n(sub2reg == i)); end
    
    current_sub_speed = zeros(nb_subregion, 1);
    for i = 1:nb_subregion
        n_i = current_sub_n(i);
        current_sub_speed(i,1) = subregion_data(i,1).ave_trip_length * (subregion_data(i,1).aMFD * n_i^2 + subregion_data(i,1).bMFD * n_i + subregion_data(i,1).cMFD) * (subregion_data(i,1).dMFD + subregion_data(i,1).eMFD * CAV_penetration + subregion_data(i,1).fMFD * CAV_penetration^2);
        if current_sub_speed(i,1) < 0.5, current_sub_speed(i,1) = 0.5; end
    end
    
    % 实时起终点负荷初态矩阵作为隐式参数全自动被底部函数截获映射
    [traveler_info_main, guidance_opt, cur_t_HDV, cur_t_CAV] = update_main_list_latest(...
        traveler_info_main, previous_time, previous_sub_speed, current_time, current_sub_speed, region_data, guidance_Init, tHp, Hp, Hc, tfinq, Q_S, current_sub_n);
    
    % 记录时序路由结果供后续双模式绘图分析比对
    theta_save_HDV_cell{end+1, 1} = cur_t_HDV;
    theta_save_CAV_cell{end+1, 1} = cur_t_CAV;
    
    % 基于动态执行序列完赛核销判定
    for k = length(traveler_info_main):-1:1
        veh = traveler_info_main(k);
        if veh.curr_leg_idx > length(veh.dynamic_sub_seq) && veh.current_location_sub == veh.Destination_sub
            save_veh_data_cell{end+1, 1} = veh; 
            traveler_info_main(k) = [];
        end
    end
    
    previous_sub_speed = current_sub_speed; 
    accumulation_save = [accumulation_save, current_reg_n];
    time_plot = [time_plot, current_time]; 
    guidance_Init = guidance_opt;
    
    count = count + 1
end
toc;

% 智能保存适配当前激活控制流的终态结果矩阵
if flag_No_Control == 1
    save('Results_19Sub_7Reg_CAV60_NC_test1.mat');
else
    save('Results_19Sub_7Reg_CAV60_MPC_test1.mat');
end