%% inputs_values.m - 19子区域与7大区全新拓扑精确对齐配置文件
global nb_region nb_subregion reg_DGraph sub2reg sub_DGraph border_capacity recieving_cap min_flow...
       Q_S parameter_logit K_shortestpaths MFD_L MFD_L_2border...
       period_guidance control_number t_init guidance_lb guidance_ub...
       CAV_penetration route_guidance_step_size route_guidance_indices subregion_data region_data...
       tHp Hp Hc tfinq Tc Tstep

% 1. 基础仿真时间与控制步长参数
Tstep = 60;
period_guidance = 1;
Tc = Tstep * period_guidance;
Hp = 6;
Hc = 2;
tHp = Tc * Hp;
t_init = 0;
tfinq = 3600;
tfin = tfinq + 6000;

% 2. 宏微观空间维度界定
nb_region = 7;
nb_subregion = 19; % 底层坍缩至 19 节点

% =========================================================================
% 🚀 3. 核心重构：上层 7 个大区域 (Macro-regions) 全新物理链接关系
% 共计 13 条宏观通道，完美表征"外环-内环-核心"的蜂窝拓扑
% =========================================================================
neighbours_reg = [
    1 2; 1 4; 1 6;        % R1 的邻居
    2 3; 2 5; 2 6;        % R2 的新增/保留邻居
    3 4; 3 5;             % R3 的邻居
    4 5; 4 6;             % R4 的内环通道
    5 6; 5 7;             % R5 与内环/红心通道
    6 7                   % R6 与红心通道
];
UGraph_reg = sparse(neighbours_reg(:,1), neighbours_reg(:,2), ones(size(neighbours_reg,1),1), nb_region, nb_region);
reg_DGraph = UGraph_reg + UGraph_reg'; % 自动生成 7x7 宏观对称连通图

% =========================================================================
% 🚀 4. 底层 19 个子区域向 7 个大区的精准物理归属映射
% R1=[1,2,3], R2=[4,5,6], R3=[7,8,9], R4=[10,11,12], R5=[13,14,15], R6=[16,17,18], R7=[19]
% =========================================================================
sub2reg = zeros(1, nb_subregion);
sub2reg([1, 2, 3])    = 1;
sub2reg([4, 5, 6])    = 2;
sub2reg([7, 8, 9])    = 3;
sub2reg([10, 11, 12]) = 4;
sub2reg([13, 14, 15]) = 5; % 完美吻合图片黄色包裹范围
sub2reg([16, 17, 18]) = 6; % 完美吻合图片粉色包裹范围
sub2reg(19)           = 7; % 核心红心仅含节点 19

% =========================================================================
% 🚀 5. 依据拓扑图精确建立 19 节点全域微观双向邻接边
% 涵盖大区内部衔接与全部跨大区物理边界
% =========================================================================
subregion_neighbours = [
    % ----- 大区内部物理连通通道 -----
    1 2; 2 3;             % R1 内部
    4 5; 5 6;             % R2 内部
    7 8; 8 9;             % R3 内部
    10 11; 11 12;         % R4 内部
    13 14; 14 15;         % R5 内部
    16 17; 17 18;         % R6 内部
                          % R7(19) 无内部边
    
    % ----- 跨大区微观物理接壤边界 -----
    3 4;                  % R1 - R2
    1 12;                 % R1 - R4
    1 18; 2 18; 2 17; 3 17; % R1 - R6接壤面
    6 7;                  % R2 - R3
    5 15; 6 15;           % R2 - R5接壤面
    4 16; 5 16; 4 17;     % R2 - R6接壤面
    9 10;                 % R3 - R4
    7 15; 8 15; 9 15; 9 14; % R3 - R5接壤面
    10 14; 11 14; 11 13; 12 13; % R4 - R5接壤面
    12 18;                % R4 - R6接壤面
    13 18; 15 16;         % R5 - R6接壤面
    13 19; 14 19; 15 19;  % R5 - R7接壤面 (通往红心)
    16 19; 17 19; 18 19   % R6 - R7接壤面 (通往红心)
];

sub_UGraph = sparse(subregion_neighbours(:,1), subregion_neighbours(:,2), 1, nb_subregion, nb_subregion);
sub_DGraph = sub_UGraph + sub_UGraph';
sub_DGraph(sub_DGraph > 1) = 1; % 二值化处理

% 6. 下放拟合属性至 19 节点 (单区平均跨度等比例适配)
subregion_data = struct;
for r = 1:nb_subregion
    subregion_data(r,1).ave_trip_length = 2300;
    subregion_data(r,1).aMFD = 4.787e-10; 
    subregion_data(r,1).bMFD = -1.528e-5;
    subregion_data(r,1).cMFD = 1.257e-1;
    subregion_data(r,1).dMFD = 3.636e-10;
    subregion_data(r,1).eMFD = -4.802e-6;
    subregion_data(r,1).fMFD = 8.346e-2;
    subregion_data(r,1).njamMFD = 15660;
    
    a = subregion_data(r,1).aMFD; b = subregion_data(r,1).bMFD; c = subregion_data(r,1).cMFD;
    subregion_data(r,1).ncriticalMFD = (-b - sqrt(b^2 - 3*a*c)) / (3*a);
    subregion_data(r,1).travel_length_2border = 1300;
end

% 7. 向上整合大区宏观容量上限
region_data = struct;
for i = 1:nb_region
    region_data(i,1).njamMFD = sum([subregion_data(sub2reg == i).njamMFD]);
    region_data(i,1).ncriticalMFD = sum([subregion_data(sub2reg == i).ncriticalMFD]);
    region_data(i,1).ave_trip_length = sum([subregion_data(sub2reg == i).ave_trip_length]);
end

% 8. 控制器引导变量分配字典初始化
min_flow = 0.015; K_shortestpaths = 2; parameter_logit = 0.1;
guidance_lb = 0; guidance_ub = 1; CAV_penetration = 0.6;
control_number = nb_region * (nb_region - 1) * K_shortestpaths;
route_guidance_indices = cell(nb_region, nb_region);
curr_idx = 0;
for i = 1:nb_region
    for j = 1:nb_region
        if i ~= j
            route_guidance_indices{i,j} = curr_idx + (1:K_shortestpaths);
            curr_idx = curr_idx + K_shortestpaths;
        end
    end
end