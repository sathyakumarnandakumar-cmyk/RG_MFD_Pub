function [g_output, travel_time_HDV, travel_time_CAV_SO] = pmfd_region(n_input, region_data, subreg_struct)
% 宏观大区 MFD 推算函数：基于所属子区域(subregion) MFD 产出的严格累加求和
% 支持 标量宏观总积累量输入(自动均摊求和) 或 向量实际积累量输入(独立遍历求和)

global CAV_penetration 

% --- 1. 安全提取对应子区域的基础 MFD 参数 ---
% 优先读取显式传入的第三个子区域参数，其次尝试全局变量，最后安全兜底
if nargin >= 3 && ~isempty(subreg_struct) && isfield(subreg_struct, 'aMFD')
    s_data = subreg_struct(1);
elseif ~isempty(subregion_data) && isfield(subregion_data, 'aMFD')
    s_data = subregion_data(1);
else
    % 标准底层子区域多项式拟合系数兜底
    s_data.aMFD = 4.787e-10;
    s_data.bMFD = -1.528e-5;
    s_data.cMFD = 1.257e-1;
    s_data.dMFD = 3.636e-10;
    s_data.eMFD = -4.802e-6;
    s_data.fMFD = 8.346e-2;
end

A = s_data.aMFD;
B = s_data.bMFD;
C = s_data.cMFD;
D = s_data.dMFD;
E = s_data.eMFD;
F = s_data.fMFD;

% 计算渗透率调整因子
if isempty(CAV_penetration)
    pen_factor = 1;
else
    pen_factor = D + E * CAV_penetration + F * CAV_penetration^2;
end

% 提取宏观大区平均行程跨度 (用于计算大区平均旅行时间)
if isfield(region_data, 'ave_trip_length')
    L_reg = region_data.ave_trip_length;
else
    L_reg = 2300;
end

num_sub_per_reg = 7; % 每个宏观大区固定包含的子区域个数

% --- 2. 核心求和机制：测算大区总流量 g_output 与 宏观导数 dg/dN ---
if isscalar(n_input)
    % 场景 A：输入为大区宏观总车辆数 N_I，均摊至 7 个子区域求和
    N_total = n_input;
    n_sub   = N_total / num_sub_per_reg;
    
    % 计算单一子区域的瞬时产出
    g_sub = (A * n_sub^3 + B * n_sub^2 + C * n_sub) * pen_factor / 3600;
    g_sub = max(g_sub, 0);
    
    % 宏观总流量 = 7 个子区域流量严格求和
    g_output = num_sub_per_reg * g_sub;
    
    % 链式法则求导：由于 g_reg(N) = 7 * g_sub(N/7)，则 dg_reg/dN = g_sub'(N/7)
    dg_dN = (3 * A * n_sub^2 + 2 * B * n_sub + C) * pen_factor / 3600;
    
else
    % 场景 B：输入为各个独立子区域积累量构成的数组
    N_total = sum(n_input);
    g_output = 0;
    dg_dN_weighted = 0;
    
    for k = 1:length(n_input)
        n_sub = n_input(k);
        g_sub = (A * n_sub^3 + B * n_sub^2 + C * n_sub) * pen_factor / 3600;
        g_sub = max(g_sub, 0);
        
        % 逐一累加各子区域实际流量
        g_output = g_output + g_sub; 
        
        % 累加各局部斜率的加权贡献
        dg_sub = (3 * A * n_sub^2 + 2 * B * n_sub + C) * pen_factor / 3600;
        if N_total > 0
            dg_dN_weighted = dg_dN_weighted + dg_sub * (n_sub / N_total);
        end
    end
    dg_dN = dg_dN_weighted;
end

g_output = max(g_output, 1e-6); % 物理兜底保障，避免除以 0

% --- 3. 宏观 HDV 用户最优旅行时间推算 T = L * N / g ---
if N_total <= 0
    travel_time_HDV = 0;
else
    travel_time_HDV = (L_reg * N_total) / g_output;
end

% --- 4. 宏观 SO 系统最优边际旅行时间推算 ---
% 严格遵循数学推导 MC = d(T * N)/dN = L * N * (2/g - N * g' / g^2)
if N_total <= 0
    travel_time_CAV_SO = 0;
else
    term1 = 2 / g_output;
    term2 = (N_total * dg_dN) / (g_output^2);
    travel_time_CAV_SO = L_reg * N_total * (term1 - term2);
end

travel_time_CAV_SO = max(travel_time_CAV_SO, 0);

end