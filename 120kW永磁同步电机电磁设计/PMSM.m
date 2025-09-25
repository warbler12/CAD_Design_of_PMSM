%% 120kW PMSM电磁设计主程序
clc; clear; close all;

%% ==================== 基本参数输入 ====================
input.Pn = 120e3;        % 额定功率 [W]
input.Nn = 6500;         % 额定转速 [rpm]
input.Nmax = 20000;      % 最大转速 [rpm]
input.Vph = 200;         % 额定相电压 [Vrms]
input.Iph = 220;         % 额定相电流 [A]
input.J = 5e6;           % 电流密度 [A/m²]
input.eta = 0.97;        % 额定效率
input.Connection = 'Y';  % 绕组接法

% 结构参数
input.D1 = 0.22;         % 定子外径 [m]
input.D1i = 0.12;        % 定子内径 [m]
input.L = 0.2;           % 铁芯长度 [m]
input.g = 1e-3;          % 气隙长度 [m]
input.Qs = 48;           % 定子槽数
input.p = 4;             % 极对数
input.SlotType = 'Round';% 槽型
input.Hs0 = 1.02e-3;
input.Hs2 = 29.58e-3;
input.Bs0 = 1.93e-3;
input.Bs1 = 3.15e-3;
input.Bs2 = 5.8e-3;

% 材料参数
input.Br = 1.318;        % 永磁体剩磁 [T]
input.Hc = 1006264;      % 永磁体矫顽力 [A/m]
input.SteelGrade = '20SW1200H'; % 硅钢片牌号

%% ==================== 电磁计算主函数 ====================
[Design, Performance] = PMSM_Design_Function(input);

%% ==================== 结果显示 ====================
disp('===== 主要几何尺寸 =====');
fprintf('定子内径 D1i          = %.2f mm\n', input.D1i * 1e3);
fprintf('定子外径 D1           = %.2f mm\n', input.D1 * 1e3);
fprintf('转子外径 D2           = %.2f mm\n', Design.D2 * 1e3);
fprintf('转子内径 D2i          = %.2f mm\n', Design.D2i * 1e3);
fprintf('有效铁芯长度 La       = %.2f mm\n', Design.La * 1e3);
fprintf('极距 tau              = %.2f mm\n', Design.tau * 1e3);
fprintf('齿距 t                = %.2f mm\n', Design.t * 1e3);
fprintf('齿宽 bt               = %.2f mm\n', Design.bt * 1e3);
fprintf('定子轭高              = %.2f mm\n', Design.h_j_stator * 1e3);
fprintf('转子轭高              = %.2f mm\n', Design.h_j_rotor * 1e3);

disp('===== 槽型结构参数 =====');
fprintf('槽型类型              = %s\n', input.SlotType);
fprintf('槽底高度 Hs0          = %.2f mm\n', input.Hs0 * 1e3);
fprintf('齿高     Hs2          = %.2f mm\n', input.Hs2 * 1e3);
fprintf('槽口宽度 Bs0          = %.2f mm\n', input.Bs0 * 1e3);
fprintf('中部槽宽 Bs1          = %.2f mm\n', input.Bs1 * 1e3);
fprintf('槽底宽度 Bs2          = %.2f mm\n', input.Bs2 * 1e3);
fprintf('槽面积估算 As         = %.2f mm²\n', ...
        Calculate_Slot_Area(input.Hs0, input.Hs2, input.Bs0, input.Bs1, input.Bs2) * 1e6);

disp('===== 绕组参数 =====');
fprintf('并联支路数 a          = 2\n');
fprintf('每槽导体数 Ns         = %d\n', Design.Ns);
fprintf('每相有效串联导体 Nph_e = %d\n', Design.Nph_e);
fprintf('绕组节距              = %d/%d\n', Design.y, round(input.Qs/(2 * input.p)));
fprintf('绕组系数 kw           = %.4f\n', Design.kw);
fprintf('导线直径              = %.4f\n', Design.D_cond);
fprintf('绕组配置              = 单层同心式绕组，10 根并联导线\n');
fprintf('导体截面积 A_wire     = %.2f mm²\n', Design.WireArea * 1e6);
fprintf('槽满率                = %.1f %%\n', Design.SlotFill * 100);


disp('===== 磁通与磁压降 =====');
fprintf('主磁通 Phi_m          = %.3f mWb\n', Design.Phi_m * 1e3);
fprintf('气隙磁密 B_delta      = %.3f T\n', Design.B_delta);
fprintf('齿部磁密 B_t          = %.3f T\n', Design.B_t);
fprintf('定子轭部磁密 B_j_stat  = %.3f T\n', Design.B_j_stator);
fprintf('转子轭部磁密 B_j_rot   = %.3f T\n', Design.B_j_rotor);
fprintf('\n磁压降 F_m           = %.1f A\n', Design.F_m);
fprintf('  - 气隙压降           = %.1f A (%.1f%%)\n', ...
    Design.F_delta, 100*Design.F_delta/Design.F_m);
fprintf('  - 齿部压降           = %.1f A (%.1f%%)\n', ...
    Design.F_t, 100*Design.F_t/Design.F_m);
fprintf('  - 定子轭部压降       = %.1f A (%.1f%%)\n', ...
    Design.F_j_stator, 100*Design.F_j_stator/Design.F_m);
fprintf('  - 转子轭部压降       = %.1f A (%.1f%%)\n', ...
    Design.F_j_rotor, 100*Design.F_j_rotor/Design.F_m);
fprintf('总磁压降校验          = %.1f A (误差 %.1f%%)\n', ...
    Design.F_delta + Design.F_t + Design.F_j_rotor + Design.F_j_stator, ...
    100*abs(Design.F_m - (Design.F_delta + Design.F_t + ...
    Design.F_j_rotor + Design.F_j_stator)) / Design.F_m);

disp('===== 电气参数与等效模型 =====');
fprintf('相电阻 Rph            = %.4f Ω (75℃)\n', Design.Rph);
fprintf('主电抗 Xm             = %.4f Ω\n', Design.Xm);
fprintf('漏抗 Xσ               = %.4f Ω\n', Design.X_sigma);
fprintf('  - 槽漏抗 Xs         = %.4f Ω\n', Design.Xs);
fprintf('  - 谐波漏抗 Xδ       = %.4f Ω\n', Design.X_delta);
fprintf('  - 端部漏抗 XE       = %.4f Ω\n', Design.XE);
fprintf('  - 齿顶漏抗 Xt       = %.4f Ω\n', Design.X_t);
fprintf('标幺主电抗 Xm*        = %.4f\n', Design.Xm_nom);
fprintf('标幺漏抗 Xσ*          = %.4f\n', Design.X_sigma_nom);
fprintf('标幺电阻 R*           = %.4f\n', Design.Rph_nom);

disp('===== 损耗与效率 =====');
fprintf('铜耗 Pcu               = %.2f kW\n', Performance.Pcu / 1e3);
fprintf('铁耗 Pfe               = %.2f kW\n', Performance.Pfe / 1e3);
fprintf('机械损耗 Pmec          = %.2f kW\n', Performance.Pmec / 1e3);
fprintf('附加损耗 Padd          = %.2f kW\n', Performance.Padd / 1e3);
fprintf('输入功率 Pin           = %.2f kW\n', (input.Pn + Performance.Pcu + ...
        Performance.Pfe + Performance.Pmec + Performance.Padd) / 1e3);
fprintf('输出功率 Pout          = %.2f kW\n', input.Pn / 1e3);
fprintf('效率η                 = %.2f %%\n', Performance.Efficiency * 100);
fprintf('额定转矩Tn             = %.2f Nm\n', Performance.Tn);

disp('===== 反电势与占比分析 =====');
fprintf('电压法占比 k_E_emf       = %.4f\n', Design.k_E_emf);
fprintf('磁链法占比 k_E_phi       = %.4f\n', Design.k_E_calc);
fprintf('理论功率因数估计         = %.4f\n', Design.PF);
fprintf('额定工作点功率因数        = %.4f\n', Performance.PF);

disp('===== 起动性能估算 =====');
fprintf('起动电流 Ist             = %.2f A\n', Performance.Ist);
fprintf('起动电流 Ist / In        = %.2f\n', Performance.Ist_ratio);
fprintf('起动转矩 Tst             = %.2f Nm\n', Performance.Tst);
fprintf('起动转矩 Tst / Tn        = %.2f\n', Performance.Tst_ratio);

%% ==================== 函数定义 ====================
function [Design, Performance] = PMSM_Design_Function(input)
    %% ======  加载 B-H 曲线数据并构建插值函数 ======
    data = readtable('20SW1200H_B-H Curve.xlsx', 'VariableNamingRule', 'preserve');
    B_data = data.("B (T)"); % 单位: T
    H_data = data.("H (A/m)"); % 单位: A/m

    % 构建插值函数：根据B查找H
    get_H_from_B = @(B_query) interp1(B_data, H_data, B_query, 'linear', 'extrap');
    
    %% ======  扩大 B-H 曲线范围 ======
    % 检查 B-H 曲线的范围
    B_max = max(B_data);
    H_max = max(H_data);

    % 线性外插
    H_new = linspace(H_max, H_max * 1.5, 100)'; % 转换为列向量
    slope_BH = (B_data(end) - B_data(end-1)) / (H_data(end) - H_data(end-1)); % 末端斜率
    B_new = B_data(end) + slope_BH * (H_new - H_data(end));

    % 更新 B-H 曲线
    H_data = [H_data; H_new]; % 垂直拼接
    B_data = [B_data; B_new]; % 垂直拼接

    %% 参数设置
    mu_0 = 4*pi*1e-7;               % 真空磁导率
    mu_m = 1.05;                    % 永磁体相对磁导率
    omega_n = input.Nn * 2*pi/60;   % 额定机械角速度
    omega_n_e = omega_n * input.p;   % 额定电角速度
    f_elec = input.p * input.Nn / 60; % 计算电频率
    I_kw = input.Pn /(3 * input.Vph); % 功电流

    % 电磁负荷选择
    A = 45e3;                       % 线负荷估计 [A/m^2]
    B_delta = 1.4;                  % 气隙磁密估计 [T]
    k_Fe = 0.95;                    % 铁芯有效长度系数
    k_Nm = 1.11;                    % 电枢电动势波形系数
    k_dp = 0.96;                    % 叠片系数
    alpha_p = 0.64;                 % 极弧系数

    % 功率因数
    P_active = input.Pn;                    % 有功功率
    S_apparent = input.Vph * input.Iph * 3; % 三相视在功率
    Performance.PF = P_active / S_apparent; % 功率因数
    
    %% 转子设计
    Design.D2 = input.D1i - 2 * input.g;              % 转子外径
    Design.D2i = input.D1i / input.D1 * Design.D2;    % 转子内径 

    Design.mag_thick = 6e-3;            % 永磁体厚度
    Design.w_m = 45e-3;                 % 永磁体宽度
    Design.bridge_thick = 2e-3;         % 桥厚
    Design.rib = 3e-3;                  % 肋宽
    Design.h_j_rotor = (Design.D2 - Design.D2i) / 2 - Design.mag_thick - Design.bridge_thick; % 轭高
    L_j_rotor = pi * (Design.D2 + Design.D2i) / 2 / (4 * input.p);              % 每极轭部磁路计算长度

    %% 绕组设计
    % 并联支路数
    a = 2; 

    % 绕组系数计算
    [Design.kw, Design.y] = Winding_Factor(input.Qs, input.p);

    % 极距计算
    Design.tau = pi * (input.D1i - input.g) / (2 * input.p);  %极距
    beta = Design.y / round(input.Qs/(2 * input.p));    % 绕组节距比 (5/6节距)
    q = input.Qs/(3*2*input.p);                         % 每极每相槽数

    % 每相串联导体数
    Phi_m = alpha_p * Design.tau * input.L * B_delta;                        % 磁通量
    Nph = ceil(input.eta * Performance.PF * pi * input.D1i * A /(3 * I_kw)); % 每相串联导体数
    N1 = Nph / 2;                                                            % 每相串联匝数

    % 每槽导体数 (单层绕组)
    Design.Ns = ceil(Nph * 3 * a/ input.Qs);                  % 每槽导体数
    Design.Nph_e = ceil(Design.Ns * input.Qs /(3 * a) * k_dp);% 每相有效串联导体数
    Ndp = Design.Nph_e / (2 * input.p);                       % 每极每相有效匝数

    % 实际反电势验证
    E0 = 4.44 * Design.kw * Design.Ns * (input.Qs/2) * ...
         (input.p*omega_n/(2*pi)) * Phi_m / (2*input.p); 
    
    %% 槽型设计
    % 导线选择 
    I_stator = I_kw / (input.eta * Performance.PF); % 定子电流估计
    Design.WireArea = I_stator / (a * input.J); % 导线截面积 22.68mm^2
    
    % 选择标准线规, 10根并绕，2.2698mm^2
    Design.D_cond = 1.7e-3;

    Design.t = pi * input.D1i / input.Qs;           % 齿距计算
    Design.bt = pi * (input.D1i + 2 * input.Hs0 + 2* input.Hs2) / input.Qs - input.Bs2; % 齿宽计算
    B_t = B_delta * Design.t / (k_Fe * Design.bt);  % 齿部磁密

    % 轭部磁密
    Phi_j = Phi_m / 2;                              % 轭部最大磁通估计
    Design.h_j_stator = (input.D1 - input.D1i) / 2 - input.Hs0 - input.Hs2 - input.Bs2 / 2 + input.Bs2 / 2 / 3; % 定子轭高
    l_j = input.L * k_Fe;                           % 定转子轴向长度估算
    B_j_stator = Phi_j / (Design.h_j_stator * l_j); % 定子轭部磁密估算
    B_j_rotor = Phi_j / (Design.h_j_rotor * l_j);   % 转子轭部磁密估算

    % 槽满率计算
    delta_i = 0.3e-3;
    A_s = Calculate_Slot_Area(input.Hs0, input.Hs2, input.Bs0, input.Bs1, input.Bs2); % 槽面积函数
    A_i = delta_i * (2 * input.Hs2 + pi * input.Bs2 / 2);                             % 估算绝缘所占面积
    Design.SlotFill = (Design.Ns * (1.05 *Design.WireArea)) / (A_s - A_i);            % 槽满率

    %% 性能计算
    % 定子每相电阻计算
    l_turn = 2*(input.L + 0.15*pi*(input.D1i - 2*input.g)/input.p); % 单匝长度
    rho_Cu = 1.75e-8*(1 + 0.004*(75-15));                       % 75℃铜电阻率
    Design.Rph = rho_Cu * l_turn * Nph / (a * Design.WireArea); % 定子每相电阻
    
    % 损耗计算
    Performance.Pcu = 3 * input.Iph^2 * Design.Rph; % 铜耗
    Performance.Pfe = Calculate_CoreLoss(input, Design, omega_n, l_j, B_t, B_j_stator, B_j_rotor); % 铁耗函数估算
    Performance.Pmec = 0.01 * input.Pn;             % 机械损耗估算
    Performance.Padd = 0.005 * input.Pn;            % 附加损耗估算

    % 效率验证
    Pout = input.Pn;
    Pin = Pout + Performance.Pcu + Performance.Pfe + Performance.Pmec + Performance.Padd;
    Performance.Efficiency = Pout / Pin;
    
    % 转矩特性
    Performance.Tn = Pout / omega_n;
    
    %% 主要尺寸
    K_E = 0.0108 * log(input.Pn * 1e-3) - 0.013 * input.p + 0.931; % 铁心利用效率
    P = K_E * input.Pn / Performance.PF / Performance.Efficiency;  % 修正后的电机总功率 P
    V = (6.1 * P) / (alpha_p * k_Nm * k_dp * B_delta * A * input.Nn);% V = D*l^2
    Design.La = V / input.D1i ^ 2 + 2 * input.g;                   % 铁心有效长度

    %% 磁路计算
    % 转子永磁体磁动势
    l_m = input.L + 2 * input.g;                        % 永磁体轴向长度
    A_m = l_m * Design.w_m;                             % 永磁体磁通通过面积
    Design.R_m = Design.mag_thick /(mu_0 * mu_m * A_m); % 永磁体磁阻
    A_g = alpha_p * Design.tau * l_m;                   % 气隙磁通通过面积
    Design.R_g = input.g / (mu_0 * A_g);                % 气隙磁阻
    Design.H = input.Br / (mu_0 * mu_m);                % 永磁体磁场强度
    Design.F_m = Design.H * Design.mag_thick;             % 永磁体磁动势
    Design.Phi_m = Design.F_m /(Design.R_g + Design.R_m); % 主磁通

    % 转子极联轭磁压降
    Design.Phi_j = Design.Phi_m / 2;                            % 轭部最大磁通
    Design.B_j_rotor = Design.Phi_j / (Design.h_j_rotor * l_j); % 轭部磁密
    Design.H_j_rotor = get_H_from_B(Design.B_j_rotor);          % 查表得轭部磁场强度[A/m]
    Design.F_j_rotor = Design.H_j_rotor * L_j_rotor;            % 轭部磁压降

    % 气隙磁压降
    Design.B_delta = Design.Phi_m / (alpha_p * Design.tau * l_m);             % 气隙磁密
    k_delta = Design.t * (4.4 * input.g + 0.75 * input.Bs0) / ...
              (Design.t * (4.4 * input.g + 0.75 * input.Bs0) - input.Bs0^2);  % 气隙系数
    Design.F_delta = Design.B_delta * input.g * k_delta / mu_0;               % 气隙磁压降
    
    % 定子齿部磁压降
    Design.B_t = Design.B_delta * Design.t / (k_Fe * Design.bt); % 齿部磁密
    Design.H_t = get_H_from_B(Design.B_t);                       % 查表得齿部磁场强度[A/m]
    if Design.B_t>1.8 
        Design.B_t_prime = Design.B_delta * Design.t / (k_Fe * Design.bt); % 视在磁密
        k_s = (Design.t - Design.bt) / Design.bt;                   % 槽系数
        slope = mu_0 * k_s;                                         % 斜线斜率 (B = B_t' - slope * H)
        H_range = linspace(0, max(H_data) , 1000);                  % 扩展H范围
        B_slope = Design.B_t_prime - slope * H_range;               % 定义斜线方程：B = B_t_prime - slope * H
        
        figure;
        plot(H_range, B_slope, 'b-', 'LineWidth', 2); % 斜线
        hold on;
        plot(H_data, B_data, 'r-', 'LineWidth', 2); % B-H 曲线
        legend('斜线 B = B_t'' - slope * H', 'B-H 曲线');
        grid on;
        xlabel('H (A/m)');
        ylabel('B (T)');
        title('斜线与 B-H 曲线的交点');
    
        Design.H_t = 105909;
        Design.B_t = 2.10314;
    end
    L_t = input.Hs0 + input.Hs2 + input.Bs2 / 6;                 % 齿的磁路计算长度
    Design.F_t = Design.H_t * L_t;                               % 齿部磁压降

    % 定子齿联轭磁压降
    C_j = 0.25;                                             % 轭部磁压降校正系数
    L_j_stator = pi * ((input.D1 + input.D1i) / 2 + input.Hs0 + ...
                 input.Hs2 + input.Bs2 / 2)/ (4 * input.p); % 每极轭部磁路计算长度
    Design.B_j_stator = Design.Phi_j / (Design.h_j_stator * l_j); % 轭部磁密
    Design.H_j_stator = get_H_from_B(Design.B_j_stator);    % 查表得轭部磁场强度[A/m]
    Design.F_j_stator = C_j * Design.H_j_stator * L_j_stator; %轭部磁压降

    %% 电气参数
    %% ==================== 主电抗计算 ====================
    delta_eff = input.g * k_delta;                      % 有效气隙[m]
    Design.Xm = 4 * mu_0 * f_elec * (3/pi) * ((Nph*k_dp)^2 / input.p) *...
                Design.La * Design.tau / delta_eff;     % 主电抗
    
    %% ==================== 漏电抗计算 ====================
    % 槽比漏磁导 (单层短距绕组)
    K_U = (3 - beta) / 4;
    lambda_s = K_U*((input.Hs0/input.Bs1) + (input.Hs2/(3*input.Bs1))); % 槽比漏磁导

    % 槽漏抗
    Design.Xs = 4*pi*mu_0*f_elec*(Nph^2/(input.p*q))*Design.La*lambda_s;

    % 谐波漏抗 
    sigma_s = 0.02;                                             % 谐波磁阻系数 查表 q=2,beta=5/6
    lambda_delta = 3*q*Design.tau*sigma_s/(pi*pi*delta_eff);    % 谐波比漏磁导
    Design.X_delta = 4*pi*mu_0*f_elec*(Nph^2/(input.p*q))*Design.La*lambda_delta; 

    % 齿顶漏抗
    sigma = 2/pi*(atan(input.Bs0/(2*input.g))-input.g/input.Bs0*log(1+input.Bs0^2/(2*input.g)^2));
    lambda_td = 0.2284+0.0796*input.g/input.Bs0-0.25*input.Bs0/input.g*(1-sigma);     
    lambda_tq = 0.2164+0.3184*(Design.bt/input.Bs0)^0.5;
    lambda_t = alpha_p*lambda_td+(1-alpha_p)*lambda_tq;                             % 齿顶比漏磁导
    Design.X_t = 4*pi*mu_0*f_elec*(Nph^2/(input.p*q))*Design.La*K_U*lambda_t;       % 齿顶漏抗

    % 端部漏抗 
    l_end = max(0.4*pi*input.D1i/(2*input.p), 0.05);          % 端部长度估算
    lambda_E = 0.67*q*(l_end - 0.64*Design.tau)/Design.La;    % 端部漏磁导
    Design.XE = 4*pi*mu_0*f_elec*(Nph^2/(input.p*q))*Design.La*lambda_E; % 端部漏抗
    
    % 总漏抗
    Design.X_sigma = Design.Xs + Design.X_delta + Design.X_t + Design.XE;
    
    %% ==================== 标幺值换算 ====================
    Design.Z_base = input.Vph / input.Iph;           % 基阻抗
    Design.Xm_nom = Design.Xm / Design.Z_base;       % 主电抗标幺
    Design.X_sigma_nom = Design.X_sigma / Design.Z_base; % 漏抗标幺
    Design.Rph_nom = Design.Rph / Design.Z_base;     % 电阻标幺

    %% ==================== 电压占比法评估 Ke ====================
    I_P_nom = 1 / Performance.Efficiency;            % 有功电流标幺
    I_Q_nom = Design.X_sigma_nom * I_P_nom;          % 无功电流估算
    Design.PF = I_P_nom / sqrt(I_P_nom^2 + I_Q_nom^2); % 功率因数估计
    Design.k_E_emf = 1 - (I_Q_nom * Design.X_sigma_nom + I_P_nom * Design.Rph_nom);
   
    %% ==================== 磁链占比评估 Ke ====================
    Design.k_E_calc = E0 / input.Vph; % 与峰值电压比较
    
    %% 计算起动性能
    % V/f控制
    f_start = 5;                        % 起动频率 (Hz)
    omega_sync = 2*pi*f_start;

    % 起动电流
    Ust = input.Vph/f_elec * f_start ;               % 起动电压
    Ust = max(Ust, 50);                              % 最低电压限制50V
    L_s = Design.Xm / omega_n;                       % 等效电感
    Zst = sqrt(Design.Rph^2 + (omega_sync * L_s)^2); % 起动阻抗
    Performance.Ist = Ust / Zst;                        % 起动电流
    Performance.Ist_ratio = Performance.Ist / input.Iph;% 起动电流倍数

    % 起动转矩
    Performance.Tst = (3*input.p/omega_sync) * (Ust^2 * Design.Rph) /...
    (Design.Rph^2 + (2*pi*f_start*L_s)^2); 
    Performance.Tst_ratio = Performance.Tst / Performance.Tn; % 起动转矩
    
end

%% 绕组系数计算子函数
function [kw, y] = Winding_Factor(Qs, p)
    % 计算分布系数
    q = Qs / (3 * 2 * p);       % 每极每相槽数
    alpha_elec = 2*pi*p / Qs;   % 槽距角(电角度)
    kd = sin(q*alpha_elec/2) / (q * sin(alpha_elec/2)); % 分布系数
    
    % 计算短距系数
    y = floor(5*Qs/(6*2*p));            % 节距
    beta = (Qs/(2*p) - y) * alpha_elec; % 短距角
    kp = cos(beta/2);                   % 短距系数
    
    kw = kd * kp;                       % 绕组系数
end

%% 槽型计算子函数 
function SlotArea = Calculate_Slot_Area(Hs0, Hs2, Bs0, Bs1, Bs2)
    SlotArea = (Bs1 + Bs2) * Hs2 / 2 + pi * Bs2^2 / 8;
end

%% 铁耗计算子函数
function Pfe = Calculate_CoreLoss(input, Design, omega_n, l_j, B_tooth, B_yoke_stator, B_yoke_rotor)
    % 硅钢片参数 (20SW1200H典型值)
    kh = 114.21;    % 磁滞损耗系数
    kc = 0.23;      % 涡流损耗系数
    ke = 0.42;      % 附加损耗系数
    
    % 体积计算 (简化)
    V_tooth = 0.3 * l_j * pi*(input.D1-2*input.g) * 0.02;        % 齿部体积
    V_yoke_stator = 0.7 * l_j * pi*(input.D1-2*input.g) * 0.03;  % 定子轭部体积
    
    % 转子轭部体积估计
    D_rotor_avg = (Design.D2 + Design.D2i)/2;                       % 平均转子直径
    A_rotor_yoke = Design.h_j_rotor * pi * D_rotor_avg / input.p;   % 每极转子轭面积
    V_yoke_rotor = A_rotor_yoke * l_j;                              % 转子轭部体积

    % 损耗计算 (Steinmetz方程)
    f_elec = input.p * omega_n / (2*pi);
    Pfe_tooth = (kh * f_elec * B_tooth^1.8 + kc * (f_elec * B_tooth)^2 + ...
                ke * (f_elec * B_tooth)^1.5)* V_tooth;
    Pfe_yoke_stator = (kh * f_elec * B_yoke_stator^1.8 + kc * (f_elec * B_yoke_stator)^2 + ...
                ke * (f_elec * B_yoke_stator)^1.5) * V_yoke_stator;
    Pfe_yoke_rotor = (kh * f_elec * B_yoke_rotor^1.8 + kc * (f_elec * B_yoke_rotor)^2 + ...
                ke * (f_elec * B_yoke_rotor)^1.5) * V_yoke_rotor;

     % 总铁耗
    Pfe = Pfe_tooth + Pfe_yoke_stator + Pfe_yoke_rotor;
end

