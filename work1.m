%% 数据加载与预处理
clear; clc; close all;
load('recorded2.mat'); 

% 系统参数
c = physconst('LightSpeed'); 
f_start = 59.0000e9; 
f_end = 61.9979e9; 
BW = f_end - f_start; 
samples_per_chirp = 64; 
chirps_per_frame = 32; 
range_resolution = c / (2 * BW); 
range_axis = (0:samples_per_chirp/2-1) * range_resolution; 
frame_rate = 1 / 0.125;           % 8Hz采样率
num_frames = size(recorded_data, 1); 
time_axis = (0:num_frames-1)/frame_rate;

% 人体定位
human_distance = 0.2; 
[~, human_range_idx] = min(abs(range_axis - human_distance));

%% 相位信号提取（优化噪声抑制）
static_background = mean(recorded_data(1:10,:,:), 1); 
phase_signal = zeros(num_frames, 1);

for frame_idx = 1:num_frames
    frame_data = squeeze(recorded_data(frame_idx,:,:));
    
    % 改进预处理
    frame_data = frame_data - median(frame_data, 2);
    frame_data = frame_data - squeeze(static_background);
    
    % 距离FFT
    range_fft = fft(frame_data, [], 2);
    
    % 精确选择距离门（插值法）
    target_phase = angle(range_fft(:,human_range_idx));
    
    % 相位差分消除固定相位偏移
    if frame_idx > 1
        prev_phase = angle(range_fft_prev(:,human_range_idx));
        phase_diff = target_phase - prev_phase;
        phase_diff = mod(phase_diff + pi, 2*pi) - pi; % 包裹差分
        phase_signal(frame_idx) = mean(phase_diff);
    end
    range_fft_prev = range_fft; % 保存当前帧用于差分
end

% 补偿首帧数据
phase_signal(1) = phase_signal(2); 
% 相位处理链
phase_unwrapped = unwrap(phase_signal); 
[b,a] = butter(4, [0.1 2]/(frame_rate/2), 'bandpass');
phase_filtered = filtfilt(b,a,phase_unwrapped);

%% 自适应小波分解（频率精确匹配）
wavelet_name = 'db8';             % 更高阶小波提升频率分辨率
decom_level = 7;                  % 增加分解层数
[c, l] = wavedec(phase_filtered, decom_level, wavelet_name);

% 生理信号频带重构（基于0.1-0.5Hz呼吸，0.8-2Hz心跳）
% 小波层频率计算（fs=8Hz）
freq_bands = frame_rate./2.^(1:decom_level+1);
disp('小波分解频带划分（Hz）:');
disp(freq_bands);

% 呼吸信号重构（0.1-0.5Hz对应D4+D5层）
resp_details = [4,5];             % D4:0.25-0.5Hz, D5:0.125-0.25Hz
resp_signal = wrcoef('a', c, l, wavelet_name, decom_level); % A7≈0-0.0625Hz
for i = resp_details
    resp_signal = resp_signal + wrcoef('d', c, l, wavelet_name, i);
end

% 心跳信号重构（0.8-2Hz对应D2+D3层）
heart_details = [2,3];            % D2:1-2Hz, D3:0.5-1Hz
heart_signal = zeros(size(phase_filtered));
for i = heart_details
    heart_signal = heart_signal + wrcoef('d', c, l, wavelet_name, i);
end

% 频带精确滤波
[b_resp,a_resp] = butter(4, [0.1 0.5]/(frame_rate/2), 'bandpass');
[b_heart,a_heart] = butter(4, [0.8 2]/(frame_rate/2), 'bandpass');
resp_signal = filtfilt(b_resp,a_resp,resp_signal);
heart_signal = filtfilt(b_heart,a_heart,heart_signal);


%% CPMV信号生成（改进方法）
CPMV = phase_filtered - resp_signal;

% 预处理增强
CPMV = CPMV - mean(CPMV);
CPMV = detrend(CPMV);
CPMV = CPMV/max(abs(CPMV));

% 计算最大可行分解层数
max_level = wmaxlev(length(CPMV), wavelet_name);

% 自适应小波去噪
if max_level >= 5
    CPMV = wden(CPMV, 'rigrsure', 's', 'sln', 5, wavelet_name);
else
    CPMV = wden(CPMV, 'rigrsure', 's', 'sln', max_level, wavelet_name);
end

SCPMV = gradient(gradient(CPMV)); 

%% 可视化分析（增强显示）
%% 时频分析
figure('Name','时频分析', 'Position',[100 100 800 800])
subplot(211)
pspectrum(resp_signal, frame_rate, 'FrequencyLimits',[0 2])
title('呼吸信号时频谱')
subplot(212)
pspectrum(heart_signal, frame_rate, 'FrequencyLimits',[0 2])
title('心跳信号时频谱')

%% 特征信号显示
figure('Name','生理信号特征', 'Position',[100 100 800 800])
subplot(311)
plot(time_axis, phase_filtered)
title('预处理相位信号'), xlabel('时间 (s)'), grid on

subplot(312)
plot(time_axis, resp_signal)
hold on
[yh,yl] = envelope(resp_signal, 20, 'peak');
plot(time_axis, yh, 'r--'), legend('信号','包络')
title('呼吸信号（0.1-0.5Hz）'), xlabel('时间 (s)'), grid on

subplot(313)
plot(time_axis, heart_signal)
hold on
[yh,yl] = envelope(heart_signal, 20, 'peak');
plot(time_axis, yh, 'r--'), legend('信号','包络')
title('心跳信号（0.8-2Hz）'), xlabel('时间 (s)'), grid on

figure('Name','CPMV信号特征', 'Position',[100 100 800 800])
subplot(211)
plot(time_axis, CPMV)
title('CPMV'), xlabel('时间 (s)'), grid on

subplot(212)
plot(time_axis, SCPMV)
title('SCPMV二阶导数'), xlabel('时间 (s)'), grid on

%% 生理参数计算（鲁棒性增强）
% 转换为列向量并去趋势
resp_signal = detrend(resp_signal(:));
heart_signal = detrend(heart_signal(:));



% 呼吸率计算（带有效性校验）
% 修改findpeaks参数
[resp_locs, resp_pks] = v_findpeaks(resp_signal,'l',16,1:length(resp_signal)); 



% 心率计算（带异常值剔除）
[heart_locs, heart_pks] = v_findpeaks(heart_signal,'l',2,1:length(heart_signal)); 


%% 结果显示

% 峰值可视化验证
figure('Name','峰值检测验证-时间轴','Position',[100 100 800 800])
subplot(211)
plot(time_axis, resp_signal), hold on
plot(time_axis(resp_locs), resp_pks, 'rv', 'MarkerFaceColor','r') % 时间轴标记
title(['呼吸信号'])
xlabel('时间 (s)'), grid on

subplot(212)
plot(time_axis, heart_signal), hold on
plot(time_axis(heart_locs), heart_pks, 'rv', 'MarkerFaceColor','r') % 时间轴标记
title(['心跳信号'])
xlabel('时间 (s)'), grid on
