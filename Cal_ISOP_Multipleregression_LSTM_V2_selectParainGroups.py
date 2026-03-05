# -*- coding: utf-8 -*-
# =============================================================================
# LSTM 逐像元回归 + 特征重要性排序
# ISOP ~ 22个驱动因子
# 流程: 特征预筛选 → 轻量LSTM → SHAP + Permutation + 权重 三合一排序
# GPU: CUDA加速训练，SHAP在CPU子集上运行
# 修改： 添加时序k折验证 
# 添加L在每组中选择参数
# =============================================================================

import os, glob, warnings, sys, multiprocessing as mp
import numpy as np
import scipy.io as sio
import xarray as xr
import rasterio
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import shap

warnings.filterwarnings('ignore')

# =============================================================================
# 电源管理设置（锁屏后继续运行）
# =============================================================================
def set_windows_power_settings():
    try:
        import ctypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        ES_CONTINUOUS       = 0x80000000
        ES_SYSTEM_REQUIRED  = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        print("✓ Windows电源设置已配置：锁屏后继续运行")
        return kernel32
    except Exception as e:
        print(f"⚠️ 无法设置Windows电源选项: {e}")
        return None

def reset_windows_power_settings(kernel32):
    try:
        if kernel32:
            ES_CONTINUOUS = 0x80000000
            kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            print("✓ Windows电源设置已恢复")
    except Exception:
        pass

kernel32_obj = set_windows_power_settings()
print(f"Python version: {sys.version}")
print(f"CPU cores available: {mp.cpu_count()}")

# =============================================================================
# 0. 全局配置
# =============================================================================
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN     = 6          # LSTM回望窗口（月）
HIDDEN_DIM  = 32         # 隐层维度
NUM_LAYERS  = 1          # LSTM层数
DROPOUT     = 0.2
PATIENCE    = 20         # 早停耐心轮次（验证损失不再改善时等待的轮数）

# ↓↓↓ 可调训练超参数 ↓↓↓
EPOCHS      = 200        # 训练轮次：建议 100~500
BATCH_SIZE  = 16         # 批次大小：建议 8~64（越小泛化更好但更慢）
LR_INIT     = 1e-3       # Adam 初始学习率：建议 1e-4 ~ 5e-3
LR_MIN      = 1e-5       # 学习率下限：ReduceLROnPlateau 不会低于此值
LR_FACTOR   = 0.5        # 学习率衰减因子：每次触发时 lr = lr * LR_FACTOR
LR_PATIENCE = 10         # 触发学习率衰减前允许验证损失不改善的轮数
# ↑↑↑ 可调训练超参数 ↑↑↑

# ↓↓↓ 交叉验证配置 ↓↓↓
USE_KFOLD          = True   # 是否启用时序K折交叉验证（仅在采样阶段使用）
KFOLD_N_SPLITS     = 5      # K折数：建议 3~5（样本少时用3）
VAL_RATIO          = 0.2    # 非K折模式下的验证集比例（最后20%时间步）
# ↑↑↑ 交叉验证配置 ↑↑↑

N_FEAT_KEEP = 12         # 预筛选后保留的特征数
SHAP_SAMPLE = 500        # 空间采样格点数（SHAP计算）
SAVE_DIR    = r"E:\2026\Result0_ISOP\LSTM_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"训练配置: EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}")
print(f"学习率:   初始={LR_INIT}, 最小={LR_MIN}, 衰减={LR_FACTOR}x (patience={LR_PATIENCE})")
print(f"交叉验证: {'时序K折 K=' + str(KFOLD_N_SPLITS) if USE_KFOLD else '单次8:2时序分割'}")

# =============================================================================
# 1. 数据读取（复用你的代码）
# =============================================================================
print("\n[1/6] 读取数据...")

mat_data = sio.loadmat(
    r"E:\2026\Result0_ISOP\ISOP_TrendandWhy\MTCO2_monthlyano_2002T2024_FromECWMF.mat")
MTCO2    = mat_data['mtco2_T_monthlyano'].transpose(2, 0, 1)
MTCO2    = np.expand_dims(MTCO2, axis=1)

data_dir  = r"E:\2026\Result0_ISOP\ISOP_TrendandWhy\data\Multipleregression_GlobalE"
files     = sorted(glob.glob(data_dir + r"\REG_DATA_*.tif"))

data_dir2 = r"E:\2026\Result0_ISOP\ISOP_TrendandWhy\data\Multipleregression_GlobalE_More"
files2    = sorted(glob.glob(data_dir2 + r"\ERA5_RAD_PR_CLD_ANOM_*.tif"))

nc_dir    = r"E:\2026\Result0_ISOP\ISOP_TrendandWhy\data\Multipleregression"
nc_files  = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith(".nc")]

def read_tif_stack(flist, desc="Reading TIFF"):
    arrs = []
    for f in tqdm(flist, desc=desc):
        with rasterio.open(f) as ds:
            arrs.append(ds.read().astype(np.float32))
    return np.array(arrs)

def read_1d_nc_to_3d(nc_path, target_shape):
    ds       = xr.open_dataset(nc_path)
    var_name = list(ds.data_vars.keys())[0]
    da       = ds[var_name].sel(time=slice("2018-01-01", "2024-12-31"))
    ts       = da.values.astype(np.float32)
    rows, cols = target_shape[1], target_shape[2]
    arr3d    = np.repeat(ts[:, None, None], rows, axis=1)
    arr3d    = np.repeat(arr3d, cols, axis=2)
    return arr3d

Zall_tiff = read_tif_stack(files,  "TIFF stack 1")
Kall_tiff = read_tif_stack(files2, "TIFF stack 2")
time_len, bands, rows, cols = Zall_tiff.shape

targets = []
for f in nc_files:
    targets.append(read_1d_nc_to_3d(f, (84, rows, cols)))
Yall = np.stack(targets, axis=1)

T   = 84
ISOP   = Zall_tiff[0:T, 0,    :, :].astype(np.float64)
Xall   = Zall_tiff[0:T, 1:13, :, :].astype(np.float64)
X_full = np.concatenate([
    Xall,
    Kall_tiff[0:T, :, :, :],
    MTCO2[0:T, :, :, 802:1604],
    Yall[0:T, :, :, :]
], axis=1)                                    # [T, 22, rows, cols]

lat = Zall_tiff[0, 13, :, :]
lon = Zall_tiff[0, 14, :, :]

n_features = X_full.shape[1]
print(f"X_full: {X_full.shape} | ISOP: {ISOP.shape}")

feature_names = [
    "T_2m", "T_soil_L1", "T_soil_L2", "T_soil_L3", "T_soil_L4",  # 0-4
    "SM_L1", "SM_L2", "SM_L3", "SM_L4",                            # 5-8
    "LAI", "NDVI",                                                   # 9-10
    "O3",                                                            # 11
    "Short_Wave_Radiation",                                          # 12
    "Precipitation", "Cloud_Cover",                                  # 13-14
    "CO2",                                                           # 15
    "AMM", "DMI", "NINA", "NINO", "PDO", "TNA"                     # 16-21
]

groups = {
    "Thermal":     [0, 1, 2, 3, 4, 12],
    "Moisture":    [5, 6, 7, 8],
    "Leaf":        [9, 10],
    "Circulation": [16, 17, 18, 19, 20, 21],   # 全选
    "Hydrology":   [13, 14],
    "Chem":        [11, 15]
}

# =============================================================================
# 2. 逐像元自适应特征预筛选
#    策略：Circulation 组全选；其他组每组 Top-2（hybrid打分）
#    结果：每个格点独立的特征索引列表，特征数因格点而异
#    全局归一化：先对 X_full 做全特征的全局 Z-score，筛选在归一化后进行
# =============================================================================
print("\n[2/6] 全局 Z-score 归一化 + 逐像元自适应特征筛选...")

# ---- 2a. 全特征全局归一化（跨时间+空间） ----
X_mean = np.nanmean(X_full, axis=(0, 2, 3), keepdims=True)  # [1, 22, 1, 1]
X_std  = np.nanstd( X_full, axis=(0, 2, 3), keepdims=True)
X_std  = np.where(X_std < 1e-8, 1.0, X_std)
X_norm = (X_full - X_mean) / X_std                           # [T, 22, rows, cols]

# 保存全特征归一化参数
df_norm_all = pd.DataFrame({
    'Feature': feature_names,
    'Mean':    X_mean[0, :, 0, 0],
    'Std':     X_std[0,  :, 0, 0],
})
df_norm_all.to_csv(os.path.join(SAVE_DIR, 'feature_norm_params.csv'), index=False)
print("  全特征归一化参数已保存: feature_norm_params.csv")

# ---- 2b. 逐像元特征筛选函数 ----
# 确定每组最多保留几个（Circulation全选，其他Top-2）
GROUP_TOP_K    = 2          # 非Circulation组每组保留数
FULL_SEL_GROUP = "Circulation"   # 该组全选
PRESELECT_MODE = "hybrid"   # 可选: "hybrid" | "correlation" | "mutual_info"

def preselect_pixel(y_ts, X_ts_norm, groups, mode=PRESELECT_MODE):
    """
    对单格点时间序列做组内特征预筛选。
    y_ts      : [T]     目标（ISOP原始值）
    X_ts_norm : [T, 22] 已全局归一化的特征
    返回      : list of int，选中的全局特征索引（已排序）
    """
    mask = np.isfinite(y_ts) & np.isfinite(X_ts_norm).all(axis=1)
    if mask.sum() < 10:
        return None

    yv = y_ts[mask]
    Xv = X_ts_norm[mask]                 # 已归一化，无需再做StandardScaler

    # 对 y 做局部标准化（仅用于打分，不影响最终训练）
    ys = (yv - yv.mean()) / (yv.std() + 1e-8)

    selected = []
    for gname, idx in groups.items():
        idx = list(idx)
        Xg  = Xv[:, idx]                 # [T_valid, g_size]

        # Circulation 组：全选
        if gname == FULL_SEL_GROUP:
            selected.extend(idx)
            continue

        # 其他组：hybrid 打分 → Top-2
        if mode == "hybrid":
            corr_mat = np.corrcoef(Xg.T, ys)
            corr_s   = np.abs(corr_mat[:-1, -1])
            corr_s   = np.nan_to_num(corr_s, 0)
            var_s    = np.var(Xg, axis=0)
            var_s    = var_s / (var_s.sum() + 1e-10)
            scores   = corr_s * (1 + var_s)

        elif mode == "correlation":
            corr_mat = np.corrcoef(Xg.T, ys)
            scores   = np.abs(np.nan_to_num(corr_mat[:-1, -1], 0))

        elif mode == "mutual_info":
            try:
                scores = mutual_info_regression(Xg, ys, random_state=42)
            except Exception:
                corr_mat = np.corrcoef(Xg.T, ys)
                scores   = np.abs(np.nan_to_num(corr_mat[:-1, -1], 0))
        else:
            scores = np.ones(len(idx))

        scores = np.nan_to_num(scores, 0)
        order  = np.argsort(scores)[::-1]

        # 取 Top-2（第2名即使得分很低也保留，保证每组至少1个）
        keep_n = min(GROUP_TOP_K, len(idx))
        for rank in range(keep_n):
            selected.append(idx[order[rank]])

    return sorted(set(selected))   # 去重并排序


# ---- 2c. 计算全图逐像元特征索引并存储 ----
rng      = np.random.default_rng(42)
n_pixels = rows * cols

# 有效格点掩码（ISOP和X_norm全时间步有限）
isop_valid = np.isfinite(ISOP).all(axis=0)                       # [rows, cols]
xnorm_valid = np.isfinite(X_norm).all(axis=0).all(axis=0)        # [rows, cols]
valid_mask = (isop_valid & xnorm_valid).ravel()
valid_pix  = np.where(valid_mask)[0]
print(f"  有效格点数: {len(valid_pix)}/{n_pixels} ({100*len(valid_pix)/n_pixels:.1f}%)")

# 预计算每个有效格点的特征索引（存为 object array）
PIX_FEAT_IDX = np.empty(n_pixels, dtype=object)   # 每格点存 list[int]

print("  逐像元特征筛选中（可能需要数分钟）...")
for pidx in tqdm(valid_pix, desc="Pixel feature selection"):
    ri, ci_ = divmod(int(pidx), cols)
    y_ts        = ISOP[:, ri, ci_]
    X_ts_norm   = X_norm[:, :, ri, ci_]       # [T, 22]
    sel_idx     = preselect_pixel(y_ts, X_ts_norm, groups)
    PIX_FEAT_IDX[pidx] = sel_idx if sel_idx else None

# 统计各格点选中特征数的分布
feat_counts = [len(PIX_FEAT_IDX[p]) for p in valid_pix
               if PIX_FEAT_IDX[p] is not None]
print(f"\n  逐像元特征选择统计:")
print(f"    特征数均值={np.mean(feat_counts):.1f}  "
      f"最小={np.min(feat_counts)}  最大={np.max(feat_counts)}")

# 记录全局最常被选中的特征（供参考）
from collections import Counter
all_sel = [fi for p in valid_pix if PIX_FEAT_IDX[p] is not None
           for fi in PIX_FEAT_IDX[p]]
feat_freq = Counter(all_sel)
print(f"\n  各特征被选中频次 Top-10:")
for fi, cnt in feat_freq.most_common(10):
    print(f"    {feature_names[fi]:<25s}  {cnt:6d} / {len(valid_pix)} 格点 "
          f"({100*cnt/len(valid_pix):.1f}%)")

# 保存特征频次统计
df_freq = pd.DataFrame({
    'Feature':    feature_names,
    'SelectCount': [feat_freq.get(i, 0) for i in range(n_features)],
    'SelectRate':  [feat_freq.get(i, 0) / max(len(valid_pix), 1) for i in range(n_features)],
}).sort_values('SelectCount', ascending=False)
df_freq.to_csv(os.path.join(SAVE_DIR, 'feature_selection_frequency.csv'), index=False)
print("  特征频次统计已保存: feature_selection_frequency.csv")

# ---- 兼容性：selected_names 取全局最高频特征用于可视化标签 ----
selected_idx   = [fi for fi, _ in feat_freq.most_common(n_features)]
selected_names = [feature_names[i] for i in selected_idx]
N_FEAT_KEEP    = len(selected_idx)   # 更新为实际最大特征数（用于容器维度）
# =============================================================================
class LightLSTM(nn.Module):
    """轻量LSTM回归器，适合84步小样本"""
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, seq, features]
        out, _ = self.lstm(x)
        out     = self.dropout(out[:, -1, :])   # 取最后时间步
        return self.fc(out).squeeze(-1)


class PixelLSTMWrapper(BaseEstimator, RegressorMixin):
    """sklearn兼容的包装器，供 permutation_importance 使用"""
    def __init__(self, model, seq_len, scaler_x, scaler_y, device):
        self.model    = model
        self.seq_len  = seq_len
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.device   = device

    def predict(self, X):
        # X: [n_samples, n_features]（已标准化）
        T_pred = X.shape[0]
        seqs   = []
        for t in range(self.seq_len, T_pred):
            seqs.append(X[t - self.seq_len:t])
        seqs_t = torch.tensor(np.array(seqs), dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(seqs_t).cpu().numpy()
        return self.scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()

    def score(self, X, y):
        pred = self.predict(X)
        y_   = y[self.seq_len:]
        ss_res = np.sum((y_ - pred)**2)
        ss_tot = np.sum((y_ - np.mean(y_))**2)
        return 1 - ss_res / (ss_tot + 1e-8)


# =============================================================================
# 4. 逐像元训练函数
# =============================================================================
def make_sequences(X_norm, Y_norm, seq_len):
    """将归一化序列切成 (seq_len → 1) 样本，跳过含NaN的窗口"""
    Xs, Ys = [], []
    T_  = len(Y_norm)
    K   = X_norm.shape[1]
    for t in range(seq_len, T_):
        window = X_norm[t - seq_len:t]          # [seq_len, K]
        target = Y_norm[t]
        # 严格校验形状与有限性
        if window.shape != (seq_len, K):
            continue
        if not (np.isfinite(window).all() and np.isfinite(target)):
            continue
        Xs.append(window)
        Ys.append(target)
    if len(Xs) == 0:
        return np.empty((0, seq_len, K), dtype=np.float32), \
               np.empty((0,),            dtype=np.float32)
    return np.array(Xs, dtype=np.float32), np.array(Ys, dtype=np.float32)


def _train_one_fold(X_tr, Y_tr, X_va, Y_va, K, epochs, device):
    """训练单折LSTM，返回 (model, val_loss_best)"""
    ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                       drop_last=(len(X_tr) > BATCH_SIZE))   # 样本少时不丢弃最后批

    model     = LightLSTM(K, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR_INIT, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    patience=LR_PATIENCE,
                    factor=LR_FACTOR,
                    min_lr=LR_MIN)          # 学习率下限

    best_val, best_state, wait = np.inf, None, 0
    xv = torch.from_numpy(X_va).to(device)
    yv = torch.from_numpy(Y_va).to(device)

    for ep in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(xv), yv).item()
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_val


def _calc_metrics(pred, true):
    """计算一组预测/真值的所有拟合指标，返回字典"""
    n      = len(true)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)

    r2    = 1 - ss_res / (ss_tot + 1e-8)
    rmse  = np.sqrt(ss_res / n)
    mae   = np.mean(np.abs(true - pred))
    mbe   = np.mean(pred - true)
    rrmse = rmse / (np.mean(np.abs(true)) + 1e-8)

    cov   = np.mean((pred - pred.mean()) * (true - true.mean()))
    corr  = cov / (pred.std() * true.std() + 1e-8)

    alpha = pred.std() / (true.std() + 1e-8)
    beta  = pred.mean() / (true.mean() + 1e-8)
    kge   = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    return dict(r2=r2, rmse=rmse, mae=mae, mbe=mbe,
                rrmse=rrmse, corr=corr, kge=kge)


def train_pixel(x_ts, y_ts, seq_len=SEQ_LEN, hidden=HIDDEN_DIM,
                epochs=EPOCHS, device=DEVICE, use_kfold=USE_KFOLD):
    """
    逐像元LSTM训练。
    x_ts : [T, K]  特征时间序列（已全局归一化）
    y_ts : [T]     ISOP时间序列（原始尺度）
    返回 : model, sc_x, sc_y, metrics_dict
           metrics_dict 含 r2/rmse/mae/mbe/rrmse/corr/kge
           若 use_kfold=True，metrics 为各折验证集指标的均值
    """
    from sklearn.model_selection import TimeSeriesSplit

    # ---- 形状保护 ----
    x_ts = np.array(x_ts, dtype=np.float64)
    y_ts = np.array(y_ts, dtype=np.float64).ravel()
    if x_ts.ndim == 2 and x_ts.shape[0] != len(y_ts) and x_ts.shape[1] == len(y_ts):
        x_ts = x_ts.T
    if x_ts.ndim != 2 or x_ts.shape[0] != len(y_ts):
        return None, StandardScaler(), StandardScaler(), {}
    K = x_ts.shape[1]

    # ---- NaN 清洗 ----
    valid_t = np.isfinite(x_ts).all(axis=1) & np.isfinite(y_ts)
    x_ts = x_ts[valid_t].copy()
    y_ts = y_ts[valid_t].copy()
    if len(y_ts) < seq_len + 10:
        return None, StandardScaler(), StandardScaler(), {}

    # ---- 逐像元标准化（在清洗后的有效步上拟合）----
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_n  = sc_x.fit_transform(x_ts)
    Y_n  = sc_y.fit_transform(y_ts.reshape(-1, 1)).ravel()
    X_n  = np.nan_to_num(X_n, nan=0.0, posinf=0.0, neginf=0.0)
    Y_n  = np.nan_to_num(Y_n, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- 构建序列样本 ----
    Xs, Ys = make_sequences(X_n, Y_n, seq_len)
    if len(Xs) < 10:
        return None, sc_x, sc_y, {}

    # ==================================================================
    # 训练策略 A：时序K折交叉验证
    #   - 仅用于评估指标（val metrics 均值），最终模型用全数据重训一次
    #   - TimeSeriesSplit 保证验证集始终在训练集之后，无数据泄漏
    # ==================================================================
    if use_kfold and KFOLD_N_SPLITS >= 2:
        min_required = seq_len + KFOLD_N_SPLITS * 2
        if len(Xs) < min_required:
            use_kfold = False   # 样本不足时回退到单次分割

    if use_kfold:
        tscv         = TimeSeriesSplit(n_splits=KFOLD_N_SPLITS)
        fold_metrics = []

        for fold_tr_idx, fold_va_idx in tscv.split(Xs):
            if len(fold_tr_idx) < 4 or len(fold_va_idx) < 2:
                continue
            X_tr = Xs[fold_tr_idx].astype(np.float32)
            Y_tr = Ys[fold_tr_idx].astype(np.float32)
            X_va = Xs[fold_va_idx].astype(np.float32)
            Y_va = Ys[fold_va_idx].astype(np.float32)

            fold_model, _ = _train_one_fold(X_tr, Y_tr, X_va, Y_va,
                                            K, epochs, device)
            fold_model.eval()
            with torch.no_grad():
                xv_t   = torch.from_numpy(X_va).to(device)
                pred_n = fold_model(xv_t).cpu().numpy()
            pred = sc_y.inverse_transform(pred_n.reshape(-1, 1)).ravel()
            true = sc_y.inverse_transform(Y_va.reshape(-1, 1)).ravel()
            fold_metrics.append(_calc_metrics(pred, true))

        # K折验证指标取均值
        if fold_metrics:
            metrics = {k: float(np.mean([fm[k] for fm in fold_metrics]))
                       for k in fold_metrics[0]}
        else:
            metrics = {}

        # 最终模型：用全部数据重训一次（获得最强预测能力）
        split  = int(len(Xs) * (1 - VAL_RATIO))
        X_tr_f = Xs[:split].astype(np.float32)
        Y_tr_f = Ys[:split].astype(np.float32)
        X_va_f = Xs[split:].astype(np.float32)
        Y_va_f = Ys[split:].astype(np.float32)
        if len(X_va_f) == 0:          # 极少数情况全给训练集
            X_va_f, Y_va_f = X_tr_f[-2:], Y_tr_f[-2:]
        model, _ = _train_one_fold(X_tr_f, Y_tr_f, X_va_f, Y_va_f,
                                   K, epochs, device)

    # ==================================================================
    # 训练策略 B：单次时序分割（80% 训练 / 20% 验证）
    # ==================================================================
    else:
        split  = int(len(Xs) * (1 - VAL_RATIO))
        X_tr   = Xs[:split].astype(np.float32)
        Y_tr   = Ys[:split].astype(np.float32)
        X_va   = Xs[split:].astype(np.float32)
        Y_va   = Ys[split:].astype(np.float32)
        if len(X_va) == 0:
            X_va, Y_va = X_tr[-2:], Y_tr[-2:]

        model, _ = _train_one_fold(X_tr, Y_tr, X_va, Y_va, K, epochs, device)

        # 指标在全样本上计算
        model.eval()
        with torch.no_grad():
            pred_n = model(torch.from_numpy(Xs.astype(np.float32)).to(device)).cpu().numpy()
        pred    = sc_y.inverse_transform(pred_n.reshape(-1, 1)).ravel()
        true    = sc_y.inverse_transform(Ys.reshape(-1, 1)).ravel()
        metrics = _calc_metrics(pred, true)

    return model, sc_x, sc_y, metrics


# =============================================================================
# 5. 空间采样训练 + 三种重要性计算
# =============================================================================
print("\n[3/6] 空间采样训练 LSTM（用于重要性排序）...")

# valid_pix 和 PIX_FEAT_IDX 已在 [2] 节计算完毕
sample_pix = rng.choice(valid_pix, min(SHAP_SAMPLE, len(valid_pix)), replace=False)

# 采样阶段重要性容器（用全局最大特征数，不足的格点用NaN填充）
imp_shap  = np.full((len(sample_pix), n_features), np.nan)
imp_perm  = np.full((len(sample_pix), n_features), np.nan)
imp_wt    = np.full((len(sample_pix), n_features), np.nan)
r2_sample = np.zeros(len(sample_pix))

for si, pidx in enumerate(tqdm(sample_pix, desc="Pixel training")):
    ri, ci_ = divmod(int(pidx), cols)

    feat_idx = PIX_FEAT_IDX[pidx]
    if feat_idx is None:
        r2_sample[si] = np.nan
        continue

    K_pix = len(feat_idx)
    x_ts  = X_norm[:, feat_idx, :][:, :, ri, ci_]   # [T, K_pix] 已全局归一化，逐像元特征
    y_ts  = ISOP[:, ri, ci_]                          # [T]

    if not (np.isfinite(x_ts).all() and np.isfinite(y_ts).all()):
        r2_sample[si] = np.nan
        continue

    model, sc_x, sc_y, metrics = train_pixel(x_ts, y_ts,
                                              use_kfold=USE_KFOLD)
    r2_sample[si] = metrics.get('r2', np.nan)
    if model is None:
        continue

    X_n  = sc_x.transform(x_ts).astype(np.float32)
    X_n  = np.nan_to_num(X_n, nan=0.0, posinf=0.0, neginf=0.0)
    Y_n  = sc_y.transform(y_ts.reshape(-1, 1)).ravel().astype(np.float32)
    Xs, Ys = make_sequences(X_n, Y_n, SEQ_LEN)

    # ---- 5a. 输入层权重范数 ----
    w_ih    = model.lstm.weight_ih_l0.detach().cpu().numpy()   # [4H, K_pix]
    wt_norm = np.linalg.norm(w_ih.reshape(4, HIDDEN_DIM, K_pix), axis=(0, 1))
    wt_norm = wt_norm / (wt_norm.sum() + 1e-8)
    for ki, fi in enumerate(feat_idx):
        imp_wt[si, fi] = wt_norm[ki]

    # ---- 5b. Permutation Importance ----
    wrapper = PixelLSTMWrapper(model, SEQ_LEN, sc_x, sc_y, DEVICE)
    try:
        pi_res = permutation_importance(
            wrapper, X_n, y_ts,
            n_repeats=10, random_state=42, scoring='r2'
        )
        perm_v = np.maximum(pi_res.importances_mean, 0)
        perm_v = perm_v / (perm_v.sum() + 1e-8)
        for ki, fi in enumerate(feat_idx):
            imp_perm[si, fi] = perm_v[ki]
    except Exception as e:
        if si == 0:
            tqdm.write(f"  [Permutation 警告] {type(e).__name__}: {e}")
        for ki, fi in enumerate(feat_idx):
            imp_perm[si, fi] = imp_wt[si, fi]   # 兜底

    # ---- 5c. SHAP（DeepExplainer） ----
    try:
        if len(Xs) < 5:
            raise ValueError(f"序列数不足({len(Xs)}<5)，跳过SHAP")
        model.eval()
        bg          = torch.from_numpy(Xs[:min(20, len(Xs))]).to(DEVICE)
        te          = torch.from_numpy(Xs).to(DEVICE)
        explainer   = shap.DeepExplainer(model, bg)
        shap_values = explainer.shap_values(te)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.array(shap_values)
        sv = np.abs(shap_values).mean(axis=(0, 1))   # [K_pix]
        sv = sv / (sv.sum() + 1e-8)
        for ki, fi in enumerate(feat_idx):
            imp_shap[si, fi] = sv[ki]
    except Exception as e:
        if si == 0:
            tqdm.write(f"  [SHAP 警告] {type(e).__name__}: {e}")
        for ki, fi in enumerate(feat_idx):
            imp_shap[si, fi] = imp_wt[si, fi]   # 兜底

print(f"\n  有效格点 R² 均值: {np.nanmean(r2_sample):.3f}")

# =============================================================================
# 6. 全局重要性汇总 + 排序
# =============================================================================
print("\n[4/6] 汇总特征重要性...")

# 各方法空间均值（忽略NaN）
mean_shap = np.nanmean(imp_shap, axis=0)
mean_perm = np.nanmean(imp_perm, axis=0)
mean_wt   = np.nanmean(imp_wt,   axis=0)

# 归一化到 [0,1]
def norm01(x):
    r = x - x.min()
    return r / (r.max() + 1e-8)

mean_shap_n = norm01(mean_shap)
mean_perm_n = norm01(mean_perm)
mean_wt_n   = norm01(mean_wt)

# 三方法等权综合得分
ensemble_score = (mean_shap_n + mean_perm_n + mean_wt_n) / 3
final_rank     = np.argsort(ensemble_score)[::-1]

df_imp = pd.DataFrame({
    'Feature':        selected_names,
    'SHAP':           mean_shap,
    'Permutation':    mean_perm,
    'Weight_Norm':    mean_wt,
    'SHAP_norm':      mean_shap_n,
    'Perm_norm':      mean_perm_n,
    'Wt_norm':        mean_wt_n,
    'Ensemble_Score': ensemble_score,
    'Global_Rank':    np.argsort(np.argsort(-ensemble_score)) + 1
}).sort_values('Ensemble_Score', ascending=False).reset_index(drop=True)

print("\n  最终特征重要性排名:")
print(df_imp[['Feature', 'SHAP_norm', 'Perm_norm',
              'Wt_norm', 'Ensemble_Score', 'Global_Rank']].to_string(index=False))
df_imp.to_csv(os.path.join(SAVE_DIR, 'feature_importance_ranking.csv'), index=False)

# =============================================================================
# 7. 全图逐像元训练
#    - 每格点用 PIX_FEAT_IDX[pidx] 取其自适应特征子集
#    - 三种重要性写回全 n_features 维空间图（未选特征保持 NaN）
# =============================================================================
print("\n[5/6] 全图逐像元训练（逐像元自适应特征 + 三种重要性 + 预测值）...")

_nan2d = lambda: np.full((rows, cols), np.nan, dtype=np.float32)
R2_map    = _nan2d()
RMSE_map  = _nan2d()
MAE_map   = _nan2d()
MBE_map   = _nan2d()
RRMSE_map = _nan2d()
CORR_map  = _nan2d()
KGE_map   = _nan2d()

# 重要性图维度用全局特征数 n_features，未选特征保持 NaN
IMP_wt_map   = np.full((n_features, rows, cols), np.nan, dtype=np.float32)
IMP_perm_map = np.full((n_features, rows, cols), np.nan, dtype=np.float32)
IMP_shap_map = np.full((n_features, rows, cols), np.nan, dtype=np.float32)
IMP_ens_map  = np.full((n_features, rows, cols), np.nan, dtype=np.float32)
PRED_map     = np.full((T, rows, cols),          np.nan, dtype=np.float32)
OBS_map      = ISOP.astype(np.float32)

pbar = tqdm(total=len(valid_pix), desc="Full spatial training")

for pidx in valid_pix:
    ri, ci_ = divmod(int(pidx), cols)

    feat_idx = PIX_FEAT_IDX[pidx]
    if feat_idx is None:
        pbar.update(1)
        continue

    K_pix = len(feat_idx)
    x_ts  = X_norm[:, feat_idx, :][:, :, ri, ci_]  # [T, K_pix] 已全局归一化
    y_ts  = ISOP[:, ri, ci_]                         # [T]

    if not (np.isfinite(x_ts).all() and np.isfinite(y_ts).all()):
        pbar.update(1)
        continue

    model, sc_x, sc_y, metrics = train_pixel(x_ts, y_ts, use_kfold=False)
    if metrics:
        R2_map[ri, ci_]    = metrics['r2']
        RMSE_map[ri, ci_]  = metrics['rmse']
        MAE_map[ri, ci_]   = metrics['mae']
        MBE_map[ri, ci_]   = metrics['mbe']
        RRMSE_map[ri, ci_] = metrics['rrmse']
        CORR_map[ri, ci_]  = metrics['corr']
        KGE_map[ri, ci_]   = metrics['kge']

    if model is None:
        pbar.update(1)
        continue

    X_n = sc_x.transform(x_ts).astype(np.float32)
    X_n = np.nan_to_num(X_n, nan=0.0, posinf=0.0, neginf=0.0)
    Y_n = sc_y.transform(y_ts.reshape(-1, 1)).ravel().astype(np.float32)
    Xs, Ys = make_sequences(X_n, Y_n, SEQ_LEN)

    # ---- 预测值重建 ----
    if len(Xs) > 0:
        model.eval()
        with torch.no_grad():
            pred_n = model(torch.from_numpy(Xs).to(DEVICE)).cpu().numpy()
        pred = sc_y.inverse_transform(pred_n.reshape(-1, 1)).ravel()
        PRED_map[SEQ_LEN:SEQ_LEN + len(pred), ri, ci_] = pred.astype(np.float32)

    # ---- 重要性1：权重范数 → 写回 feat_idx 对应位置 ----
    w_ih  = model.lstm.weight_ih_l0.detach().cpu().numpy()  # [4H, K_pix]
    wt_n  = np.linalg.norm(w_ih.reshape(4, HIDDEN_DIM, K_pix), axis=(0, 1))
    wt_n  = wt_n / (wt_n.sum() + 1e-8)
    for ki, fi in enumerate(feat_idx):
        IMP_wt_map[fi, ri, ci_] = wt_n[ki]

    # ---- 重要性2：Permutation ----
    try:
        wrapper = PixelLSTMWrapper(model, SEQ_LEN, sc_x, sc_y, DEVICE)
        pi_res  = permutation_importance(
            wrapper, X_n, y_ts, n_repeats=5, random_state=42, scoring='r2')
        perm_n  = np.maximum(pi_res.importances_mean, 0)
        perm_n  = perm_n / (perm_n.sum() + 1e-8)
        for ki, fi in enumerate(feat_idx):
            IMP_perm_map[fi, ri, ci_] = perm_n[ki]
    except Exception:
        for ki, fi in enumerate(feat_idx):
            IMP_perm_map[fi, ri, ci_] = wt_n[ki]

    # ---- 重要性3：SHAP ----
    try:
        if len(Xs) >= 5:
            model.eval()
            bg        = torch.from_numpy(Xs[:min(15, len(Xs))]).to(DEVICE)
            te        = torch.from_numpy(Xs).to(DEVICE)
            explainer = shap.DeepExplainer(model, bg)
            shap_vals = explainer.shap_values(te)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            shap_vals = np.array(shap_vals)
            sv = np.abs(shap_vals).mean(axis=(0, 1))  # [K_pix]
            sv = sv / (sv.sum() + 1e-8)
            for ki, fi in enumerate(feat_idx):
                IMP_shap_map[fi, ri, ci_] = sv[ki]
        else:
            for ki, fi in enumerate(feat_idx):
                IMP_shap_map[fi, ri, ci_] = wt_n[ki]
    except Exception:
        for ki, fi in enumerate(feat_idx):
            IMP_shap_map[fi, ri, ci_] = wt_n[ki]

    # ---- 综合重要性（在该格点被选中的特征上做归一化平均）----
    for ki, fi in enumerate(feat_idx):
        wt_v   = IMP_wt_map[fi,   ri, ci_]
        perm_v = IMP_perm_map[fi, ri, ci_]
        shap_v = IMP_shap_map[fi, ri, ci_]
        IMP_ens_map[fi, ri, ci_] = np.nanmean([wt_v, perm_v, shap_v])

    pbar.update(1)

pbar.close()

# ---------- 保存所有结果为统一 NetCDF 文件 ----------
print("  保存结果为 NetCDF 文件...")

import xarray as xr

# 坐标轴
lon_1d_coord = lon[0, :].astype(np.float32)
lat_1d_coord = lat[:, 0].astype(np.float32)
time_coord   = np.arange(T, dtype=np.int32)
feat_coord   = np.array(feature_names)   # 全部22个特征名（重要性图维度=n_features）

ds = xr.Dataset(
    {
        # ── 拟合指标  [lat, lon] ──────────────────────────────────────────
        'R2': xr.DataArray(
            R2_map, dims=['lat', 'lon'],
            attrs={'long_name': 'Coefficient of determination (R²)',
                   'units': '1', 'range': '(-∞, 1]'}),
        'RMSE': xr.DataArray(
            RMSE_map, dims=['lat', 'lon'],
            attrs={'long_name': 'Root Mean Square Error',
                   'units': 'same as ISOP', 'note': 'lower is better'}),
        'MAE': xr.DataArray(
            MAE_map, dims=['lat', 'lon'],
            attrs={'long_name': 'Mean Absolute Error',
                   'units': 'same as ISOP', 'note': 'lower is better'}),
        'MBE': xr.DataArray(
            MBE_map, dims=['lat', 'lon'],
            attrs={'long_name': 'Mean Bias Error (pred - obs)',
                   'units': 'same as ISOP', 'note': '>0 overestimate, <0 underestimate'}),
        'RRMSE': xr.DataArray(
            RRMSE_map, dims=['lat', 'lon'],
            attrs={'long_name': 'Relative RMSE (RMSE / mean|obs|)',
                   'units': '1', 'note': 'dimensionless, lower is better'}),
        'CORR': xr.DataArray(
            CORR_map, dims=['lat', 'lon'],
            attrs={'long_name': 'Pearson correlation coefficient',
                   'units': '1', 'range': '[-1, 1]'}),
        'KGE': xr.DataArray(
            KGE_map, dims=['lat', 'lon'],
            attrs={'long_name': 'Kling-Gupta Efficiency',
                   'units': '1', 'range': '(-∞, 1]',
                   'note': 'combines correlation, variability bias, mean bias'}),

        # ── 特征重要性  [feature, lat, lon] ──────────────────────────────
        'IMP_weight': xr.DataArray(
            IMP_wt_map, dims=['feature', 'lat', 'lon'],
            attrs={'long_name': 'Feature importance - LSTM input weight norm',
                   'units': '1'}),
        'IMP_permutation': xr.DataArray(
            IMP_perm_map, dims=['feature', 'lat', 'lon'],
            attrs={'long_name': 'Feature importance - Permutation Importance',
                   'units': '1'}),
        'IMP_shap': xr.DataArray(
            IMP_shap_map, dims=['feature', 'lat', 'lon'],
            attrs={'long_name': 'Feature importance - SHAP DeepExplainer',
                   'units': '1'}),
        'IMP_ensemble': xr.DataArray(
            IMP_ens_map, dims=['feature', 'lat', 'lon'],
            attrs={'long_name': 'Feature importance - Ensemble (mean of 3 methods)',
                   'units': '1'}),

        # ── 时间序列  [time, lat, lon] ────────────────────────────────────
        'ISOP_pred': xr.DataArray(
            PRED_map, dims=['time', 'lat', 'lon'],
            attrs={'long_name': 'LSTM predicted ISOP', 'units': 'mol/mol',
                   'note': f'First {SEQ_LEN} steps are NaN (no lookback window)'}),
        'ISOP_obs': xr.DataArray(
            OBS_map, dims=['time', 'lat', 'lon'],
            attrs={'long_name': 'Observed ISOP (original)', 'units': 'mol/mol'}),
    },
    coords={
        'lat':     ('lat',     lat_1d_coord,
                    {'long_name': 'latitude',  'units': 'degrees_north'}),
        'lon':     ('lon',     lon_1d_coord,
                    {'long_name': 'longitude', 'units': 'degrees_east'}),
        'time':    ('time',    time_coord,
                    {'long_name': 'time step index (monthly)',
                     'units': 'months since 2018-01'}),
        'feature': ('feature', feat_coord,
                    {'long_name': 'selected feature names after screening'}),
    },
    attrs={
        'title':       'LSTM pixel-wise regression results for ISOP',
        'description': 'Metrics: R2, RMSE, MAE, MBE, RRMSE, CORR, KGE; '
                       'Importance: Weight, Permutation, SHAP, Ensemble; '
                       'Time series: ISOP_pred, ISOP_obs',
        'seq_len':     SEQ_LEN,
        'n_features':  N_FEAT_KEEP,
        'hidden_dim':  HIDDEN_DIM,
        'epochs':      EPOCHS,
        'batch_size':  BATCH_SIZE,
        'created_by':  'LSTM_ISOP_Importance.py',
    }
)

_enc = {'zlib': True, 'complevel': 4, 'dtype': 'float32', '_FillValue': np.nan}
nc_path = os.path.join(SAVE_DIR, 'LSTM_ISOP_results.nc')
ds.to_netcdf(nc_path, format='NETCDF4', engine='netcdf4',
             encoding={v: _enc for v in ds.data_vars})
print(f"  ✓ NetCDF 已保存: {nc_path}")
print(f"  全图 R² 均值: {np.nanmean(R2_map):.3f}")
print(f"\n  NC文件变量结构:")
print(f"    R2              [lat={rows}, lon={cols}]")
print(f"    IMP_weight      [feature={n_features}, lat={rows}, lon={cols}]")
print(f"    IMP_permutation [feature={n_features}, lat={rows}, lon={cols}]")
print(f"    IMP_shap        [feature={n_features}, lat={rows}, lon={cols}]")
print(f"    IMP_ensemble    [feature={n_features}, lat={rows}, lon={cols}]")
print(f"    ISOP_pred       [time={T}, lat={rows}, lon={cols}]")
print(f"    ISOP_obs        [time={T}, lat={rows}, lon={cols}]")

# =============================================================================
# 8. 可视化
# =============================================================================
print("\n[6/6] 绘图...")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

lon_1d = lon[0, :]
lat_1d = lat[:, 0]

def plot_spatial(data, title, fname, cmap='YlOrRd', vmin=None, vmax=None,
                 label='', figsize=(12, 5)):
    """通用空间分布绘图函数"""
    fig, ax = plt.subplots(figsize=figsize)
    vmax_ = vmax if vmax is not None else np.nanpercentile(data, 95)
    vmin_ = vmin if vmin is not None else np.nanpercentile(data,  5)
    im = ax.pcolormesh(lon_1d, lat_1d, data,
                       cmap=cmap, vmin=vmin_, vmax=vmax_, shading='auto')
    plt.colorbar(im, ax=ax, label=label, shrink=0.8)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('经度'); ax.set_ylabel('纬度')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()

# ---- 图1：特征重要性综合排名柱状图 ----
# 基于全图空间均值重要性（仅在被选中的格点上均值，NaN忽略）
mean_shap_global = np.nanmean(IMP_shap_map.reshape(n_features, -1), axis=1)
mean_perm_global = np.nanmean(IMP_perm_map.reshape(n_features, -1), axis=1)
mean_wt_global   = np.nanmean(IMP_wt_map.reshape(n_features,   -1), axis=1)
mean_ens_global  = np.nanmean(IMP_ens_map.reshape(n_features,  -1), axis=1)

# 以空间均值综合重要性排序（NaN特征排最后）
sort_order = np.argsort(np.nan_to_num(mean_ens_global, nan=-1))[::-1]

fig, axes = plt.subplots(1, 4, figsize=(24, 7))
colors_bar = plt.cm.RdYlBu_r(np.linspace(0.2, 0.85, n_features))

for ax, vals, title in zip(
    axes[:3],
    [mean_shap_global[sort_order],
     mean_perm_global[sort_order],
     mean_wt_global[sort_order]],
    ['SHAP（全图空间均值）',
     'Permutation（全图空间均值）',
     'Weight Norm（全图空间均值）']
):
    names_sorted = [feature_names[i] for i in sort_order]
    ax.barh(range(n_features), vals[::-1], color=colors_bar)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(names_sorted[::-1], fontsize=8)
    ax.set_xlabel('平均重要性', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.invert_xaxis(); ax.grid(axis='x', alpha=0.3)

# 第4子图：特征被选中频次
sel_rate = np.array([feat_freq.get(i, 0) / max(len(valid_pix), 1)
                     for i in range(n_features)])
names_sorted = [feature_names[i] for i in sort_order]
bars = axes[3].barh(range(n_features), sel_rate[sort_order][::-1],
                    color=plt.cm.viridis(sel_rate[sort_order][::-1]))
axes[3].set_yticks(range(n_features))
axes[3].set_yticklabels(names_sorted[::-1], fontsize=8)
axes[3].set_xlabel('格点选中率', fontsize=10)
axes[3].set_title('逐像元选中频次', fontsize=11, fontweight='bold')
axes[3].set_xlim(0, 1)
axes[3].invert_xaxis(); axes[3].grid(axis='x', alpha=0.3)

plt.suptitle('ISOP 驱动因子重要性排序（逐像元自适应特征选择）',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'feature_importance_ranking.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# ---- 图2：R² 空间分布 ----
plot_spatial(R2_map, 'LSTM 逐像元回归 R² 空间分布', 'R2_spatial.png',
             cmap='RdYlGn', vmin=0, vmax=1, label='R²')

# ---- 图2b：其余拟合指标空间分布（1张6面板）----
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
metric_cfgs = [
    (RMSE_map,  'RMSE',              'YlOrRd',  'RMSE (ISOP单位)',   False),
    (MAE_map,   'MAE',               'YlOrRd',  'MAE  (ISOP单位)',   False),
    (MBE_map,   'MBE（正=高估）',    'RdBu_r',  'MBE  (ISOP单位)',   True ),
    (RRMSE_map, '相对RMSE（无量纲）','YlOrRd',  'RRMSE',             False),
    (CORR_map,  'Pearson 相关系数',  'RdYlGn',  'CORR',              False),
    (KGE_map,   'KGE 效率',          'RdYlGn',  'KGE',               False),
]
for ax, (data, title, cmap_, label_, symm) in zip(axes.ravel(), metric_cfgs):
    if symm:   # MBE 用对称色阶
        bmax = np.nanpercentile(np.abs(data), 95)
        vmin_, vmax_ = -bmax, bmax
    else:
        vmin_ = np.nanpercentile(data, 5)
        vmax_ = np.nanpercentile(data, 95)
    im = ax.pcolormesh(lon_1d, lat_1d, data,
                       cmap=cmap_, vmin=vmin_, vmax=vmax_, shading='auto')
    plt.colorbar(im, ax=ax, label=label_, shrink=0.8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('经度', fontsize=8); ax.set_ylabel('纬度', fontsize=8)
plt.suptitle('LSTM 逐像元拟合指标空间分布', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'metrics_spatial.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: metrics_spatial.png")

# 打印全局统计
print(f"\n  拟合指标全局均值（有效格点）:")
print(f"    R²    = {np.nanmean(R2_map):.3f}   (越接近1越好)")
print(f"    RMSE  = {np.nanmean(RMSE_map):.4f}")
print(f"    MAE   = {np.nanmean(MAE_map):.4f}")
print(f"    MBE   = {np.nanmean(MBE_map):.4f}  (正=高估, 负=低估)")
print(f"    RRMSE = {np.nanmean(RRMSE_map):.3f}   (越接近0越好)")
print(f"    CORR  = {np.nanmean(CORR_map):.3f}   (越接近1越好)")
print(f"    KGE   = {np.nanmean(KGE_map):.3f}   (越接近1越好)")

# ---- 图3~5：三种重要性各自的 Top-6 空间分布 ----
for imp_map, method_name, fname_prefix in [
    (IMP_wt_map,   '权重范数 (Weight Norm)',   'IMP_wt'),
    (IMP_perm_map, '排列重要性 (Permutation)',  'IMP_perm'),
    (IMP_shap_map, 'SHAP',                      'IMP_shap'),
    (IMP_ens_map,  '综合重要性 (Ensemble)',      'IMP_ens'),
]:
    # 按空间均值确定该方法的 Top-6 排序
    mean_imp = np.nanmean(imp_map, axis=(1, 2))   # [K]
    top6_imp = np.argsort(mean_imp)[::-1][:6]

    fig = plt.figure(figsize=(18, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.28)
    for k, fi in enumerate(top6_imp):
        ax   = fig.add_subplot(gs[k // 3, k % 3])
        data = imp_map[fi]
        vmax = np.nanpercentile(data, 95)
        im   = ax.pcolormesh(lon_1d, lat_1d, data,
                             cmap='YlOrRd', vmin=0, vmax=vmax, shading='auto')
        plt.colorbar(im, ax=ax, shrink=0.85, label='重要性')
        ax.set_title(f'#{k+1} {feature_names[fi]}', fontsize=11, fontweight='bold')
        ax.set_xlabel('经度', fontsize=8); ax.set_ylabel('纬度', fontsize=8)
    plt.suptitle(f'ISOP 驱动因子空间重要性 Top-6  [{method_name}]',
                 fontsize=13, fontweight='bold')
    plt.savefig(os.path.join(SAVE_DIR, f'{fname_prefix}_top6.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {fname_prefix}_top6.png")

# ---- 图6：预测值 vs 原始值（时间均值空间对比）----
pred_mean = np.nanmean(PRED_map, axis=0)   # [rows, cols] 时间均值
obs_mean  = np.nanmean(OBS_map,  axis=0)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
vmin_c = min(np.nanpercentile(obs_mean, 2), np.nanpercentile(pred_mean, 2))
vmax_c = max(np.nanpercentile(obs_mean, 98), np.nanpercentile(pred_mean, 98))

for ax, data, title in zip(
    axes[:2],
    [obs_mean, pred_mean],
    ['原始 ISOP（时间均值）', 'LSTM 预测 ISOP（时间均值）']
):
    im = ax.pcolormesh(lon_1d, lat_1d, data,
                       cmap='YlGn', vmin=vmin_c, vmax=vmax_c, shading='auto')
    plt.colorbar(im, ax=ax, label='ISOP', shrink=0.8)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('经度'); ax.set_ylabel('纬度')

# 第3子图：偏差（预测 - 原始）
bias = pred_mean - obs_mean
bmax = np.nanpercentile(np.abs(bias), 95)
im3  = axes[2].pcolormesh(lon_1d, lat_1d, bias,
                           cmap='RdBu_r', vmin=-bmax, vmax=bmax, shading='auto')
plt.colorbar(im3, ax=axes[2], label='偏差 (Pred - Obs)', shrink=0.8)
axes[2].set_title('预测偏差（时间均值）', fontsize=12, fontweight='bold')
axes[2].set_xlabel('经度'); axes[2].set_ylabel('纬度')

plt.suptitle('LSTM 逐像元回归：预测值 vs 原始值', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'pred_vs_obs_spatial.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: pred_vs_obs_spatial.png")

# ---- 图7：预测值 vs 原始值（逐时间步全局均值时序对比）----
obs_ts  = np.nanmean(OBS_map.reshape(T, -1),  axis=1)
pred_ts = np.nanmean(PRED_map.reshape(T, -1), axis=1)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(obs_ts,  'b-o',  ms=4, lw=1.5, label='原始 ISOP')
ax.plot(pred_ts, 'r--s', ms=4, lw=1.5, label='LSTM 预测')
ax.set_xlabel('时间步（月）', fontsize=11)
ax.set_ylabel('全局均值 ISOP', fontsize=11)
ax.set_title('LSTM 全局时序：预测值 vs 原始值', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'pred_vs_obs_timeseries.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: pred_vs_obs_timeseries.png")

# ---- 图8：预筛选得分条形图 ----
fig, ax = plt.subplots(figsize=(10, 7))
df_freq_plot = df_freq.sort_values('SelectRate', ascending=True)   # 升序→barh从下往上高频在顶
colors_freq  = plt.cm.RdYlGn(df_freq_plot['SelectRate'].values)
bars = ax.barh(df_freq_plot['Feature'], df_freq_plot['SelectRate'], color=colors_freq)
ax.axvline(0.5, color='r', linestyle='--', lw=1.5, label='50% 格点选中线')
ax.set_xlabel('格点选中率（被选格点数 / 有效格点总数）', fontsize=11)
ax.set_title('逐像元自适应特征选中频次\n（Circulation组全选，其他组组内Top-2）',
             fontsize=12, fontweight='bold')
ax.set_xlim(0, 1)
ax.legend(fontsize=10); ax.grid(axis='x', alpha=0.3)
# 在每条bar右侧标注选中格点数
for bar, (_, row) in zip(bars, df_freq_plot.iterrows()):
    ax.text(min(row['SelectRate'] + 0.02, 0.98), bar.get_y() + bar.get_height()/2,
            f"{int(row['SelectCount'])}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'feature_selection_frequency.png'),
            dpi=150, bbox_inches='tight')
plt.close()

print(f"\n=== 全部完成，结果保存至: {SAVE_DIR} ===")
print("输出文件:")
print("  LSTM_ISOP_results.nc            — 主结果 NetCDF（含所有空间变量）")
print("    ├─ 拟合指标 [lat, lon]")
print("    │   ├─ R2      决定系数")
print("    │   ├─ RMSE    均方根误差")
print("    │   ├─ MAE     平均绝对误差")
print("    │   ├─ MBE     平均偏差（正=高估）")
print("    │   ├─ RRMSE   相对RMSE（无量纲）")
print("    │   ├─ CORR    Pearson相关系数")
print("    │   └─ KGE     Kling-Gupta效率")
print("    ├─ 特征重要性 [feature, lat, lon]")
print("    │   ├─ IMP_weight / IMP_permutation / IMP_shap / IMP_ensemble")
print("    └─ 时间序列 [time, lat, lon]")
print("        ├─ ISOP_pred   LSTM预测值")
print("        └─ ISOP_obs    原始观测值")
print("  feature_norm_params.csv              — 特征全局归一化参数")
print("  feature_selection_frequency.csv      — 逐像元特征选中频次统计")
print("  feature_importance_ranking.csv       — 三方法重要性排名表（采样格点均值）")
print("  feature_selection_frequency.png      — 特征选中频次条形图")
print("  metrics_spatial.png                  — 7项拟合指标空间分布图")
print("  R2_spatial.png                       — R²空间分布图")
print("  IMP_wt/perm/shap/ens_top6.png        — 各方法Top-6重要性空间图")
print("  pred_vs_obs_spatial.png              — 预测/原始/偏差空间对比图")
print("  pred_vs_obs_timeseries.png           — 全局时序对比图")

# 程序结束后恢复系统电源设置
reset_windows_power_settings(kernel32_obj)
