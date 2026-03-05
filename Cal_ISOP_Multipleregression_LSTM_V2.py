
# =============================================================================
# LSTM 逐像元回归 + 特征重要性排序
# ISOP ~ 22个驱动因子
# 流程: 特征预筛选 → 轻量LSTM → SHAP + Permutation + 权重 三合一排序
# GPU: CUDA加速训练，SHAP在CPU子集上运行
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
HIDDEN_DIM  = 32         # 隐层维度（小样本用轻量结构）
NUM_LAYERS  = 1          # LSTM层数
DROPOUT     = 0.2
LR          = 1e-3
PATIENCE    = 20         # 早停耐心轮次

# ↓↓↓ 可调训练超参数 ↓↓↓
EPOCHS      = 200        # 训练轮次：建议范围 100~500，样本少时不宜过大
BATCH_SIZE  = 16         # 批次大小：建议范围 8~64，越小梯度噪声越大但泛化更好
# ↑↑↑ 可调训练超参数 ↑↑↑

N_FEAT_KEEP = 12         # 预筛选后保留的特征数
SHAP_SAMPLE = 500        # 空间采样格点数（SHAP计算）
SAVE_DIR    = r"E:\2026\Result0_ISOP\LSTM_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"训练配置: EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, SEQ_LEN={SEQ_LEN}")

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

n_features = X_full.shape[1]   # 22
print(f"X_full: {X_full.shape} | ISOP: {ISOP.shape}")

feature_names = [
    "T_2m", "T_soil_L1", "T_soil_L2", "T_soil_L3", "T_soil_L4",
    "SM_L1", "SM_L2", "SM_L3", "SM_L4",
    "LAI", "NDVI", "O3",
    "Short_Wave_Radiation", "Precipitation", "Cloud_Cover",
    "CO2",
    "AMM", "DMI", "NINA", "NINO", "PDO", "TNA"
]

groups = {
    "Thermal":     [0, 1, 2, 3, 4, 12],
    "Moisture":    [5, 6, 7, 8],
    "Leaf":        [9, 10],
    "Circulation": [16, 17, 18, 20, 21],
    "Hydrology":   [13, 14],
    "Chem":        [11, 15]
}

# =============================================================================
# 2. 特征预筛选（全局统计，快速降维）
# =============================================================================
print("\n[2/6] 特征预筛选（Spearman + 互信息 联合筛选）...")

# 展平空间维度，随机采样1万格点做全局筛选
rng      = np.random.default_rng(42)
n_pixels = rows * cols
sample_n = min(10000, n_pixels)
pix_idx  = rng.choice(n_pixels, sample_n, replace=False)

# [sample_n × T] → 展平时间和格点
X_flat = X_full.reshape(T, n_features, -1)[:, :, pix_idx]  # [T, 22, sample]
Y_flat = ISOP.reshape(T, -1)[:, pix_idx]                   # [T, sample]

# 每个特征与ISOP的全局Spearman相关（时间维度）
X_s = X_flat.reshape(T, n_features, -1).transpose(2, 0, 1)  # [sample, T, 22]
Y_s = Y_flat.T                                                # [sample, T]

spearman_scores = np.zeros(n_features)
mi_scores       = np.zeros(n_features)

yi_all = Y_s.ravel()
for fi in range(n_features):
    xi = X_s[:, :, fi].ravel()    # [sample*T]
    valid_sp = np.isfinite(xi) & np.isfinite(yi_all)
    if valid_sp.sum() < 10:
        spearman_scores[fi] = 0.0
        continue
    r, _ = spearmanr(xi[valid_sp], yi_all[valid_sp])
    spearman_scores[fi] = abs(r) if np.isfinite(r) else 0.0

# 互信息（取时间序列展平后的采样子集，控制速度）
sub = min(5000, sample_n * T)
idx_sub = rng.choice(sample_n * T, sub, replace=False)
X_mi = X_s.reshape(-1, n_features)[idx_sub]
Y_mi = Y_s.ravel()[idx_sub]

# 过滤含 NaN/Inf 的样本行（X任意特征或Y含NaN均丢弃）
valid_mi = np.isfinite(X_mi).all(axis=1) & np.isfinite(Y_mi)
X_mi     = X_mi[valid_mi]
Y_mi     = Y_mi[valid_mi]
print(f"  互信息有效样本: {valid_mi.sum()}/{len(valid_mi)}")

if len(X_mi) < 10:
    print("  警告: 有效样本不足10，MI得分全置0")
    mi_scores = np.zeros(n_features)
else:
    mi_scores = mutual_info_regression(X_mi, Y_mi, random_state=42)
mi_scores = mi_scores / (mi_scores.max() + 1e-8)

# 联合得分：Spearman + MI 归一化平均
joint_score = 0.5 * (spearman_scores / spearman_scores.max()) + \
              0.5 * mi_scores

# 排序并保留 top-N
rank_idx       = np.argsort(joint_score)[::-1]
selected_idx   = rank_idx[:N_FEAT_KEEP]
selected_names = [feature_names[i] for i in selected_idx]

print(f"\n  预筛选保留 {N_FEAT_KEEP}/{n_features} 个特征:")
for rank, fi in enumerate(selected_idx):
    print(f"    {rank+1:2d}. {feature_names[fi]:<25s}  "
          f"Spearman={spearman_scores[fi]:.3f}  MI={mi_scores[fi]:.3f}  "
          f"Joint={joint_score[fi]:.3f}")

# 保存筛选得分
df_score = pd.DataFrame({
    'Feature':       feature_names,
    'Spearman':      spearman_scores,
    'MI':            mi_scores,
    'Joint':         joint_score,
    'Selected':      [i in selected_idx for i in range(n_features)]
}).sort_values('Joint', ascending=False)
df_score.to_csv(os.path.join(SAVE_DIR, 'feature_screening_scores.csv'), index=False)

X_sel = X_full[:, selected_idx, :, :]   # [T, K, rows, cols]  K=N_FEAT_KEEP

# =============================================================================
# 2b. 全局标准差归一化
#     对每个特征，跨所有有效格点和时间步计算全局均值/标准差，
#     使不同量纲特征（温度/湿度/辐射等）在同一尺度上输入LSTM。
#     归一化在空间上统一，保留格点间相对差异。
# =============================================================================
print("\n[2b] 全局标准差归一化（Z-score）...")

# X_sel: [T, K, rows, cols] → 对每个特征(axis=0,2,3)计算均值和标准差
X_sel_mean = np.nanmean(X_sel, axis=(0, 2, 3), keepdims=True)  # [1, K, 1, 1]
X_sel_std  = np.nanstd( X_sel, axis=(0, 2, 3), keepdims=True)  # [1, K, 1, 1]
X_sel_std  = np.where(X_sel_std < 1e-8, 1.0, X_sel_std)        # 防止除以0

X_sel_norm = (X_sel - X_sel_mean) / X_sel_std                  # [T, K, rows, cols]

# ISOP 也做全局归一化（用于预筛选阶段的Spearman/MI一致性，train_pixel内部仍逐像元归一化）
ISOP_mean = np.nanmean(ISOP)
ISOP_std  = np.nanstd(ISOP)
ISOP_std  = max(ISOP_std, 1e-8)

# 打印归一化参数（供验证）
print(f"  特征全局归一化参数 [K={N_FEAT_KEEP}]:")
for ki, name in enumerate(selected_names):
    print(f"    {name:<25s}  mean={X_sel_mean[0,ki,0,0]:10.4f}  std={X_sel_std[0,ki,0,0]:10.4f}")
print(f"  ISOP: mean={ISOP_mean:.4f}  std={ISOP_std:.4f}")

# 保存归一化参数到CSV，方便后续反归一化
df_norm = pd.DataFrame({
    'Feature': selected_names,
    'Mean':    X_sel_mean[0, :, 0, 0],
    'Std':     X_sel_std[0,  :, 0, 0],
})
df_norm.to_csv(os.path.join(SAVE_DIR, 'feature_norm_params.csv'), index=False)
print("  归一化参数已保存: feature_norm_params.csv")
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


def train_pixel(x_ts, y_ts, seq_len=SEQ_LEN, hidden=HIDDEN_DIM,
                epochs=EPOCHS, lr=LR, patience=PATIENCE, device=DEVICE):
    """
    x_ts : [T, K]  该格点的特征时间序列
    y_ts : [T]     该格点的ISOP时间序列
    返回: model, scaler_x, scaler_y, train_r2
    """
    # ---- 形状标准化 ----
    x_ts = np.array(x_ts, dtype=np.float64)
    y_ts = np.array(y_ts, dtype=np.float64).ravel()   # 强制 [T]

    # 若传入 [K, T] 自动转置为 [T, K]
    if x_ts.ndim == 2 and x_ts.shape[0] != len(y_ts) and x_ts.shape[1] == len(y_ts):
        x_ts = x_ts.T
    # 形状仍不合法则跳过
    if x_ts.ndim != 2 or x_ts.shape[0] != len(y_ts):
        return None, StandardScaler(), StandardScaler(), {}

    K = x_ts.shape[1]

    # ---- 严格NaN/Inf清洗：逐时间步剔除任意特征或目标含异常值的行 ----
    valid_t = np.isfinite(x_ts).all(axis=1) & np.isfinite(y_ts)
    x_ts    = x_ts[valid_t].copy()
    y_ts    = y_ts[valid_t].copy()

    # 有效步数不足则跳过
    if len(y_ts) < seq_len + 10:
        return None, StandardScaler(), StandardScaler(), {}

    # 标准化
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_n  = sc_x.fit_transform(x_ts)
    Y_n  = sc_y.fit_transform(y_ts.reshape(-1, 1)).ravel()

    # 标准化后再次检查（防止常数列导致inf）
    X_n = np.nan_to_num(X_n, nan=0.0, posinf=0.0, neginf=0.0)
    Y_n = np.nan_to_num(Y_n, nan=0.0, posinf=0.0, neginf=0.0)

    # 构建序列
    Xs, Ys = make_sequences(X_n, Y_n, seq_len)
    if len(Xs) < 10:
        return None, sc_x, sc_y, {}  # 样本太少跳过

    # 8:2 时序分割
    split  = int(len(Xs) * 0.8)
    X_tr, X_va = Xs[:split], Xs[split:]
    Y_tr, Y_va = Ys[:split], Ys[split:]

    ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)

    model     = LightLSTM(K, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=10, factor=0.5)

    best_val, best_state, wait = np.inf, None, 0
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
            xv = torch.from_numpy(X_va).to(device)
            yv = torch.from_numpy(Y_va).to(device)
            val_loss = criterion(model(xv), yv).item()
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)

    # ---------- 全样本推理（原始尺度）----------
    model.eval()
    with torch.no_grad():
        xt     = torch.from_numpy(Xs).to(device)
        pred_n = model(xt).cpu().numpy()
    pred = sc_y.inverse_transform(pred_n.reshape(-1, 1)).ravel()
    true = sc_y.inverse_transform(Ys.reshape(-1, 1)).ravel()
    n    = len(true)

    # ---------- 计算多种拟合指标 ----------
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)

    r2   = 1 - ss_res / (ss_tot + 1e-8)                          # 决定系数 R²
    rmse = np.sqrt(ss_res / n)                                    # 均方根误差
    mae  = np.mean(np.abs(true - pred))                           # 平均绝对误差
    mbe  = np.mean(pred - true)                                   # 平均偏差（正=高估）

    # 相对误差（以观测均值为参考，避免除以0）
    obs_mean_px = np.mean(np.abs(true))
    rrmse = rmse / (obs_mean_px + 1e-8)                          # 相对RMSE（无量纲）
    nse   = 1 - ss_res / (ss_tot + 1e-8)                         # Nash-Sutcliffe效率（=R²对回归）

    # Pearson相关系数
    cov   = np.mean((pred - pred.mean()) * (true - true.mean()))
    corr  = cov / (pred.std() * true.std() + 1e-8)

    # KGE（Kling-Gupta效率）= 1 - sqrt((r-1)² + (α-1)² + (β-1)²)
    alpha = pred.std() / (true.std() + 1e-8)                     # 变异性比
    beta  = pred.mean() / (true.mean() + 1e-8)                   # 偏差比
    kge   = 1 - np.sqrt((corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    metrics = dict(r2=r2, rmse=rmse, mae=mae, mbe=mbe,
                   rrmse=rrmse, corr=corr, kge=kge)

    return model, sc_x, sc_y, metrics


# =============================================================================
# 5. 空间采样训练 + 三种重要性计算
# =============================================================================
print("\n[3/6] 空间采样训练 LSTM（用于重要性排序）...")

# 随机采样 SHAP_SAMPLE 个有效格点
# valid_mask 同时检查 ISOP 和 X_sel_norm 在所有时间步均有限
pix_flat   = np.arange(n_pixels)

# ISOP有效：[T,rows,cols] → all(axis=0) → [rows,cols]
isop_valid  = np.isfinite(ISOP).all(axis=0)                          # [rows, cols]
# X_sel_norm有效：[T,K,rows,cols] → all(axis=0,1) → [rows,cols]
xsel_valid  = np.isfinite(X_sel_norm).all(axis=0).all(axis=0)        # [rows, cols]
valid_mask  = (isop_valid & xsel_valid).ravel()                       # [rows*cols], C-order

valid_pix  = pix_flat[valid_mask]
sample_pix = rng.choice(valid_pix, min(SHAP_SAMPLE, len(valid_pix)), replace=False)
print(f"  有效格点数: {len(valid_pix)}/{n_pixels} ({100*len(valid_pix)/n_pixels:.1f}%)")

# 存储结果
imp_shap  = np.zeros((len(sample_pix), N_FEAT_KEEP))   # SHAP均值绝对值
imp_perm  = np.zeros((len(sample_pix), N_FEAT_KEEP))   # Permutation
imp_wt    = np.zeros((len(sample_pix), N_FEAT_KEEP))   # 输入层权重范数
r2_sample = np.zeros(len(sample_pix))

for si, pidx in enumerate(tqdm(sample_pix, desc="Pixel training")):
    ri, ci_ = divmod(int(pidx), cols)          # C-order: ri=行, ci_=列

    x_ts = X_sel_norm[:, :, ri, ci_]          # [T, K] 已全局归一化
    y_ts = ISOP[:, ri, ci_]                    # [T]    原始值，train_pixel内逐像元归一化

    # 已由valid_mask保证，此处做最后保险检查
    if not (np.isfinite(x_ts).all() and np.isfinite(y_ts).all()):
        r2_sample[si] = np.nan
        continue

    model, sc_x, sc_y, metrics = train_pixel(x_ts, y_ts)
    r2_sample[si] = metrics.get('r2', np.nan)
    if model is None:
        continue

    X_n  = sc_x.transform(x_ts).astype(np.float32)
    Y_n  = sc_y.transform(y_ts.reshape(-1, 1)).ravel().astype(np.float32)
    Xs, Ys = make_sequences(X_n, Y_n, SEQ_LEN)

    # ---- 5a. 输入层权重范数（快速粗排） ----
    w_ih = model.lstm.weight_ih_l0.detach().cpu().numpy()   # [4H, K]
    # 每个特征对应的列范数（4个门合并）
    wt_norm = np.linalg.norm(w_ih.reshape(4, HIDDEN_DIM, N_FEAT_KEEP), axis=(0, 1))
    imp_wt[si] = wt_norm / (wt_norm.sum() + 1e-8)

    # ---- 5b. Permutation Importance ----
    _perm_err_printed = False
    wrapper = PixelLSTMWrapper(model, SEQ_LEN, sc_x, sc_y, DEVICE)
    try:
        pi_res = permutation_importance(
            wrapper, X_n, y_ts,
            n_repeats=10, random_state=42, scoring='r2'
        )
        perm_v = np.maximum(pi_res.importances_mean, 0)
        imp_perm[si] = perm_v / (perm_v.sum() + 1e-8)
    except Exception as e:
        if si == 0:   # 只打印第一次错误，避免刷屏
            tqdm.write(f"  [Permutation 警告] {type(e).__name__}: {e}")
        imp_perm[si] = imp_wt[si]   # 兜底：用权重范数代替

    # ---- 5c. SHAP（DeepExplainer） ----
    try:
        if len(Xs) < 5:
            raise ValueError(f"序列数不足({len(Xs)}<5)，跳过SHAP")
        model.eval()
        bg = torch.from_numpy(Xs[:min(20, len(Xs))]).to(DEVICE)
        te = torch.from_numpy(Xs).to(DEVICE)
        explainer   = shap.DeepExplainer(model, bg)
        shap_values = explainer.shap_values(te)   # [n_seq, seq_len, K] 或 list

        # shap_values 可能是 list（多输出模型）或 ndarray
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.array(shap_values)       # 确保是 ndarray

        sv = np.abs(shap_values).mean(axis=(0, 1))  # [K]
        imp_shap[si] = sv / (sv.sum() + 1e-8)
    except Exception as e:
        if si == 0:
            tqdm.write(f"  [SHAP 警告] {type(e).__name__}: {e}")
        imp_shap[si] = imp_wt[si]   # 兜底：用权重范数代替

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
#    输出：三种重要性空间图、预测值、原始值、R²
# =============================================================================
print("\n[5/6] 全图逐像元训练（三种重要性 + 预测值）...")

# ---------- 结果容器 ----------
_nan2d = lambda: np.full((rows, cols), np.nan, dtype=np.float32)
R2_map    = _nan2d()   # 决定系数
RMSE_map  = _nan2d()   # 均方根误差
MAE_map   = _nan2d()   # 平均绝对误差
MBE_map   = _nan2d()   # 平均偏差
RRMSE_map = _nan2d()   # 相对RMSE
CORR_map  = _nan2d()   # Pearson相关系数
KGE_map   = _nan2d()   # Kling-Gupta效率

IMP_wt_map   = np.full((N_FEAT_KEEP, rows, cols), np.nan, dtype=np.float32)
IMP_perm_map = np.full((N_FEAT_KEEP, rows, cols), np.nan, dtype=np.float32)
IMP_shap_map = np.full((N_FEAT_KEEP, rows, cols), np.nan, dtype=np.float32)
IMP_ens_map  = np.full((N_FEAT_KEEP, rows, cols), np.nan, dtype=np.float32)
PRED_map     = np.full((T, rows, cols),           np.nan, dtype=np.float32)
OBS_map      = ISOP.astype(np.float32)

total_pix = len(valid_pix)
pbar = tqdm(total=total_pix, desc="Full spatial training")

for pidx in valid_pix:
    ri, ci_ = divmod(int(pidx), cols)

    x_ts = X_sel_norm[:, :, ri, ci_]          # [T, K] 已全局归一化
    y_ts = ISOP[:, ri, ci_]                    # [T]    原始值

    if not (np.isfinite(x_ts).all() and np.isfinite(y_ts).all()):
        pbar.update(1)
        continue

    model, sc_x, sc_y, metrics = train_pixel(x_ts, y_ts)
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

    # ---- 预测值重建（填回原始时间轴，前 SEQ_LEN 步为 NaN）----
    if len(Xs) > 0:
        model.eval()
        with torch.no_grad():
            xt   = torch.from_numpy(Xs).to(DEVICE)
            pred_n = model(xt).cpu().numpy()
        pred = sc_y.inverse_transform(pred_n.reshape(-1, 1)).ravel()
        PRED_map[SEQ_LEN:SEQ_LEN + len(pred), ri, ci_] = pred.astype(np.float32)

    # ---- 重要性1：输入层权重范数 ----
    w_ih  = model.lstm.weight_ih_l0.detach().cpu().numpy()   # [4H, K]
    wt_n  = np.linalg.norm(w_ih.reshape(4, HIDDEN_DIM, N_FEAT_KEEP), axis=(0, 1))
    wt_n  = wt_n / (wt_n.sum() + 1e-8)
    IMP_wt_map[:, ri, ci_] = wt_n.astype(np.float32)

    # ---- 重要性2：Permutation Importance ----
    try:
        wrapper = PixelLSTMWrapper(model, SEQ_LEN, sc_x, sc_y, DEVICE)
        pi_res  = permutation_importance(
            wrapper, X_n, y_ts, n_repeats=5, random_state=42, scoring='r2')
        perm_n  = np.maximum(pi_res.importances_mean, 0)
        perm_n  = perm_n / (perm_n.sum() + 1e-8)
        IMP_perm_map[:, ri, ci_] = perm_n.astype(np.float32)
    except Exception:
        IMP_perm_map[:, ri, ci_] = wt_n.astype(np.float32)  # 失败时用权重范数代替

    # ---- 重要性3：SHAP（DeepExplainer）----
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
            sv        = np.abs(shap_vals).mean(axis=(0, 1))
            sv        = sv / (sv.sum() + 1e-8)
            IMP_shap_map[:, ri, ci_] = sv.astype(np.float32)
        else:
            IMP_shap_map[:, ri, ci_] = wt_n.astype(np.float32)
    except Exception:
        IMP_shap_map[:, ri, ci_] = wt_n.astype(np.float32)

    # ---- 综合重要性（三方法等权平均，各自归一化后合并）----
    def _n01(v):
        v = v - v.min()
        return v / (v.max() + 1e-8)
    ens = (_n01(wt_n) + _n01(IMP_perm_map[:, ri, ci_]) +
           _n01(IMP_shap_map[:, ri, ci_])) / 3
    IMP_ens_map[:, ri, ci_] = ens.astype(np.float32)

    pbar.update(1)

pbar.close()

# ---------- 保存所有结果为统一 NetCDF 文件 ----------
print("  保存结果为 NetCDF 文件...")

import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 坐标轴
lon_1d_coord = lon[0, :].astype(np.float32)     # [cols]
lat_1d_coord = lat[:, 0].astype(np.float32)     # [rows]
time_coord   = np.arange(T, dtype=np.int32)     # 0 ~ T-1（月序号）
feat_coord   = np.array(selected_names)          # [K] 特征名

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
print(f"    IMP_weight      [feature={N_FEAT_KEEP}, lat={rows}, lon={cols}]")
print(f"    IMP_permutation [feature={N_FEAT_KEEP}, lat={rows}, lon={cols}]")
print(f"    IMP_shap        [feature={N_FEAT_KEEP}, lat={rows}, lon={cols}]")
print(f"    IMP_ensemble    [feature={N_FEAT_KEEP}, lat={rows}, lon={cols}]")
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

# ---- 图1：特征重要性综合排名柱状图（全局采样结果）----
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors_bar = plt.cm.RdYlBu_r(np.linspace(0.2, 0.85, N_FEAT_KEEP))
for ax, vals, title in zip(
    axes,
    [mean_shap_n[final_rank], mean_perm_n[final_rank], mean_wt_n[final_rank]],
    ['SHAP (采样全局)', 'Permutation (采样全局)', 'Weight Norm (采样全局)']
):
    names_r = [selected_names[i] for i in final_rank]
    ax.barh(range(N_FEAT_KEEP), vals[::-1], color=colors_bar)
    ax.set_yticks(range(N_FEAT_KEEP)); ax.set_yticklabels(names_r[::-1], fontsize=9)
    ax.set_xlabel('归一化重要性', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.invert_xaxis(); ax.grid(axis='x', alpha=0.3)
ax_ens = axes[2].twinx()
ax_ens.plot(ensemble_score[final_rank][::-1], range(N_FEAT_KEEP),
            'ko-', ms=5, lw=1.5, label='Ensemble')
ax_ens.set_ylabel('综合得分', fontsize=9)
plt.suptitle('ISOP 驱动因子重要性排序（采样500格点）',
             fontsize=14, fontweight='bold', y=1.02)
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
        ax.set_title(f'#{k+1} {selected_names[fi]}', fontsize=11, fontweight='bold')
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
fig, ax = plt.subplots(figsize=(10, 6))
df_s = df_score.copy()
colors_sel = ['#d62728' if s else '#aec7e8' for s in df_s['Selected']]
ax.barh(df_s['Feature'], df_s['Joint'], color=colors_sel)
ax.axvline(df_s[df_s['Selected']]['Joint'].min(), color='r',
           linestyle='--', lw=1.5, label=f'筛选阈值 (Top-{N_FEAT_KEEP})')
ax.set_xlabel('联合筛选得分 (Spearman + MI)', fontsize=11)
ax.set_title('特征预筛选得分（红色=入选）', fontsize=13, fontweight='bold')
ax.legend(); ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'feature_screening.png'),
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
print("  feature_norm_params.csv         — 特征全局归一化参数")
print("  feature_screening_scores.csv    — 预筛选得分表")
print("  feature_importance_ranking.csv  — 三方法重要性排名表")
print("  metrics_spatial.png             — 7项拟合指标空间分布图")
print("  R2_spatial.png                  — R²空间分布图")
print("  IMP_wt/perm/shap/ens_top6.png   — 各方法Top-6重要性空间图")
print("  pred_vs_obs_spatial.png         — 预测/原始/偏差空间对比图")
print("  pred_vs_obs_timeseries.png      — 全局时序对比图")

# 程序结束后恢复系统电源设置
reset_windows_power_settings(kernel32_obj)