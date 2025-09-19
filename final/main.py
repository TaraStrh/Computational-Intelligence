import pandas as pd
from river import utils
import os
def save_model(model, path):
    utils.dump(model, path)
    print(f"Model saved to → {path}")
def load_model(path):
    if os.path.exists(path):
        model = utils.load(path)
        print(f"Model loaded from → {path}")
        return model
    return None
df = pd.read_csv( 'cluster-trace-gpu-v2023/csv/openb_pod_list_cpu100.csv')
import os
import math
import numpy as np
import pandas as pd
from collections import deque

from river import (
    ensemble, tree, preprocessing as pp, stream, metrics,
    linear_model, neighbors, optim, utils
)


# Save / Load helpers
def save_model(model, path):
    utils.dump(model, path)
    print(f"Model saved to → {path}")

def load_model(path):
    if os.path.exists(path):
        model = utils.load(path)
        print(f"Model loaded from → {path}")
        return model
    return None


# Data
CSV_PATH = 'cluster-trace-gpu-v2023/csv/openb_pod_list_cpu100.csv'
df = pd.read_csv(CSV_PATH)

# Feature builder
def build_features(df: pd.DataFrame):
    df = df.copy()

    num_cols = ['cpu_milli','memory_mib','num_gpu','gpu_milli',
                'creation_time','deletion_time','scheduled_time']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = 0.0

    y = df['cpu_milli'].fillna(0).astype(float)

    X = pd.DataFrame(index=df.index)
    X['mem_gib'] = (df['memory_mib'].fillna(0) / 1024.0).astype(float)
    X['num_gpu'] = df['num_gpu'].fillna(0).astype(float)
    X['gpu_milli'] = df['gpu_milli'].fillna(0).astype(float)

    ct = df['creation_time'].fillna(0).astype(float)
    dt = df['deletion_time'].fillna(0).astype(float)
    st = df['scheduled_time'].fillna(0).astype(float)
    X['lifetime']   = (dt - ct).clip(lower=0.0)
    X['sched_delay'] = (st - ct).clip(lower=0.0)

    X['has_gpu'] = (X['num_gpu'] > 0).astype(float)

    qos = df.get('qos', '').astype(str).str.upper()
    X['qos_LS'] = (qos == 'LS').astype(float)
    X['qos_BE'] = (qos == 'BE').astype(float)

    phase = df.get('pod_phase', '').astype(str).str.title()
    X['ph_Running']   = (phase == 'Running').astype(float)
    X['ph_Succeeded'] = (phase == 'Succeeded').astype(float)
    X['ph_Failed']    = (phase == 'Failed').astype(float)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    features = list(X.columns)
    return X, y, features

X, y, features = build_features(df)

# Models (experts)
def make_bag_ht():
    return ensemble.BaggingRegressor(
        model=tree.HoeffdingTreeRegressor(
            grace_period=100,
            max_depth=20,
            leaf_prediction='mean'
        ),
        n_models=20,
        seed=42
    )

def make_hat():
    return tree.HoeffdingAdaptiveTreeRegressor(
        grace_period=100,
        leaf_prediction='mean'
    )

def make_lin():
    return pp.StandardScaler() | linear_model.PARegressor()

def make_knn():
    engine = neighbors.LazySearch(
        window_size=3000, 
        min_distance_keep=0.0  
    )
    return pp.StandardScaler() | neighbors.KNNRegressor(
        n_neighbors=30,
        engine=engine,
        aggregation_method="mean"  
    )

experts = [make_bag_ht(), make_hat(), make_lin(), make_knn()]

# EWA over experts
ewa = ensemble.EWARegressor(models=experts, loss=optim.losses.Squared(), learning_rate=2e-3)

# Metrics
mae = metrics.MAE()

class OnlineMAPE:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.n = 0
        self.sum_pe = 0.0
    def update(self, y_true, y_pred):
        y_true = float(y_true); y_pred = float(y_pred)
        pe = abs(y_true - y_pred) / max(self.eps, abs(y_true))
        self.sum_pe += pe
        self.n += 1
    def get(self):
        return (self.sum_pe / max(1, self.n)) * 100.0

class OnlineWAPE:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.sum_abs_err = 0.0
        self.sum_abs_true = 0.0
    def update(self, y_true, y_pred):
        self.sum_abs_err += abs(float(y_true) - float(y_pred))
        self.sum_abs_true += abs(float(y_true))
    def get(self):
        return 100.0 * self.sum_abs_err / max(self.eps, self.sum_abs_true)

mape, wape = OnlineMAPE(), OnlineWAPE()

# Scaling and loop settings
TARGET_SCALE = float(np.percentile(np.abs(y.values), 75))
TARGET_SCALE = max(1.0, TARGET_SCALE)

last_preds = deque(maxlen=40)
WARM_UP = 200
SEEN = 0

NODE_CAPACITY = 8000  
sla_violations = 0
overprov = 0.0
scaling_actions = 0
current_nodes = 1

# Online loop
for x_row, y_true in stream.iter_pandas(X[features], y):
    SEEN += 1

    # Predict
    if SEEN <= WARM_UP:
        preds = [(m.predict_one(x_row) or 0.0) for m in experts]
        y_hat_scaled = float(np.mean(preds)) if preds else 0.0
    else:
        y_hat_scaled = ewa.predict_one(x_row) or 0.0

    # Rescale to original target and clip negatives
    y_hat = max(0.0, y_hat_scaled * TARGET_SCALE)

    # Metrics
    mae.update(y_true, y_hat)
    mape.update(y_true, y_hat)
    wape.update(y_true, y_hat)

    # Autoscaling policy
    needed_nodes = max(1, int(np.ceil(y_hat / NODE_CAPACITY)))
    if needed_nodes * NODE_CAPACITY < float(y_true):
        sla_violations += 1
    overprov += max(0.0, needed_nodes * NODE_CAPACITY - float(y_true))
    if needed_nodes != current_nodes:
        scaling_actions += 1
        current_nodes = needed_nodes

    # Train (scale target for stability)
    y_scaled = float(y_true) / TARGET_SCALE
    for m in experts:
        m.learn_one(x_row, y_scaled)
    if SEEN > WARM_UP:
        ewa.learn_one(x_row, y_scaled)

    last_preds.append((float(y_true), float(y_hat)))

# Report
print("Last 40 predictions (EWA over heterogeneous experts, scaled):")
for i, (yt, yh) in enumerate(last_preds, 1):
    print(f"{i:02d}  Actual={yt:8.2f}  Predicted={yh:8.2f}")

print(f"\nMAE={mae.get():.2f}")
print(f"MAPE={mape.get():.2f}%")
print(f"WAPE={wape.get():.2f}%")

if hasattr(ewa, "weights"):
    print("\nFinal EWA weights:")
    for name, w in zip(["bag_ht","hat","lin","knn"], ewa.weights):
        print(f"  {name:8s}: {w:.3f}")

total_steps = max(1, SEEN)
sla_violation_rate = sla_violations / total_steps
print(f"\nSLA violations: {sla_violations}  (rate = {sla_violation_rate:.3f})")
print(f"Over-provisioned capacity (mCPU): {overprov:.2f}")
print(f"Scaling actions taken: {scaling_actions}")
