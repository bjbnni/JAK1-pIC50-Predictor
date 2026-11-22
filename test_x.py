import os
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# =============================================================
# 1. Molecular Graph Construction & Hybrid GAT Architecture
# =============================================================
def smiles_to_graph_data(smiles, extra_features, target_pic50):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    atom_features = [
        [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs()
        ]
        for atom in mol.GetAtoms()
    ]
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_indices, bond_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        b_feats = [
            float(bt == Chem.rdchem.BondType.SINGLE),
            float(bt == Chem.rdchem.BondType.DOUBLE),
            float(bt == Chem.rdchem.BondType.TRIPLE),
            float(bt == Chem.rdchem.BondType.AROMATIC),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing())
        ]
        edge_indices.extend([[i, j], [j, i]])
        bond_features.extend([b_feats, b_feats])

    if len(edge_indices) == 0:
        return None

    return Data(
        x=x,
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(bond_features, dtype=torch.float),
        extra_x=torch.tensor(extra_features, dtype=torch.float),
        y=torch.tensor([target_pic50], dtype=torch.float)
    )


class JAK1Dataset(Dataset):
    def __init__(self, df, feature_cols, smiles_col='clean_smiles', target_col='pIC50'):
        super().__init__()
        self.data_list = []
        for _, row in df.iterrows():
            graph_data = smiles_to_graph_data(
                row[smiles_col],
                row[feature_cols].values.astype(float),
                float(row[target_col])
            )
            if graph_data is not None:
                self.data_list.append(graph_data)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class AdvancedHybridGAT(nn.Module):
    def __init__(self, node_in_dim=6, edge_in_dim=6, extra_in_dim=1301, hidden_dim=128, heads=4):
        super().__init__()
        self.extra_in_dim = extra_in_dim
        self.gat1 = GATConv(node_in_dim, hidden_dim, heads=heads, concat=False, edge_dim=edge_in_dim)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=edge_in_dim)
        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, edge_dim=edge_in_dim)

        self.fc_extra = nn.Sequential(
            nn.Linear(extra_in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        extra_x = data.extra_x.view(data.num_graphs, self.extra_in_dim)

        h_graph = F.elu(self.gat1(x, edge_index, edge_attr=edge_attr))
        h_graph = F.elu(self.gat2(h_graph, edge_index, edge_attr=edge_attr))
        h_graph = F.elu(self.gat3(h_graph, edge_index, edge_attr=edge_attr))
        h_graph = global_mean_pool(h_graph, batch)

        h_extra = self.fc_extra(extra_x)
        out = self.fc_out(torch.cat([h_graph, h_extra], dim=1))
        return out.squeeze(-1)


# =============================================================
# 2. Statistical Evaluation & Cached Inference Helpers
# =============================================================
def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    p_r = 0.0 if np.std(y_pred) == 0 else pearsonr(y_true, y_pred)[0]
    s_rho = 0.0 if np.std(y_pred) == 0 else spearmanr(y_true, y_pred)[0]
    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Pearson_r": p_r,
        "Spearman_rho": s_rho
    }


def print_metrics_summary(title, metrics):
    print("\n" + "=" * 50)
    print(f"   {title}")
    print("=" * 50)
    print(f"• R-Squared (R2 Score)        : {metrics['R2']:.4f}")
    print(f"• Root Mean Squared Error     : {metrics['RMSE']:.4f}")
    print(f"• Mean Absolute Error (MAE)   : {metrics['MAE']:.4f}")
    print(f"• Pearson Correlation (r)     : {metrics['Pearson_r']:.4f}")
    print(f"• Spearman Correlation (rho)  : {metrics['Spearman_rho']:.4f}")
    print("=" * 50)


@torch.no_grad()
def extract_cached_embeddings(model, loader, device):
    model.eval()
    h_graphs, extra_xs, y_trues = [], [], []
    for batch in loader:
        batch = batch.to(device)
        hg = F.elu(model.gat1(batch.x, batch.edge_index, edge_attr=batch.edge_attr))
        hg = F.elu(model.gat2(hg, batch.edge_index, edge_attr=batch.edge_attr))
        hg = F.elu(model.gat3(hg, batch.edge_index, edge_attr=batch.edge_attr))
        hg = global_mean_pool(hg, batch.batch)

        h_graphs.append(hg)
        extra_xs.append(batch.extra_x.view(batch.num_graphs, model.extra_in_dim))
        y_trues.append(batch.y.cpu().numpy())

    return torch.cat(h_graphs, dim=0), torch.cat(extra_xs, dim=0), np.concatenate(y_trues)


@torch.no_grad()
def predict_from_cached(model, h_graph, x_extra):
    model.eval()
    h_extra = model.fc_extra(x_extra)
    out = model.fc_out(torch.cat([h_graph, h_extra], dim=1)).squeeze(-1)
    return out.cpu().numpy()


# =============================================================
# 3. Main Execution: X-Scrambling & Permutation Importance
# =============================================================
if __name__ == '__main__':
    DATA_FILE = "7000.xlsx"
    MODEL_PATH = "jak1_gat_7000_best.pth"
    IMPORTANCE_FILE = "best.vv.csv"
    OUTPUT_CSV = "x_scrambling_7000_1301_report.csv"

    if not os.path.exists(IMPORTANCE_FILE):
        raise FileNotFoundError(f"Required feature file '{IMPORTANCE_FILE}' not found in the working directory.")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset file '{DATA_FILE}' not found. Required to run full validation.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation Device : {device}")
    print(f"Feature Reference  : {IMPORTANCE_FILE}")

    # 1) Load selected 1,301 features
    imp_df = pd.read_csv(IMPORTANCE_FILE)
    feature_cols = imp_df[imp_df['Importance_R2_Drop'] > 0]['Feature'].tolist()
    print(f"Selected Features  : {len(feature_cols)}")

    # 2) Replicate exact train/val/test split (random_state=42)
    df_7k = pd.read_excel(DATA_FILE)
    train_df, temp_df = train_test_split(df_7k, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    val_loader = DataLoader(JAK1Dataset(val_df, feature_cols), batch_size=64, shuffle=False)
    test_loader = DataLoader(JAK1Dataset(test_df, feature_cols), batch_size=64, shuffle=False)

    # 3) Load pre-trained model weights
    model = AdvancedHybridGAT(extra_in_dim=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # ---------------------------------------------------------
    # Step 1: Baseline Evaluation on Held-Out Test Set
    # ---------------------------------------------------------
    H_test, X_test, y_test = extract_cached_embeddings(model, test_loader, device)
    base_test_metrics = compute_metrics(y_test, predict_from_cached(model, H_test, X_test))
    print_metrics_summary("1. BASELINE PERFORMANCE ON HELD-OUT TEST SET", base_test_metrics)

    # ---------------------------------------------------------
    # Step 2: Global X-Scrambling (Randomization Test, 30 runs)
    # ---------------------------------------------------------
    rng = np.random.default_rng(42)
    full_scramble_list = []
    for _ in range(30):
        X_shuf = X_test.clone()
        for col in range(X_shuf.shape[1]):
            perm = torch.from_numpy(rng.permutation(X_shuf.shape[0])).to(device)
            X_shuf[:, col] = X_shuf[perm, col]
        preds = predict_from_cached(model, H_test, X_shuf)
        full_scramble_list.append(compute_metrics(y_test, preds))

    avg_full_scramble = {k: np.mean([m[k] for m in full_scramble_list]) for k in base_test_metrics}
    print_metrics_summary("2. GLOBAL X-SCRAMBLING CONTROL (30 ITERATIONS)", avg_full_scramble)

    # ---------------------------------------------------------
    # Step 3: Individual Permutation Feature Importance (Val Set)
    # ---------------------------------------------------------
    print(f"\nEvaluating individual permutation importance across {len(feature_cols)} features...")
    H_val, X_val, y_val = extract_cached_embeddings(model, val_loader, device)
    base_val_metrics = compute_metrics(y_val, predict_from_cached(model, H_val, X_val))

    results = []
    n_repeats = 10

    for idx, feat_name in enumerate(feature_cols):
        repeat_metrics = []
        for _ in range(n_repeats):
            X_mod = X_val.clone()
            perm = torch.from_numpy(rng.permutation(X_mod.shape[0])).to(device)
            X_mod[:, idx] = X_mod[perm, idx]
            preds = predict_from_cached(model, H_val, X_mod)
            repeat_metrics.append(compute_metrics(y_val, preds))

        avg_m = {k: np.mean([m[k] for m in repeat_metrics]) for k in base_val_metrics}
        r2_drop = base_val_metrics["R2"] - avg_m["R2"]

        results.append({
            "Feature": feat_name,
            "Importance_R2_Drop": r2_drop,
            "Scrambled_R2": avg_m["R2"],
            "Scrambled_RMSE": avg_m["RMSE"],
            "Scrambled_MAE": avg_m["MAE"],
            "Scrambled_Pearson_r": avg_m["Pearson_r"],
            "Scrambled_Spearman_rho": avg_m["Spearman_rho"],
            "Status": "Retain (Informative)" if r2_drop > 0 else "Prune (Noise/Redundant)"
        })

    res_df = pd.DataFrame(results).sort_values(by="Importance_R2_Drop", ascending=False)
    res_df.to_csv(OUTPUT_CSV, index=False)

    useful_count = (res_df["Importance_R2_Drop"] > 0).sum()
    harmful_count = (res_df["Importance_R2_Drop"] <= 0).sum()

    print(f"\nReport successfully exported to: {OUTPUT_CSV}")
    print(f"Informative Features (R2_Drop > 0) : {useful_count}")
    print(f"Noise/Redundant Features (<= 0)    : {harmful_count}")

    print("\n--- Top 10 Most Influential Molecular Descriptors ---")
    print(res_df[["Feature", "Importance_R2_Drop", "Scrambled_R2", "Scrambled_RMSE"]].head(10).to_string(index=False))

    print("\n--- Bottom 10 Features (Candidates for Pruning) ---")
    print(res_df[["Feature", "Importance_R2_Drop", "Scrambled_R2", "Scrambled_RMSE"]].tail(10).to_string(index=False))