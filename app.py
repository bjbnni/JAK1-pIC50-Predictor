import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
import joblib
import io

# ==========================================
# 1. Page Configuration & Academic Header
# ==========================================
st.set_page_config(
    page_title="JAK1 Activity Predictor | Hybrid GAT",
    layout="wide",
    page_icon="🧪",
    initial_sidebar_state="expanded"
)

# Sidebar for Research Context & Architecture Overview
with st.sidebar:
    st.header("🔬 Model Architecture")
    st.info("""
    **AdvancedHybridGAT**
    - **Graph Branch:** 3x Multi-head Graph Attention Layers (`GATConv`) capturing atom/bond topology.
    - **Tabular Branch:** MLP processing 1,301 selected Morgan Fingerprints & 2D RDKit Descriptors.
    - **Validation Performance:** $R^2 = 0.8701$ on held-out test set.
    """)
    st.markdown("---")
    st.caption("Developed for Computational Medicinal Chemistry & QSAR Screening.")

st.title("🧪 High-Throughput JAK1 Inhibitor Activity Predictor")
st.markdown(
    "Predict **Janus Kinase 1 (JAK1)** inhibitory potency ($\\text{pIC}_{50}$) using a Hybrid Graph Attention Network. "
    "Enter single or batch SMILES strings below to visualize structures, compute molecular descriptors, and export results."
)
st.divider()

# ==========================================
# 2. Hybrid GAT Model Definition
# ==========================================
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

# ==========================================
# 3. Cached Asset Loading
# ==========================================
@st.cache_resource
def load_assets():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imp_df = pd.read_csv("best.vv.csv")
    feature_cols = imp_df[imp_df['Importance_R2_Drop'] > 0]['Feature'].tolist()
    scaler = joblib.load("my_scaler.pkl")
    
    model = AdvancedHybridGAT(extra_in_dim=len(feature_cols)).to(device)
    model.load_state_dict(torch.load("jak1_gat_7000_best.pth", map_location=device))
    model.eval()
    
    return model, scaler, feature_cols, device

try:
    model, scaler, feature_cols, device = load_assets()
except Exception as e:
    st.error("⚠️ Error loading model artifacts. Ensure `my_scaler.pkl`, `best.vv.csv`, and `jak1_gat_7000_best.pth` are present in the root directory.")
    st.stop()

# ==========================================
# 4. Feature Extraction & Graph Construction
# ==========================================
def process_single_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None, None, None
    
    # Construct Molecular Graph
    atom_features = [
        [atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
         int(atom.GetHybridization()), int(atom.GetIsAromatic()), atom.GetTotalNumHs()]
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
    
    graph_data = Data(
        x=x,
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(bond_features, dtype=torch.float)
    )
    
    # Extract 1,301 Selected Molecular Descriptors & Fingerprints
    features_dict = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    for i in range(2048):
        features_dict[f"Bit_{i}"] = int(fp[i])
    for desc_name, desc_func in Descriptors.descList:
        try:
            features_dict[desc_name] = desc_func(mol)
        except Exception:
            pass
    
    new_features = [features_dict.get(col, 0.0) for col in feature_cols]
    return graph_data, new_features, mol

# ==========================================
# 5. User Input & Inference Pipeline
# ==========================================
default_input = (
    "C[C@@H]1CCN(C[C@@H]1N(C)C2=NC=NC3=C2C=CN3)C(=O)CC#N\n"
    "FC1=CC=C(CN2CCC(NC3=C4C=CN=C4NC=C3C5=NNC=C5)CC2)C=C1"
)

user_smiles_text = st.text_area(
    "Input SMILES Strings (one compound per line):",
    default_input,
    height=140,
    help="First default entry corresponds to Tofacitinib (reference JAK inhibitor)."
)

if st.button("🚀 Run Prediction & Feature Extraction", type="primary"):
    smiles_list = [s.strip() for s in user_smiles_text.split('\n') if s.strip()]
    
    if not smiles_list:
        st.warning("Please enter at least one valid SMILES string.")
    else:
        results = []
        valid_mols = []
        
        with st.spinner(f"Processing {len(smiles_list)} compound(s) and extracting 1,301 descriptors..."):
            for idx, sm in enumerate(smiles_list):
                graph_base, raw_feats, mol_obj = process_single_smiles(sm)
                
                if graph_base is None:
                    results.append({
                        "Compound_ID": f"Cmpd_{idx+1}",
                        "SMILES": sm,
                        "Predicted_pIC50": "Invalid SMILES",
                        **{col: np.nan for col in feature_cols}
                    })
                    continue
                
                # Standardize features & run GAT inference
                scaled_features = scaler.transform([raw_feats])
                graph_base.extra_x = torch.tensor(scaled_features[0], dtype=torch.float)
                graph_batch = Batch.from_data_list([graph_base]).to(device)
                
                with torch.no_grad():
                    prediction = model(graph_batch).item()
                
                pred_val = round(prediction, 4)
                valid_mols.append((f"Cmpd_{idx+1}", mol_obj, pred_val))
                
                row_data = {
                    "Compound_ID": f"Cmpd_{idx+1}",
                    "SMILES": sm,
                    "Predicted_pIC50": pred_val
                }
                row_data.update({col: raw_feats[i] for i, col in enumerate(feature_cols)})
                results.append(row_data)
        
        results_df = pd.DataFrame(results)
        st.success(f"✅ Successfully processed {len(smiles_list)} compound(s)!")
        
        # Display Prediction Summary Table
        st.subheader("📊 Prediction Summary")
        st.dataframe(results_df[["Compound_ID", "SMILES", "Predicted_pIC50"]], use_container_width=True)
        
        # Display 2D Structures for up to 4 compounds for clean visual inspection
        if valid_mols:
            st.subheader("🧬 2D Chemical Structures (Preview)")
            preview_mols = valid_mols[:4]
            cols = st.columns(len(preview_mols))
            for col, (cid, m_obj, p_val) in zip(cols, preview_mols):
                with col:
                    img = Draw.MolToImage(m_obj, size=(250, 250))
                    st.image(img, caption=f"{cid} | pIC50: {p_val}", use_container_width=True)
            if len(valid_mols) > 4:
                st.caption(f"Showing first 4 structures out of {len(valid_mols)} valid compounds.")
        
        # Export Full Dataset (Predictions + 1,301 Features)
        st.divider()
        csv_buffer = io.BytesIO()
        results_df.to_csv(csv_buffer, index=False, encoding='utf-8')
        
        st.download_button(
            label="⬇️ Download Full Results & 1,301 Features (CSV)",
            data=csv_buffer.getvalue(),
            file_name="jak1_predictions_and_features.csv",
            mime="text/csv",
            type="primary"
        )