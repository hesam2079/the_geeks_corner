	Jingxin Liu ...
	42nd International Conference on Machine Learning

## Abstract
This paper introduces a novel **Federated Node-level Clustering Network (FedNCN)** to address the problem of missing cross-subgraph links in federated graph learning under unsupervised settings. The proposed method leverages clustering prior knowledge to mend destroyed inter-client links, thereby improving the quality of node-level clustering across distributed subgraphs. FedNCN consists of three key components: **local model learning**, **cross-subgraph link mending**, and **global knowledge sharing**. Experimental results on five benchmark datasets demonstrate the superiority of FedNCN over existing methods.

---

## Conclusion
The paper proposes **FedNCN**, the first attempt to restore missing cross-subgraph links in federated node-level clustering without relying on labeled data. By integrating local clustering signals, global link restoration, and consensus prototype learning, FedNCN achieves superior performance in unsupervised federated graph clustering. Future work includes extending the framework to handle incomplete graphs and multi-view clustering scenarios.

---

## Literature Review of Key Terms

### 1. MLP-GNN
A hybrid architecture combining **Multi-Layer Perceptron (MLP)** and **Graph Neural Network (GNN)**, where MLP is used for projecting or denoising node features, while GNN captures structural information from the graph.

### 2. Cross-Subgraph Link Mending (CLM)
A strategy that restores missing edges between subgraphs using clustering signals and graph similarity estimation.

### 3. Global Knowledge Sharing (GKS)
A mechanism where a central server learns consensus prototypes from a globally mended graph and shares them back with clients to improve local models.

### 4. N-Cut Criterion
A graph partitioning technique used to minimize redundant connections in the restored graph by optimizing normalized cut values.

### 5. Prototype Learning
A method that extracts representative nodes (prototypes) within each cluster to guide clustering and representation learning.

---

## Main Idea

FedNCN addresses the problem of **unsupervised node-level clustering in a federated setting**, where:
- A complete graph is partitioned into multiple subgraphs.
- Each subgraph resides on a client and cannot be shared due to privacy constraints.
- Missing cross-subgraph links degrade clustering performance.

The core idea is to:
1. **Extract clustering signals locally** using an MLP-based projector.
2. **Upload hard-to-reconstruct signals** to the server.
3. **Restore missing links** between subgraphs using graph similarity and N-Cut minimization.
4. **Generate consensus prototypes** via a GNN-based generator.
5. **Share updated prototypes** with clients to refine local models.

---

## Methodology

### 1. Local Model Learning
Each client trains a GNN to generate latent representations $Z \in \mathbb{R}^{N \times d'}$ using:
$$
Z^{(l)} = \sigma(\tilde{A} Z^{(l-1)} W^{(l)})
$$
where $\tilde{A}$ is the normalized adjacency matrix.

K-means is applied to extract prototypes $C \in \mathbb{R}^{O \times d'}$, and top-$k$ representative nodes are selected based on Euclidean distance:
$$
f_d(z_i, c_j) = \|z_i - c_j\|^2
$$

An MLP projector reconstructs a masked version $R$ of the attribute matrix $B$:
$$
\hat{R} = \text{MLP}(R; \theta_{\text{mlp}})
$$
with loss:
$$
L_{\text{mlp}} = \frac{1}{nkO} \|B - \hat{R}\|^2
$$

### 2. Cross-Subgraph Link Mending
On the server:
- Infer global samples $ \tilde{R} $ using uploaded $ R $ and $ \theta_{\text{mlp}} $.
- Compute intra-cluster similarity using cosine similarity:
$$
\text{Sim}(r_i, r_j) = \frac{r_i^\top r_j}{\|r_i\| \cdot \|r_j\|}
$$
- Construct refined subgraphs $G_{\text{sub}}$.
- Estimate graph-level affinity matrix $S$ using graph kernels (e.g., COS, WL):
$$
s_{ij} = \text{GK}(G_i, G_j)
$$
- Build global graph $G_\phi$ and refine it using an improved N-Cut criterion:
$$
\text{Con}(a_{ij}) = \frac{a_{ij}}{f_a(V_1, V)} + \frac{a_{ij}}{f_a(V_2, V)}
$$

### 3. Global Knowledge Sharing
A GNN-based generator learns consensus prototypes $ \tilde{C} $ and parameters $ \theta_{\text{gnn}} $ from the mended graph $G_\psi$. The objective function combines reconstruction loss and KL divergence:
$$
L = L_{\text{MSE}} + L_{\text{KL}}
$$
where:
$$
L_{\text{MSE}} = \frac{1}{N_o} \|X_\psi - \hat{X}_\psi\|^2
$$
$$
L_{\text{KL}} = \text{KL}(P || Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

---

## Novelty

1. **First Work on Unsupervised Federated Node-Level Clustering**
   - Addresses the issue of missing cross-subgraph links in federated learning without supervision.

2. **Cross-Subgraph Link Mending Strategy**
   - Uses clustering signals and graph kernel methods to restore missing inter-client links.

3. **MLP-GNN Joint Optimization Framework**
   - Combines MLP for signal projection and GNN for structure learning to enhance clustering performance.

4. **Improved N-Cut Criterion**
   - Reduces redundancy in the mended graph for better clustering.

5. **Privacy-Preserving Communication**
   - Uploads only hard-to-reconstruct signals and model parameters, not raw data.

---

## Algorithms

### Algorithm 1: Training Procedure of FedNCN
```python
Initialize all clients' prototypes C^(0), θ_mlp^(0), θ_gnn^(0)
for i = 1 to E_server do
    // Local Model Learning
    for j = 1 to E_client do
        Generate Z~ using Eq.(1)
    end
    Obtain C~ from Z~
    Obtain B using Eq.(2)
    Generate R using noise
    Train MLP with Eq.(3)-(4)
    Upload R and θ_mlp to server
    // Cross-Subgraph Link Mending
    Infer R~ using Eq.(5)
    Generate G_sub using Eq.(6)-(7)
    Generate global graph G_phi using Eq.(8)
    Refine G_phi to get G_psi using Eq.(9)
    // Global Knowledge Sharing
    Train GNN using Eq.(10)-(12)
    Backpropagate C~ and θ_gnn to clients
end
Obtain Y via K-means at clients
```

### Algorithm 2: Client-Side Processing
```python
Initialize θ_mlp, θ_gnn from server
for j = 1 to E_gnn do
    Generate Z~ using GNN
end
Obtain C~ via K-means
Generate B and R
for j = 1 to E_mlp do
    Reconstruct R^ using MLP
    Optimize MLP via Eq.(4)
end
Upload R and θ_mlp
```

### Algorithm 3: Server-Side Processing
```python
Infer R~ from R and θ_mlp
Generate G_sub using Eq.(6)-(7)
Compute S using Eq.(8)
Generate G_phi
Refine G_phi to get G_psi using Eq.(9)
for i = 1 to E_gnn do
    Compute LMSE and LKL
    Optimize GNN via Eq.(10)
end
Share C~ and θ_gnn
```

---