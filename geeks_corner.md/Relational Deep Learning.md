	Jure Leskovec
	04 DEC 2023 on arxiv

Understood. Here is the previous message with equations formatted using `$equation$` and **no spaces** between `$` and `equation`. 

---

### **1. Abstract**

Relational Deep Learning (RDL) introduces an end-to-end deep representation learning approach that directly learns from relational databases organized into multiple tables connected by primary-foreign key relations. Unlike traditional methods requiring manual feature engineering, RDL utilizes a temporal, heterogeneous graph representation of the database where nodes represent rows and edges represent foreign key links. Graph Neural Networks (GNNs) are then used to automatically learn representations across the graph, enabling more accurate and faster model building. To support research in this area, the paper presents RELBENCH, a set of benchmark datasets and implementations for evaluating GNNs on relational data.

---

### **2. Conclusion**

The paper concludes that Relational Deep Learning has the potential to revolutionize how machine learning is applied to relational databases, which store a large portion of the world's data. By treating relational databases as heterogeneous graphs and applying GNNs, RDL can leverage rich predictive signals without the need for manual feature engineering. The authors emphasize that this approach opens up new opportunities for scalable and effective deep learning models on relational data. They also highlight the importance of developing further methodologies, architectures, and benchmarks to fully realize the potential of RDL.

---

### **3. Definitions and Literature Review on Key Terms**

#### **Relational Database**
A structured collection of tables with rows representing entities and columns representing attributes, interconnected via primary-foreign key relationships.

#### **Graph Neural Networks (GNNs)**
Neural networks designed to process graph-structured data, performing message passing between nodes to learn node embeddings.

#### **Message Passing**
A mechanism in GNNs where each node aggregates information from its neighbors iteratively, updating its representation based on local graph structure.

#### **Relational Entity Graph**
A heterogeneous graph representation of a relational database where each row in a table corresponds to a node, and edges represent foreign key relationships.

#### **Feature Engineering**
Manual selection and transformation of features from raw data to improve model performance, often time-consuming and error-prone.

#### **Temporal Message Passing**
A variant of message passing in GNNs that respects the temporal ordering of events, ensuring that only past information is used when predicting future outcomes.

#### **Heterogeneous Graph**
A graph with multiple types of nodes and edges, reflecting different entity types and relationships present in complex data like relational databases.

#### **Statistical Relational Learning**
A field combining probabilistic reasoning with first-order logic to model uncertainty in relational data.

#### **Knowledge Graph Embedding**
Techniques that map entities and relations in knowledge graphs into continuous vector spaces while preserving relational semantics.

#### **Shortcut Learning**
A phenomenon in deep learning where models exploit superficial patterns in training data rather than learning robust, generalizable features.

---

### **4. Mathematics Needed to Understand the Concepts**

#### **Graph Representation**
A relational database is modeled as a **graph**$G=(V,E)$, where:
- $V$: Set of nodes representing database entities (rows).
- $E \subseteq V \times V$: Set of edges representing foreign key relationships.

Each node$v \in V$has:
- A type mapping$\phi(v) \in T$, where$T$is the schema graph.
- A timestamp$\tau(v)$indicating the time the row was created.
- Initial features$x_v$extracted using encoders.

Each edge$e = (v_1, v_2) \in E$has:
- An edge type$\psi(e) \in R$, derived from the schema graph.

---

#### **Message Passing in GNNs**

The general form of message passing in GNNs is:

$$
h_v^{(l+1)}=f\left(h_v^{(l)},\left\{g(h_w^{(l)})\mid w \in N(v)\right\}\right)
$$

Where:
- $h_v^{(l)}$: Node embedding at layer$l$.
- $N(v)$: Neighbors of node$v$.
- $f$and$g$: Learnable functions.
- $\{\cdot\}$: Permutation-invariant aggregator (e.g., mean, sum).

In **relational-temporal message passing**, only neighbors with timestamps$\leq t$are considered:

$$
N_{\leq t}(v)=\{w \in N(v)\mid \tau(w) \leq t\}
$$

This ensures temporal consistency during training.

---

#### **Heterogeneous Message Passing**

For heterogeneous graphs with multiple node and edge types:

$$
h_v^{(l+1)}=f_{\phi(v)}\left(h_v^{(l)},\left\{f_R\left(\sum_{w \in N_R(v)} g_R(h_w^{(l)})\right)\mid \forall R=(T,\phi(v))\in R\right\}\right)
$$

Where:
- $N_R(v)$: Neighbors of$v$connected via relation$R$.
- $f_{\phi(v)}$,$f_R$,$g_R$: Type-specific functions.

---

#### **Node-Level Prediction Model Head**

Given final node embeddings$h_v^{(L)}$, predictions are made via a task-specific head:

$$
\hat{y}_v=f(h_v^{(L)})
$$

Where$f$is a neural network or linear layer depending on the task (regression, classification).

---

#### **Link-Level Prediction Model Head**

For link prediction tasks involving two nodes$v_1$and$v_2$:

$$
\hat{y}_{v_1,v_2}=f(h_{v_1}^{(L)}, h_{v_2}^{(L)})
$$

Where$f$could be a bilinear operator, dot product, or multi-layer perceptron.

---

#### **Loss Function**

For regression tasks (e.g., LTV prediction):

$$
\mathcal{L}=\frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|
$$

For binary classification tasks (e.g., churn prediction):

$$
\mathcal{L}=-\frac{1}{N} \sum_{i=1}^N \left[y_i \log(\hat{y}_i)+(1 - y_i)\log(1 - \hat{y}_i)\right]
$$

Average Precision (AP) or Mean Absolute Error (MAE) are used as evaluation metrics.

---
