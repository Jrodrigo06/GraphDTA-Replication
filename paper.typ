#import "@preview/clean-math-paper:0.2.2": *

#let date = datetime.today().display("[month repr:long] [day], [year]")
#show: template.with(
  title: "GraphDTA - Replication",
  authors: (
    (name: "Jerome Rodrigo"),
  ),

  date: "July 23, 2025",
  heading-color: rgb("#0000ff"),
  link-color: rgb("#008002"),
  // Insert your abstract after the colon, wrapped in brackets.
  // Example: `abstract: [This is my abstract...]`
  abstract: [This paper presents a replication study of the GraphDTA model, which integrates Graph Neural Networks (GNNs) and Convolutional Neural Networks (CNNs) to predict drug-target binding affinity. The primary goal is to validate the original study’s findings while deepening my understanding of its computational and mathematical underpinnings. I analyze the model architecture, training process, and dataset used, and explore the theoretical foundations of GNNs and CNNs in the context of bioinformatics. Through this replication, I aim to enhance my knowledge in drug discovery, bioinformatics, and machine learning.],
  keywords: ("Graph Neural Networks", "Convolutional Neural Networks", "Drug-Target Binding Affinity", "Deep Learning", "Bioinformatics", "Drug Discovery", "GraphDTA"),
)

= Introduction
This paper is a math-first walkthrough of reimplementing GraphDTA (Nguyen et al.) to learn how deep learning models can predict drug–target binding affinity. Instead of summarizing results from the original work, I rebuild the full pipeline, tokenizing protein sequences, representing small molecules as graphs, encoding them with CNNs and Graph Neural Networks, and combining the learned embeddings for regression. Along the way I unpack the math behind 1D convolutions, global pooling, message passing on molecular graphs, and the loss/objective used for affinity prediction.  

= High Level Overview


= Theorems

= Protein encoding
  == One Hot Encoding
  So we use one-hot encoding to represent amino acids in protein sequences. How the function works is that ita takes a sequence of amino acids and converts each amino acid into a one-hot encoded vector. The amino acids are represented by a number in a dictionary corresponding to their index in the amino acid vocabulary. The one-hot encoded vector is length 22, where the index corresponding to the amino acid is set to 1, and all other indices are set to 0.
  === How and why one-hot encoding works
  So one-hot encoding is a way to represent categorical data as binary vectors. Each category is represented by a vector where one element is set to 1 (the "hot" part/the category that is being represented) and all other elements in the vector are set to 0. It works for neural networks because it allows the model to learn relationships between different categories without assuming any ordinal relationship between them. In the context of protein sequences, one-hot encoding allows us to represent each amino acid as a unique vector, which can then be processed by neural networks. One-hot encoding works for amino acids really well as the feature space is small (22 amino acids) and the relationships between them are not ordinal.
 
= Drug encoding
   == SMILES to Graph representation
   === Atoms to Nodes
   Following the paper I used RDKit to convert SMILES compounds into molecular graphs. Each node in the graph is represented by a vector of features with most being one-hot encoded. The features include the atom symbol, the number of adjacent hydrogens, the number of adjacent atoms, the implicit value of the atom, and whether the atom is in a aromatic structure.
   === Bonds to Edges
   Each edge in the graph is made by taking the bonds using RDKit. The bond notes the atom indices and the graph is undirected by making edges both ways. Also the matrix is then tranposed to be used in the GNNs. 

= Graph Neural Networks
== Basic Overview
So Graph Neural Networks (GNNs) work similarly to regular neural networks, but they are built for graph-structured data. They learn to understand relationships between nodes in a graph through encoding neighboring node information. Pretty much each layer in the GNN gets information from neighboring nodes and uses it (e.g., by summing or averaging) to update the given node's representation through a shared neural transformation. This is done through a process called message passing, where each node sends and receives messages from its neighbors. The GNN learns to aggregate these messages and update the node representations iteratively. This allows the model to learn rich structure-aware embeddings, which is perfect for tasks like drug-target binding affinity prediction where the relationships between atoms in a molecule are crucial.

== GCNs
So the first approach in the GraphDTA paper is to use a Graph Convolutional Network (GCN). GCNs are a type of GNN that applies convolutional operations on graph-structured data. 

===  High Level Overview
A GCN layer lets each atom average its neighbors + itself, then passes that summary through the same tiny neural step (a shared set of weights) and a ReLU (keeps positives, zeros out negatives which adds some non-linearity which is essential for deep learning). Stacking layers lets information flow multiple bonds away. Finally, max-pool takes the largest value per feature across all atoms, giving one vector for the molecule.


=== Techical Overview
We represent each molecule as an undirected graph $G=(V,E)$ with:
- $N = |V|$ atoms (nodes),
- Node features $X in RR^{N × C}$ Where N is the number of atoms and C is the number of features per atoms (13).
- We add self-loops: $tilde(A) = A + I$ Where the degree matrix is $tilde(D) in RR^(N × N)$ with $tilde(D)_(i i) = sum_(j = 1)^(N) tilde(A)_(i j)$
- Adjacency matrix $A in {0,1}^{N × N}$ where $A_{i,j}=1$ if atoms $i$ and $j$ are bonded.
- Then we have our Degree matrix $D in RR^{N × N}$ where $D_{i,i} = sum_{j=1}^{N} A_{i,j}$ is the degree of node $i$ which is the number of connections a node has plus itself.
- The normalized weights matrix is $S_(i j)= 1/sqrt(D_(i i) D_(j j) )$ if $A_( i j) = 1$ else $S_(i j) = 0$.
  - The reason the we used a normalized weight matrix is to prevent the model from being biased towards nodes with high degrees. This helps the model learn more balanced representations of nodes in the graph.
- A single GCN layer updates node features by degree-aware neighbor avergaging followed by a shared linear map and ReLU:$ H^((l + 1)) = "ReLU"(S H^((l))W^((l))) "with" H^((l)) in RR ^(N × C_l) "and" W^((l)) in RR^(C_l × C_(l+1)) $
  - So what this equation means is that we take the current node features $H^((l))$, multiply it by the normalized weights matrix $S$, and then apply a linear transformation with weights $W^((l))$ followed by a ReLU activation function. This allows the model to learn complex relationships between nodes in the graph while preventing overfitting.
- Graph-level readout: after $L$ layers, pool node embeddings with global max to get one vector per molecule:
  $h_G[j]= "max"_{v in V} H^((L))_{v,j}$ (permutation-invariant).
   - This means that we take the maximum value of each feature across all nodes in the graph to create a single vector representation for the entire molecule. This is important because it allows us to capture the most important features of the molecule.

=== GCN Notation → Plain English 
- $G=(V,E)$: molecule as a graph (atoms $V$, bonds $E$); $N=|V|$.
- $H^((0)) = X in RR^(N × C)$: initial node features (13 per atom).
- $A in {0,1}^(N × N)$: adjacency; $A_(i j)=1$ if atoms $i$ and $j$ are bonded.
- $tilde(A) = A + I$: add self-loops so each atom keeps its own signal.
- $tilde(D)_(i i) = sum_(j=1)^(N) tilde(A)_(i j)$: degree from $tilde(A)$ (neighbors + self).
- $S = tilde(D)^(-1/2) * tilde(A) * tilde(D)^(-1/2)$: normalized neighbor-averaging weights  
  (if $tilde(A)_(i j)=1$, then $S_(i j) = 1/sqrt(tilde(D)_(i i) * tilde(D)_(j j))$, else $0$).
- $W^((l)) in RR^(C_l × C_(l+1))$: shared linear map (like a dense layer) changing feature width.
- Update rule: $H^((l+1)) = "ReLU"( S * H^((l)) * W^((l)) )$  
  → average neighbors (incl. self) → remix features → keep positives.
- Graph readout (global max): $h_G[j] = "max"_(v in V) H^((L))_(v,j)$ → one $1 × C_L$ vector per molecule.
- Shapes at a glance: $H^((0)): N × 13 → H^((1)): N × 32 → H^((3)): N × 32 →_( "max-pool" ) h_G: 1 × 32$.
