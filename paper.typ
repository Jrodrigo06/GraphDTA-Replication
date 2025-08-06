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
dadada

= Theorems

= Protein encoding
  == One Hot Encoding
  So we use one-hot encoding to represent amino acids in protein sequences. How the function works is that ita takes a sequence of amino acids and converts each amino acid into a one-hot encoded vector. The amino acids are represented by a number in a dictionary corresponding to their index in the amino acid vocabulary. The one-hot encoded vector is length 22, where the index corresponding to the amino acid is set to 1, and all other indices are set to 0.
  === How and why one-hot encoding works
  So one-hot encoding is a way to represent categorical data as binary vectors. Each category is represented by a vector where one element is set to 1 (the "hot" part/the category that is being represented) and all other elements in the vector are set to 0. It works for neural networks because it allows the model to learn relationships between different categories without assuming any ordinal relationship between them. In the context of protein sequences, one-hot encoding allows us to represent each amino acid as a unique vector, which can then be processed by neural networks. One-hot encoding works for amino acids really well as the feature space is small (22 amino acids) and the relationships between them are not ordinal.
 
= Drug encoding
   == SMILES to Graph representation
   Following the paper I used RDKit to convert SMILES compounds into molecular graphs. Each node in the graph is represented by a vector of features with most being one-hot encoded. The features include the atom symbol, the number of adjacent hydrogens, the number of adjacent atoms, the implicit value of the atom, and whether the atom is in a aromatic structure. 
    == Graph Neural Networks
