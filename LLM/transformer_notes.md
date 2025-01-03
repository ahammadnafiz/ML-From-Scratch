### 1. **Input Representation: Word Embedding**

The first step in the Transformer is converting the input text (e.g., sentences) into a form that the model can understand. This is done through **embedding**.
#### Theory:

- Words or tokens in the input are typically represented by integers (indexes in a vocabulary).

- An embedding layer maps each of these integers to dense vectors (real-valued numbers), which capture semantic properties of the words.

- For example, words like "king" and "queen" will have similar vectors in the embedding space because they have similar meanings.
#### Math:

For a sequence of \( n \) tokens, the embedding process is:

$$
\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n]
$$


Where \( $\mathbf{x}_i$ \) is the vector representing the \(i\)-th token.

---

### 2. **Positional Encoding**

Since Transformers don't have any recurrent structure (like LSTMs) or any built-in notion of word order, we need to explicitly encode the position of each token in the sequence. **Positional Encoding** adds this information to the token embeddings.

#### Theory:

- Positional encodings use sinusoidal functions to produce unique encodings for each position, making sure that the model can distinguish between tokens based on their position in the sequence.

- The sinusoidal functions ensure that nearby positions have similar encodings, while distant ones are distinguishable by the difference in their positional encodings.

#### Intuition:

- \( $\sin$ \) and \( $\cos$ \) are periodic functions, so they naturally represent positions in sequences. The formula makes sure that each position gets a unique combination of sine and cosine values.

#### Math:

For a position \( pos \) and dimension \( i \), the positional encoding is:


$$
PE_{pos, 2i} = \sin \left( \frac{pos}{10000^{\frac{2i}{d}}} \right)
\quad \text{and} \quad
PE_{pos, 2i+1} = \cos \left( \frac{pos}{10000^{\frac{2i}{d}}} \right)
$$

Where:

- \( pos \) is the position in the sequence (e.g., 1 for the first word).

- \( i \) is the dimension index (each position is represented in multiple dimensions).

- \( d \) is the total dimensionality of the embedding.


The final embedding is then:


$$
\mathbf{X} = \mathbf{X} + \mathbf{PE}
$$

This gives each token both its semantic meaning (from the embedding) and its position in the sequence.

---
### 3. **Self-Attention**

The core of the Transformer is **self-attention**, which allows the model to focus on different parts of the input sequence when encoding a token. For each token, self-attention determines which other tokens it should "attend to" based on their relationships.
#### Intuition:

- Imagine reading a sentence where the word "bank" might refer to a financial institution or the side of a river. To determine the correct meaning, the model needs to "attend" to the context in which the word appears, i.e., it looks at surrounding words.

- Self-attention calculates how much focus each word should place on the others in the sentence.

#### Math:

Self-attention calculates three vectors for each token: **Query (Q)**, **Key (K)**, and **Value (V)**. These vectors are obtained by multiplying the input embeddings by learned weight matrices \( $W_Q$ \), \( $W_K$ \), and \( $W_V$ \).

$$
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V
$$

Where:

- \( $\mathbf{X}$ \) is the input sequence of token embeddings.

- \( $W_Q, W_K, W_V$ \) are learned weight matrices.

The attention scores between a query and all keys are computed as:


$$
\text{Attention Score} = \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}

$$

Where \( $d_k$ \) is the dimensionality of the key vectors (scaling factor to stabilize gradients).

  
The softmax function is applied to these scores to normalize them into probabilities (this ensures they sum to 1):


$$
\text{Attention Weights} = \text{softmax} \left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right)
$$

Then, the output of the self-attention mechanism is the weighted sum of the value vectors:

$$
\text{Output} = \text{Attention Weights} \times \mathbf{V}
$$
---
### 4. **Multi-Head Attention**

Instead of using just one attention mechanism, the Transformer uses **multiple attention heads**. Each head learns a different set of attention weights, allowing the model to attend to different parts of the sequence in different ways.

#### Intuition:

- If one head focuses on syntactic relationships (e.g., subject-verb agreement), another might focus on semantic relationships (e.g., word meanings). By combining these, the model can learn richer representations.

#### Math:

Each attention head is computed separately:


$$
\mathbf{Q}_i = \mathbf{X} \mathbf{W}_i^Q, \quad \mathbf{K}_i = \mathbf{X} \mathbf{W}_i^K, \quad \mathbf{V}_i = \mathbf{X} \mathbf{W}_i^V
$$

Where \( i \) indexes the different attention heads.

The outputs of each head are concatenated:


$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{Head}_1, \text{Head}_2, \dots, \text{Head}_h) \mathbf{W}^O
$$

Where \( $W^O$ \) is a learned weight matrix that combines the outputs of all attention heads.

---
### 5. **Feedforward Neural Networks (FFN)**

After the attention mechanism, the output is passed through a **feedforward neural network** for each token.

#### Intuition:

- The FFN operates on each token independently and applies transformations to increase the representation's capacity.

#### Math:

The feedforward network is typically composed of two fully connected layers with a ReLU activation in between:

$$
\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2
$$

Where:

- \( $\mathbf{W}_1$ \) and \( $\mathbf{b}_1$ \) are the weights and bias of the first layer.

- \( $\mathbf{W}_2$ \) and \( $\mathbf{b}_2$ \) are the weights and bias of the second layer.

---

### 6. **Residual Connections & Layer Normalization**

**Residual connections** are added around each sub-layer (attention and feedforward) to help the model learn more effectively. **Layer normalization** is then applied to stabilize training.
#### Intuition:

- Residual connections allow the gradients to flow more easily during backpropagation, reducing the likelihood of vanishing gradients.

- Layer normalization helps the model converge faster by normalizing the activations.

#### Math:

For each sub-layer (attention or feedforward):

$$
\mathbf{z}_{\text{attn}} = \text{LayerNorm}(\mathbf{X} + \text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}))
$$


$$
\mathbf{z}_{\text{ffn}} = \text{LayerNorm}(\mathbf{z}_{\text{attn}} + \text{FFN}(\mathbf{z}_{\text{attn}}))
$$

---

### 7. **Encoder Stack**

The encoder consists of \( N \) identical layers. Each layer includes:

- Multi-head self-attention.

- Position-wise feedforward network.

- Residual connections and layer normalization.

---
### 8. **Decoder**

The decoder is similar to the encoder but with an additional layer of cross-attention. The decoder uses both the input from the encoder and its previous outputs (during autoregression) to generate the output sequence.

---
### 9. **Output Layer**

After the decoder layers, a **linear transformation** is applied, followed by **softmax**, to generate the output sequence.

#### Math:

$$
\mathbf{Y} = \text{softmax}(\mathbf{z}_{\text{dec}} \mathbf{W}_y + \mathbf{b}_y)
$$

Where

 \( $\mathbf{W}_y$ \) is the weight matrix and \( $\mathbf{b}_y$ \) is the bias.

---
### 10. **Training Objective**

The model is trained by minimizing the **cross-entropy loss** between the predicted output and the target sequence:

$$
\mathcal{L} = - \sum_{t=1}^{n} \log P(y_t | y_1, \dots, y_{t-1})
$$

Where 

\( $P(y_t | y_1, \dots, y_{t-1})$ \) is the probability of generating token \( $y_t$ \) given the previous tokens.

---

### Conclusion:

The Transformer model is a **powerful architecture** that leverages self-attention to capture relationships between all tokens in a sequence, regardless of their position. Its key features are:

- Parallelization through self-attention (unlike RNNs).

- Multiple attention heads to capture different types of relationships.

- Layer normalization and residual connections for stable training.