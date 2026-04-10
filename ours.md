## Nostra formulazione

**Variabili.**
- $x \in \mathbb{R}^D$ osservazione.
- $c \in \{0,1\}^K$ presenza di un latente.
- $e(x) = (e_1, \dots, e_K)$, $e_k \in \mathbb{R}^d$, **embedding del latente** per sample (continuo).
- $h = (h_1, \dots, h_K)$ **prototipi** (parametri globali con prior debole).

**Modello generativo.**
$$
p(c_k) = \operatorname{Bern}(c_k;\pi)
$$
$$
p_\phi(x \mid c) = \mathcal{N}\!\bigl(x;\; g_\phi(c \odot e(x)),\; I\bigr)
$$

dove $e(x) = f_\theta(x)$ è l'output deterministico dell'encoder — non un latente indipendente, ma una funzione di $x$ — quindi non compare nel condizionante del modello generativo.

**Posterior variazionale amortizzata.**

$$
q_{\theta}(c \mid x) = \prod_k \operatorname{Bern}(c_k;\alpha_k), \quad \alpha_k = a\!\left(-\|h_{k} - e_k(x)\|\right), \quad e_k(x) = f_{k}(x)
$$

dove $a(\cdot)$ è una funzione di attivazione generica (non necessariamente una sigmoide).

**Open questions:**
- $\|h - e\|$: quale norma/similarità usare? inner product, cosine similarity, distanza euclidea?
- come definire $a(\cdot)$? gumbel sigmoid, sigmoid, sigmoid\_cosine\_scoring?
- caso inner product con gumbel sigmoid: $\alpha_k = a\!\left(\frac{h_k \cdot e_k(x) + \epsilon}{\tau}\right)$. Problematica: norme non bounded.
- caso cosine similarity con sigmoid\_cosine\_scoring: $\alpha_k = a\!\left(\frac{\langle h_k, e_k(x)\rangle}{\tau}\right)$. Problematica: assenza di rumore stocastico.
