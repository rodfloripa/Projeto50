<p align="justify">

# Projeto50: Implementando Flash Attention do Zero em PyTorch

</p>

<p align="justify">

Este projeto tem como objetivo construir uma implementação educacional da <b>Flash Attention</b>, uma das otimizações mais importantes já criadas para Transformers modernos.

A ideia principal não é apenas utilizar uma biblioteca pronta, mas entender profundamente:

* por que a self-attention tradicional consome tanta memória
* como a GPU sofre com movimentação excessiva de dados
* como técnicas de <i>tiling</i> reduzem acesso à VRAM
* por que FlashAttention revolucionou LLMs modernos

O projeto demonstra na prática:

* Self-Attention clássica
* gargalo de memória
* implementação em blocos
* softmax estável incremental
* comparação de desempenho
* escalabilidade

Tudo isso usando apenas:

* Python
* PyTorch
* CUDA (opcional)
* Matplotlib

</p>

---

<p align="justify">

# Objetivos do Projeto

</p>

<p align="justify">

Ao final do projeto, será possível:

* entender o gargalo computacional da attention
* implementar Flash Attention manualmente
* visualizar o ganho de memória
* comparar diferentes tamanhos de sequência
* compreender como LLMs modernos são otimizados

O projeto envolve conhecimento em:

* Deep Learning
* Transformers
* CUDA awareness
* otimização de memória
* álgebra linear
* engenharia de IA

</p>

---

<p align="justify">

# Conceito Matemático

</p>

<p align="justify">

A self-attention tradicional é definida por:

</p>

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

<p align="justify">

O problema aparece porque a matriz:

</p>

$$
QK^T
$$

<p align="justify">

possui dimensão:

</p>

$$
n \times n
$$

<p align="justify">

Logo, o custo cresce quadraticamente:

</p>

$$
O(n^2 d)
$$

<p align="justify">

Quando o contexto cresce para milhares de tokens, o uso de memória explode.

Flash Attention resolve isso processando blocos menores sem materializar toda a matriz de attention na memória.

</p>

---

<p align="justify">

# Implementação da Self-Attention Tradicional

</p>

```python
import torch
import torch.nn.functional as F
import math

def standard_attention(Q, K, V):

    d_k = Q.size(-1)

    scores = torch.matmul(
        Q,
        K.transpose(-2, -1)
    )

    scores = scores / math.sqrt(d_k)

    attn = F.softmax(scores, dim=-1)

    output = torch.matmul(attn, V)

    return output
```

---

<p align="justify">

# Entendendo o Gargalo da Attention Tradicional

</p>

<p align="justify">

O principal problema está nesta linha:

</p>

```python
scores = torch.matmul(
    Q,
    K.transpose(-2, -1)
)
```

<p align="justify">

Aqui o modelo cria explicitamente a matriz:

</p>

$$
QK^T
$$

<p align="justify">

Se a sequência possuir 8192 tokens, então a attention terá:

</p>

$$
8192 \times 8192
$$

<p align="justify">

posições.

Isso significa dezenas de milhões de elementos simultaneamente na VRAM.

O problema não é apenas computacional.

O verdadeiro gargalo é movimentação de memória:

* leitura de VRAM
* escrita de VRAM
* cache misses
* bandwidth limitado da GPU

É exatamente isso que Flash Attention tenta resolver.

</p>

---

<p align="justify">

# Ideia Central da Flash Attention

</p>

<p align="justify">

A ideia principal é extremamente elegante:

em vez de calcular toda a attention de uma vez, dividimos o problema em pequenos blocos.

A GPU então:

* carrega um pequeno bloco
* computa parcialmente
* atualiza o softmax incrementalmente
* descarta intermediários

Assim evitamos criar a matriz gigantesca de attention na memória.

O algoritmo transforma um problema de memória em um problema de streaming computacional.

</p>

---

<p align="justify">

# Implementação da Flash Attention

</p>

```python
import torch
import math

def flash_attention(Q, K, V, block_size=256):

    B, N, D = Q.shape

    output = torch.zeros_like(Q)

    scale = 1.0 / math.sqrt(D)

    for i in range(0, N, block_size):

        q_block = Q[:, i:i + block_size]

        row_max = None
        row_sum = None
        out_block = None

        for j in range(0, N, block_size):

            k_block = K[:, j:j + block_size]
            v_block = V[:, j:j + block_size]

            scores = torch.matmul(
                q_block,
                k_block.transpose(-2, -1)
            ) * scale

            current_max = scores.max(
                dim=-1,
                keepdim=True
            ).values

            if row_max is None:

                row_max = current_max

                exp_scores = torch.exp(
                    scores - row_max
                )

                row_sum = exp_scores.sum(
                    dim=-1,
                    keepdim=True
                )

                out_block = torch.matmul(
                    exp_scores,
                    v_block
                )

            else:

                new_max = torch.maximum(
                    row_max,
                    current_max
                )

                old_scale = torch.exp(
                    row_max - new_max
                )

                exp_scores = torch.exp(
                    scores - new_max
                )

                row_sum = (
                    old_scale * row_sum +
                    exp_scores.sum(
                        dim=-1,
                        keepdim=True
                    )
                )

                out_block = (
                    old_scale * out_block +
                    torch.matmul(
                        exp_scores,
                        v_block
                    )
                )

                row_max = new_max

        output[:, i:i + block_size] = (
            out_block / row_sum
        )

    return output
```

---

<p align="justify">

# Explicação Detalhada da Flash Attention

</p>

<p align="justify">

A implementação começa extraindo:

* batch size
* tamanho da sequência
* dimensionalidade dos embeddings

</p>

```python
B, N, D = Q.shape
```

<p align="justify">

Depois criamos o tensor final:

</p>

```python
output = torch.zeros_like(Q)
```

<p align="justify">

Esse tensor armazenará o resultado final da attention.

</p>

---

<p align="justify">

# Fator de Escala

</p>

```python
scale = 1.0 / math.sqrt(D)
```

<p align="justify">

Na self-attention tradicional, os scores são divididos por:

</p>

$$
\sqrt{d_k}
$$

<p align="justify">

Isso evita explosão numérica no softmax.

Sem essa normalização:

* os valores cresceriam muito
* o softmax saturaria
* o treinamento ficaria instável

</p>

---

<p align="justify">

# Processamento em Blocos

</p>

```python
for i in range(0, N, block_size):
```

<p align="justify">

Aqui começa o conceito mais importante da Flash Attention.

Em vez de processar toda a sequência de uma vez, dividimos em pequenos blocos.

Se:

* N = 8192
* block_size = 256

então processamos apenas pequenos segmentos por vez.

Isso reduz drasticamente:

* uso de memória
* movimentação de dados
* pressão na VRAM

</p>

---

<p align="justify">

# Extração do Bloco de Query

</p>

```python
q_block = Q[:, i:i + block_size]
```

<p align="justify">

Selecionamos apenas um pequeno bloco da matriz Q.

Isso significa que a GPU trabalha apenas com uma fração da sequência em cada instante.

Esse detalhe é fundamental porque GPUs modernas funcionam muito melhor quando os dados cabem em cache.

</p>

---

<p align="justify">

# Estatísticas Incrementais

</p>

```python
row_max = None
row_sum = None
out_block = None
```

<p align="justify">

Essas variáveis armazenam:

* máximo parcial do softmax
* soma parcial exponencial
* saída parcial acumulada

Elas permitem calcular o softmax incrementalmente sem armazenar toda a matriz de attention.

Esse é um dos conceitos matemáticos mais sofisticados da Flash Attention.

</p>

---

<p align="justify">

# Processamento dos Blocos de Key e Value

</p>

```python
for j in range(0, N, block_size):
```

<p align="justify">

Agora percorremos os blocos de:

* K
* V

A attention passa a funcionar como um fluxo contínuo:

* carrega bloco
* computa
* acumula
* descarta

Isso reduz enormemente o tráfego de memória.

</p>

---

<p align="justify">

# Computação Local da Attention

</p>

```python
scores = torch.matmul(
    q_block,
    k_block.transpose(-2, -1)
) * scale
```

<p align="justify">

Aqui realizamos a multiplicação matricial local entre os blocos.

Em vez de criar:

</p>

$$
QK^T
$$

<p align="justify">

completo, criamos apenas pequenos pedaços temporários.

Esse detalhe reduz drasticamente a memória necessária.

</p>

---

<p align="justify">

# Máximo Parcial do Softmax

</p>

```python
current_max = scores.max(
    dim=-1,
    keepdim=True
).values
```

<p align="justify">

O softmax pode sofrer overflow numérico.

Por isso usamos a técnica clássica:

</p>

$$
\mathrm{softmax}(x)=\frac{e^{x-\max(x)}}{\sum e^{x-\max(x)}}
$$

<p align="justify">

Subtrair o máximo mantém os expoentes numericamente estáveis.

</p>

---

<p align="justify">

# Primeiro Bloco

</p>

```python
if row_max is None:
```

<p align="justify">

No primeiro bloco inicializamos as estatísticas do softmax.

Aqui começamos a acumular:

* soma exponencial
* saída parcial
* máximo da linha

</p>

---

<p align="justify">

# Exponencial Estável

</p>

```python
exp_scores = torch.exp(
    scores - row_max
)
```

<p align="justify">

Esse trecho evita explosões numéricas.

Sem isso:

* valores grandes explodiriam
* valores pequenos desapareceriam
* o softmax perderia precisão

</p>

---

<p align="justify">

# Soma Incremental

</p>

```python
row_sum = exp_scores.sum(
    dim=-1,
    keepdim=True
)
```

<p align="justify">

Aqui acumulamos parcialmente o denominador do softmax.

A Flash Attention nunca computa o softmax completo explicitamente.

Ela mantém apenas estatísticas resumidas.

</p>

---

<p align="justify">

# Acúmulo da Saída

</p>

```python
out_block = torch.matmul(
    exp_scores,
    v_block
)
```

<p align="justify">

Esse trecho produz a saída parcial da attention.

Cada bloco contribui parcialmente para o resultado final.

</p>

---

<p align="justify">

# Atualização Incremental Estável

</p>

```python
new_max = torch.maximum(
    row_max,
    current_max
)
```

<p align="justify">

Agora ocorre um detalhe extremamente sofisticado.

Como cada bloco possui escalas diferentes, precisamos reajustar numericamente o softmax.

Isso mantém estabilidade mesmo processando os dados em streaming.

</p>

---

<p align="justify">

# Reescala dos Valores Antigos

</p>

```python
old_scale = torch.exp(
    row_max - new_max
)
```

<p align="justify">

Os valores antigos precisam ser reescalados para permanecer consistentes com o novo máximo global.

Essa é uma das ideias centrais da Flash Attention.

</p>

---

<p align="justify">

# Atualização da Soma Global

</p>

```python
row_sum = (
    old_scale * row_sum +
    exp_scores.sum(
        dim=-1,
        keepdim=True
    )
)
```

<p align="justify">

Aqui acumulamos corretamente todas as contribuições dos blocos anteriores.

O algoritmo preserva exatamente o resultado matemático da self-attention tradicional.

</p>

---

<p align="justify">

# Atualização da Saída Parcial

</p>

```python
out_block = (
    old_scale * out_block +
    torch.matmul(
        exp_scores,
        v_block
    )
)
```

<p align="justify">

Esse trecho acumula incrementalmente a saída da attention.

Ao final de todos os blocos teremos exatamente o mesmo resultado da attention tradicional, porém usando muito menos memória.

</p>

---

<p align="justify">

# Normalização Final

</p>

```python
output[:, i:i + block_size] = (
    out_block / row_sum
)
```

<p align="justify">

Aqui concluímos o softmax.

A saída final é normalizada corretamente utilizando apenas estatísticas acumuladas.

</p>

---

<p align="justify">

# Resultados Esperados

</p>

<p align="justify">

Para sequências pequenas, Flash Attention pode parecer semelhante à attention tradicional.

Mas conforme o contexto cresce:

* attention tradicional explode em memória
* Flash Attention escala muito melhor
* a GPU trabalha de forma muito mais eficiente

O ganho principal aparece em:

* contexto longo
* batch grande
* treinamento de LLMs

</p>

---

<p align="justify">

# Melhorias Futuras

</p>

<p align="justify">

O projeto pode evoluir para:

* CUDA custom kernel
* Triton
* causal masking
* multi-head attention
* KV cache
* mixed precision
* quantização
* integração com Transformer completo

Também é possível comparar com:

* xFormers
* FlashAttention v2
* PyTorch scaled_dot_product_attention

</p>

---

<p align="justify">



</p>



---

<p align="justify">

# Conclusão

</p>

<p align="justify">

Flash Attention foi uma das maiores revoluções recentes em Transformers porque atacou o verdadeiro gargalo dos modelos modernos: movimentação de memória.

Em vez de apenas reduzir FLOPs, a técnica reorganiza o fluxo computacional para aproveitar melhor cache e VRAM.

Este projeto mostra exatamente como isso funciona na prática.

Além de aprender Transformers, o desenvolvedor passa a compreender:

* computação em GPU
* arquitetura de memória
* estabilidade numérica
* otimização de kernels
* engenharia de LLMs modernos

Esse tipo de conhecimento é extremamente valorizado em pesquisa e indústria de IA.

</p>
