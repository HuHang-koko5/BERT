# BERT
BERT study process

## _10/29_


开始写*transformer*的demo，基于[The Annoted Translater](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

今天写到了attention的具体实现之前

主要包括Encoder-Decoder的具体结构

实现的class：


	class Encoder-Decoder
	class Encoder
	class Decoder
	class EncoderLayer
	class DecoderLayer
	class LayerNorm
	class SubLayerConnection
	class Generator
	
## Encoder-Decoder类

是编码器-解码器标准类，成员包含一个Encoder类和一个Decoder类,和一个Generator类

实现了encode和decode方法

![encode-decoder](/img/Encoder-Decoder.png)

## Encoder/Decoder类

是编码器/解码器类，成员包含了N层（通常N=6）En/DecoderLayer类

![Encoder](/img/Encoder.png)

## EncoderLayer/DecoderLayer

是编码器/解码器每层的具体实现（Self Attention + Feed Forward）

![EncoderLayer](/img/EncoderLayer.png)

包括2个Residual connect层：

- attention层 
- feed_forward层


## SubLayerConnection类

实现了Residual connect


![Residual](/img/Residual.png)

x和sublayer作为参数传入forward方法返回


`return x + self.dropout(sublayer(self.norm(x)))`


传入的sublayer在encoder中分别是self Attention和Feed Forward

在Decoder中是Src Attention, Self Attention 和 Feed Forward





## 多个layer的克隆操作
    
		def clones(nn.module, N):
			return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

## LayerNorm 

用于layer normalization[-reference](https://arxiv.org/abs/1607.06450)(未读，之后补)

	class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2

## Generator

Liner + softmax
	
	class Generator(nn.Module):
	    def __init__(self, d_model, vocab):
	        super(Generator, self).__init__()
	        self.proj = nn.Linear(d_model, vocab)
	    
	    def forward(self,x):
	        return F.log_softmax(self.proj(x), dim=-1)

## subsequent_mask
在Decoder中为了防止在attention时query访问后面的字段，需要给后面添加mask

用到了`np.triu`方法，生成上三角矩阵
	
	def subsequent_mask(size):
	    attn_shape = (1,size,size)
	    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	    return torch.from_numpy(subsequent_mask) == 0

![triu](/img/triu.png)



## Train record

### this part show the result of my trian with different hyperparameter selection

## Format

	'LR': 1e-6

	'BATCHSIZE': 32

	'DEVICE': rtx2070s
	
	'EPOCH': 6

Pic of result:

acc chart:

loss chart:

heatmap:








