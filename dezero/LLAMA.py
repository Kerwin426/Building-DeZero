import numpy as np
from functions import Function ,reshape,cat,matual,softmax,dropout,mean_squared_error
from typing import Union ,Literal,Callable
from layers import Linear ,Parameter,Layer
import json
from models import Model,MLP
from core import Variable 



class Tokenizer:
    def __init__(self,model_path:str):
        with open(model_path,"r",encoding="utf-8") as f:
            model = json.load(f)
        self.vocab = model["tokens"]
        self.scores =model["scores"]
        self.bos_id = 1
        self.eos_id = 2
        self.n_words = len(self.vocab)
        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"]
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1
    def str_lookup(self,token:str)->int:
        try:
            index = self.vocab.index(token)
            return index
        except ValueError as err:
            return -1
    def encode(
        self,
        text:str,
        add_bos :bool = True,
        add_eos: bool = False,
        add_prefix: bool = True,
        add_new_bos: bool = False,
    )->list[int]:
        tokens = []
        for pos, char in enumerate(text):
            id = self.str_lookup(char)
            if id >=0:
                tokens.append(id)
        while True:
            best_score = -1e10
            best_id = -1
            best_idx = -1
            for i in range(len(tokens)-1):
                # 采用BPE原理，对于每一个字符token，逐步合并出现频率最高的连续的两个字符组合
                string = self.vocab[tokens[i]]+self.vocab[tokens[i+1]]
                id = self.str_lookup(string)
                if id != -1 and self.scores[id]>best_score:
                    best_score = self.scores[id]
                    best_id = id 
                    best_idx = i
            if best_idx == -1:
                break
            tokens[best_idx] = best_id
            tokens = tokens[0:best_idx+1]+tokens[best_idx+2:]
        if add_bos:
            tokens.insert(0,self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        if add_prefix:
            tokens.insert(0, self.special_tokens['sop'])
            tokens.insert(0, self.special_tokens['[gMASK]'])
        if add_new_bos:
            tokens.append(self.bos_id)
        return tokens

    def decode(self,ids:list[int])->str:
        res = []
        for i in ids:
            token = self.vocab[i]
            res.append(token)
        text = "".join(res)
        # 移除可能存在的句子起始和结束标识符
        text = text.strip("<s>").strip("</s>")
        return text

class SelfAttention(Model):
    def __init__(self,args:'LLaMaArgs',rope_apply:Callable):
        super(SelfAttention,self).__init__()
        assert args.num_heads *args.head_dim ==args.hidden_size
        assert args.num_heads % args.num_key_value_heads ==0
        assert args.head_dim %2 ==0

        self.max_len = args.max_len
        self.max_batch_size = args.max_batch_size
        self.enable_kv_cache = args.enable_kv_cache
        self.use_gpu = args.use_gpu
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_heads
        self.head_dim = args.head_dim
        self.num_key_value_heads = args.num_key_value_heads
        self.attention_bias = args.attention_bias
        self.dropout_ratio = args.dropout_ratio
        self.dropout_on = args.dropout_ratio !=0
        self.kv_repeat_num = self.num_heads // self.num_key_value_heads
        self.rope_apply =  rope_apply
        
        self.q_proj = Linear(in_size=self.hidden_size,out_size=self.num_heads*self.head_dim,
                             nobias= ~self.attention_bias)
        self.k_proj = Linear(in_size=self.hidden_size,out_size=self.num_key_value_heads*self.head_dim,
                             nobias=~self.attention_bias)
        self.v_proj = Linear(in_size=self.hidden_size,out_size=self.num_key_value_heads*self.head_dim,
                             nobias=~self.attention_bias)
        self.o_proj = Linear(in_size=self.hidden_size,out_size=self.hidden_size,nobias=~self.attention_bias)


        # 启用kv cache
        if self.enable_kv_cache:
            self.k_cache =  Variable(np.zeros([self.max_batch_size,self.num_key_value_heads,0,self.head_dim]))
            self.v_cache = Variable(np.zeros([self.max_batch_size, self.num_key_value_heads, 0, self.head_dim]))
            if self.use_gpu:
                self.k_cache.to_gpu()
                self.v_cache.to_gpu()
    def forward(self,x,cos_pos,sin_pos):
        batch_size = x.shape[0]
        length = x.shape[1]
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.reshape(batch_size,length,self.num_heads,self.head_dim).transpose(0,2,1,3)
        k = k.reshape(batch_size,length,self.num_key_value_heads,self.head_dim).transpose(0,2,1,3)
        v = v.reshape(batch_size,length,self.num_key_value_heads,self.head_dim).transpose(0,2,1,3)

        # 这里的rope_apply 是两种不同版本实现的apply函数
        q = self.rope_apply(q,cos_pos,sin_pos)
        k = self.rope_apply(k,cos_pos,sin_pos)

        if self.enable_kv_cache:
            start_pos = self.k_cache.shape[2]# 好像也是0
        else:
            start_pos = 0
        
        if self.enable_kv_cache:
            self.k_cache =  cat((self.k_cache,k),axis=2)
            self.v_cache = cat((self.v_cache,v),axis=2)
        
        # 保持kv头数一致
        if self.num_heads != self.num_key_value_heads:
            k = k[:, np.arange(self.num_key_value_heads).repeat(self.kv_repeat_num), :, :]
            v = v[:, np.arange(self.num_key_value_heads).repeat(self.kv_repeat_num), :, :]

        attention_weight = matual(q,k.transpose(0,1,3,2))/np.sqrt(self.head_dim)
        mask = np.full((length,length),-np.inf)
        mask = np.triu(mask,k=1)
        mask = np.concatenate((np.zeros((length,start_pos)),mask),axis=1)

        if self.use_gpu:
            from cuda import as_cupy
            mask = as_cupy(mask)
        
        attention_weight = attention_weight + mask
        attention_weight = softmax(attention_weight,axis=-1)
        if self.dropout_on:
            attention_weight = dropout(attention_weight,self.dropout_ratio)
        
        output = matual(attention_weight,v)
        output = output.transpose(0,2,1,3).reshape(batch_size,length,self.hidden_size)
        output = self.o_proj(output)
        return output

# 激活函数 是对 relu 的改进 y =  x*sigmoid(x)
class SiLU(Function):
    def forward(self, x: np.ndarray) -> tuple[np.ndarray]:
        self.sigmoid = 1/(1+np.exp(-x))
        y = x*self.sigmoid
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        y = self.outputs[0]()
        gx = gy*(y+self.sigmoid*(1-y))
        return gx


def silu(x):
    return SiLU()(x)


class SwiGLUFeedForwardNetwork(Model):
    def __init__(self,hidden_size:int ,intermediate_size:int,use_bias:bool =False):
        super(SwiGLUFeedForwardNetwork,self).__init__()
        self.fc_gate = Linear(in_size=hidden_size,out_size=intermediate_size,nobias=~use_bias)
        self.fc_up = Linear(in_size=hidden_size,out_size=intermediate_size,nobias=~use_bias)
        self.fc_down = Linear(in_size=intermediate_size,out_size=hidden_size,nobias=~use_bias)
    
    def forward(self,x):
        x1 = self.fc_up(x)
        x = silu(self.fc_gate(x))
        x = x*x1
        x = self.fc_down(x)
        return x


# 均方根层归一化 是对层归一化的改进 直接计算特征的均方根 简化步骤
class RMSNormFunction(Function):
    def __init__(self,eps:float = 1e-6):
        self.epsilon = eps
    def forward(self,x:np.ndarray,w:np.ndarray)->tuple[np.ndarray]:
        self.rms_inv = ((x**2).sum(axis=x.ndim-1,keepdims=True)/x.shape[-1]+self.epsilon)**(-1/2)
        self.rms_x = x*self.rms_inv
        y = self.rms_x *w
        return y 
    def backward(self, gy:np.ndarray)->Union[tuple[np.ndarray,...],np.ndarray]:
        x,w = self.inputs
        gw = (gy*self.rms_x).sum(axis=tuple([i for i in range(x.ndim-1)]))
        gx = gy *w*self.rms_inv-x*(self.rms_inv**3)*((gy*w*x).sum(axis=x.ndim-1,keepdims=True)/x.shape[-1])
        return gx ,gw

def rms_norm(x,w,eps=1e-6):
    return RMSNormFunction(eps=eps)(x,w)

class RMSNorm(Layer):
    def __init__(self,hidden_size:int,eps:float = 1e-6):
        super(RMSNorm,self).__init__()
        self.weight = Parameter(np.ones(hidden_size),'weight')
        self.epsilon = eps
    
    def forward(self,x):
        return rms_norm(x,self.weight,eps=self.epsilon)



#dataclass
class LLaMaArgs:
    vocab_size :int = 64783
    num_layers: int = 12
    hidden_size :int = 1024
    num_heads:int = 8
    head_dim:int = 128
    num_key_value_heads:int = 8
    attention_bias:bool=False
    weight_share:bool=True
    rope_type: Literal['Llama','HF']='Llama'
    rope_theta:float = 10000.0
    enable_kv_cache:bool = True
    ffn_intermediate_size:int = 2752
    ffn_bias:bool = False
    max_len :int = 1024
    rms_eps:float = 1e-5
    dropout_ratio:float = 0.0
    max_batch_size:int =1 
    use_gpu :bool =True

baby_llama_zh = LLaMaArgs(
    vocab_size=64783,
    num_layers=12,
    hidden_size=1024,
    num_heads=8,
    head_dim=128,
    num_key_value_heads=8,
    attention_bias=False,
    weight_share=True,
    rope_type='Llama',
    rope_theta=10000.0,
    enable_kv_cache=True,
    ffn_intermediate_size=2752,
    ffn_bias=False,
    max_len=1024,
    rms_eps=1e-5,
    dropout_ratio=0.0,
    max_batch_size=1,
    use_gpu=True,
)

if __name__ =='__main__':
    np.random.seed(3996)

    # baby llama 用的是chatglm2-6b的分词器
    model_dict_baby_llama_zh = {
        'args': baby_llama_zh,
        'weight_path': '',
        'tokenizer_path':'',
    }
    model_dict = model_dict_baby_llama_zh
