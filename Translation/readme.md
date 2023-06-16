## Transformer for Neural Machine Translation(En-Zh)

#### 1. Introduction
- Here we will train a Transformer Seq2Seq model to translate English to Chinese
- The translation dataset can be obtained from [nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus), you only have to download `translation2019zh.zip` then execute the following command to extract data:
```shell
unzip translation2019zh.zip -d en-zh
```
- Here I download the archive to `dataset/nmt`, then my folder looks like this:
```text
dataset
└── nmt
    └── en-zh
        ├── translation2019zh_train.json
        └── translation2019zh_valid.json
```

#### 2. Pre-processing
- For this task we have to do our own BPE, and we use Byte-Level BPE to get subword, check the script `process.py` 
- If you want to train from scratch or build your own subword model, you can modify `process.py` and execute it
- If you just want to simply use the model, you don't have to execute the script because I've integrated it into model

#### 3. Load dataset, Build model, Train model
- For this task, we build a Transformer Seq2Seq model, I implement the architecture by myself instead of using `torch.nn.Transformer`
- All the hyperparameters are nearly the same as paper [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf), only few modifications like pre-norm, byte-level BPE, you can check `run.sh` for more details
- Here I use 2 NVIDIA GeForce RTX 3090 to train, and precision is `fp16` each epoch will cost about 2 hours
- If you want to train from scratch, you don't have to modify anything, make sure you have at least 2 GPU simply execute `sh run.sh`
- If you just want to use the model, simply execute `python run.py --mode test`

#### 4. Translate
- Here we support batch greedy decode and batch beam search(still under development~)
- Given the English sentence and translate to Chinese, let's see some examples

**Source Sentences**
```text
I want to train a neural machine translation model
print this paper and send it to Bob
have you ever used my machine translation model?
It's sunny today, let's go to the beach to have a happy holiday!
```
**Target Sentences**
```text
我想训练一个神经网络机器翻译模型
打印这篇文章，然后发送到鲍勃
你曾用过我的机器翻译模型吗？
今天阳光明媚，我们一起去海边过个愉快的假日吧！
```

- There are two areas that can be optimized: (1)I forgot to convert all text to lowercase (2)Maybe 30 epochs is not enough cause I find loss is still declining, but I don't have enough resources to do this and the effect is not bad, so I keep it
- Beam Search is still under development cause its complexity, wait for it

#### 5.References
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [Transformer Architecture](https://blog.csdn.net/zhaohongfei_358/article/details/126019181)
- [Transformer Implementation Tutorial](https://zhuanlan.zhihu.com/p/347709112)
- [从零详细解读什么是Transformer模型](https://juejin.cn/post/7236634856083472445)