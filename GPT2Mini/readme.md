## Pretrained Language Model based on GPT-2 Architecture

#### 1. Introduction
- Here we will train a auto-regressive language model based on GPT-2 Architecture(Chinese)
- We collect the pretrain corpus by crawling specified web page
- The pretrained model can directly generate text

#### 2. Collect data and Pre-processing
- We use `scrapy` to crawl the data, all the prose data will be used for pretraining, thanks [新散文网](hhttps://www.xinsanwen.cn/), [半壁江](http://read.banbijiang.com/) for allowing to crawl
```shell
cd textcrawler
scrapy crawl prose   # crawl prose, about 476M, mainly used for pretrain
scrapy crawl prose2   

mv prose.json ../../dataset/corpus   # move to the specified location
mv prose2.json ../../dataset/corpus  
```
- Then we merge all the prose data and simply process the data to fit context size 1024 tokens, execute `python process.py`
- As usual, we have to do build our own tokenizer. But our corpus is too small so here we use `BERT-Chinese tokenizer`
- After all operations, my folder looks like this:
```text
dataset
└── corpus
    ├── prose.json
    ├── prose2.json
    ├── merge.json
    └── processed.json
```

#### 3. Load dataset, Build model, Train model
- For this task, we build a Transformer model, I implement the architecture based on GPT-2 architecture
- All the hyperparameters are nearly the same as [BERT base](https://arxiv.org/pdf/1810.04805.pdf), only few modifications like pre-norm, gelu activation, you can check `run.sh` for more details
- You can simply regard this model as BERT, but this is a unidirectional language model, using the autoregressive method to pretrain from scratch
- Here I use 2 NVIDIA GeForce RTX 3090 to train, and precision is `fp16` each epoch will cost about 70 minutes
- Since we use pre-norm, actually I find there's no need to do warm-up
- If you want to train from scratch, you don't have to modify anything, make sure you have at least 2 GPU, simply execute `sh run.sh`
- If you just want to use the model, simply execute `python run.py --mode test`

#### 4. Generate Text
- Here we support top_p decode, let's see some generated text, you can set the temperature and top_p sampling rate by executing `python run.py --mode test --temperature 1.0 --top-p 0.9` 

**Prompts**
```text
怀揣着梦想
起风了
```
**Generations**
```text
怀揣着梦想，回到故乡的怀抱，故乡的童年在远方隐隐的召唤。在睡梦中不断的寻找中，看到故乡的青山绿水、白云红柳
、花朵鲜花尤其是在白发苍苍的老人身上，就有了童年的影子，无论是白发他心中永远的牵挂。他总是怀揣着对故乡的情
愫，在梦中回到故乡，踏遍故乡的山山水水，去寻找寻找我心中那个永远的故乡，去聆听岁月的钟声，去感受一份飘零的
情感，寻找一份爱的寄托。我鸟语花香的春日。每次回到故乡，回到故乡的情结中，我的心总是会像春水一样柔柔荡漾，
时时泛起心中的涟漪，那是儿时常玩耍的地方，那是父母、祖祖辈辈抚育我长大的地方，我的心中总会涌起一种异样的情
感，为异乡的亲人送去些许的温暖，他们或许不知道我是这样的心情的，但他们不慌不忙的脚步，让我总是感觉到那里蕴
藏着故乡的淳朴、善良和感恩，他们总是

起风了，雪花纷纷扬扬地飘扬，那漫天飘洒的洁白像是刚刚洗过一样，洁白无瑕，简单的美丽犹如在清明节前的第一场雪，
更是让人爱不释手，该落的落，该凝结的凝结成一簇簇热烈奔放的火焰，绪飘飞，心中顿时生出无限的欢乐和自豪，心中
有了一种说不出的陶醉和陶醉。你看，不远处，几棵枯死的杏树在寒风中瑟瑟发抖，那样的树叶不正在风中瑟瑟发抖吗？
不，这是冬天的雪，那样的阵阵飘洒，犹如大地的绿叶在飘飘洒洒，而那铺天盖地的绿叶，犹如万马奔腾，更像是绿色地
毯上的毯子，温暖舒适，美丽无限。那飘飞的绿叶，犹如飘舞的绿云，尽管无法形成千条万条的图案，但那潇洒飘飞的美
丽，犹如灵动的精灵，让人遐想万千，万千的心灵在那美妙的画面中寻觅和感动。那花儿们似乎并没有感觉到，在那无限
飘飘洒洒的飘飞的绿叶和飘飞的雪花中，那
```

- I think the quality is not too bad, there might be several reasons:(1)the dataset is too small (2)scheduler might be not that unreasonable, I plan to leave these work to followers

#### 5.References
- [Improving Language Understanding
by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
