## BERT for Generating Poetry

#### 1. 介绍
- 这里我们使用HugginFace上预训练的[bert-base-chinese](https://huggingface.co/bert-base-chinese)来训练一个古诗生成模型
- 数据集使用GitHub上开源的[全唐诗](https://github.com/chinese-poetry/chinese-poetry/tree/master/%E5%85%A8%E5%94%90%E8%AF%97)，将数据集下载到`dataset`文件夹中并将`全唐诗`重命名为`json`，我们只需要用到诗集数据所以部分文件是用不到的，删掉下列json文件
```text
authors.song.json
authors.tang.json
表面结构字.json
```
- 然后做数据处理，代码详见[process.py](process.py)，注意将main中的路径替换为你自己的数据集所在路径，这里只保留长度小于等于128的诗，运行代码`python process.py`即可。处理完毕之后`dataset`看起来如下（总数据量大概有29w条）:
```text
dataset
└── poetry
    ├── json
    └── poetry.txt
```

#### 2. 加载数据、搭建模型、进行训练
- Bert原本是双向语言模型，但由于其已经进行过预训练所以包含了很多知识，我们也可以将其作为decoder来进行文本生成
- 这里我是将HuggingFace中的模型克隆到了本地，所以你也需要克隆到本地或者说换一种加载预训练模型的方式：
```python
from transformers import AutoTokenizer, BertModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")
```
- 在`run.py`中更改上述提到的部分即可直接开始训练，执行命令`python run.py --mode train`
- 这里总共训练了10轮，使用NVIDIA GeForce RTX 3090单卡训练一轮时间开销大概在22min，当然只训两轮你也可以把模型提出来看看效果

#### 3. 生成诗词
- 代码支持随机生成诗词，也可以给定每句诗的首字生成藏头诗，可以自行设置`temperature`和`top p`等参数，接下来看看效果（注意下面的展示是自己手动换行的）
- 随机生成执行`python run.py --mode test`
```text
薄宦侵寻懒作劳，不如归去问陶陶。
新秋鸡唱风前急，野水云浮月下高。
行役不知今日是，登临聊复旧时劳。
新诗如许当年意，曾和商霖未肯劳。
```
- 生成藏头诗执行`python run.py --mode test --category acrostic --head 坤坤快跑`
```text
坤元受运应阴阳，坤德回还象纬昌。
快雨渐催来岁晚，跑云还许好时长。
```
- 支持设置`temperature`和`top_p`，执行`python run.py --mode test --category acrostic --head 只因太美 --temperature 0.7 --top-p 0.8`
```text
只因诸葛太平年，因得仙经与世传。太守不知天下事，美哉何似此中仙。
```

#### 4. 参考
- [用BERT实现自动写诗](https://aistudio.baidu.com/aistudio/projectdetail/1689372)
