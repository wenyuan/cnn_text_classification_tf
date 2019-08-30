# CNN Text Classification in Tensorflow
> TensorFlow中使用CNN实现英文文本分类（代码注释）

Forked from [dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)
主要是解读以上开源项目，详情可以参见[CNN实现英文文本分类](https://www.wenyuanblog.com/blogs/tensorflow-cnn-english-text-classification-implement.html)


## Requirements
- Python 3
- tensorflow==1.5.0
- numpy==1.16.3


## Training
打印参数：

```bash
./train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement

```

Train:

```bash
./train.py
```

## Evaluating

```bash
./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.<br/>
用训练输出的目录替换检查点（checkpoint）目录。如果要使用自己的数据，就更改`eval.py`中的 **checkpoint_dir** 参数。
