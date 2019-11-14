Requirements
------------

Install requirements for computer vision experiments with pip:

```
pip install numpy tensorflow keras scipy pandas nltk pickle 
```


Depending on your Python installation you might want to use anaconda or other tools.


Experiments
-----------

Run models:

    ```
    cd Fasttext/CNN/BiLSTM/Transformer
    python3 train.py
    ```

This model is for mr, subj, cr, mpqa, sst2, TREC. The default is for TREC, and if you want to run this model on other dataset, you can:

    ```
    cd Fasttext/CNN/BiLSTM/Transformer
    python3 train.py --data mr/subj/cr/mpqa/sst2
    ```

You can use other classification dataset. Please put your dataset on dir data, and preprocess your data according to mr.