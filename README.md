# Sentiment-Analysis-For-IMDB-Reviews

Main goal of this project is to make sentiment analysis to IMDB reviews dataset. BERT model is used for the task. Codes are written with PyTorch framework. The notebook with the best resulted experiment is shared in the repository.

## Data

IMDB review dataset was used for the project. Dataset contains 50K reviews with labels "positive" and "negative". 

Dataset can be downloaded from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Methodology

### Data Preprocessing

For data preprocessing, firstly applied drop_duplicates method to the data since there are some duplicate values in the 'review' column. Then applied a function to clean HTML tags, URLs, emojis, punctuations etc. from the text and also fix some pronouns and nots, for example "don't" ---> 'do not' or "you're" ---> "you are".

After cleaning the text data then encoded the labels as 0 for 'negative' labels and 1 for 'positive' labels. Then, generated train-validation splits with ratios 85%-15% from ~90% of the whole dataset and left the remaining part for the testing.

After cleaning the text, tokenized the text data by BERT tokenizer with parameters,

    max_len = 512 # max tokens that BERT can work with
    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
    truncation = True,           # Pad & truncate all sentences.
    padding = True,
    return_attention_mask = True,   # Construct attention. masks.
    return_token_type_ids=False
        
then extracted the input data and attention mask and converted them to PyTorch tensor. Finally, generated a dataloader with PyTorch's DataLoader library with batch size of 32.


### Models, Configurations and Experiments

In the model, pre-trained BERT ("bert-base-uncased") is used as the base model and added some linear layers end of it for experiments. I used Binary Cross Entropy as loss function and AdamW as optimizer. Even if I trained the model with 10 epochs, I made the experiments with 30 epochs and applied early stopping with patience equals to 10.

For experiments, first of all, with small epoch size, decided whether to clean stopwords or not. After running training sessions with both settings, data with stopwords performed better therefore did not clean the stopwords for the next experiments. 

Different setups are used. At the beginning, I used two different learning rate values, 2e-5 and 1e-3. For both learning rate, I used the architecture with hidden sizes 768-512 and "get_linear_schedule_with_warmup" as learning rate scheduler. Since, model with 1e-3 performs better, I used that value in the next steps. 

Also, I worked with different learning rate schedulers. "get_linear_schedule_with_warmup" and "ReduceLROnPlateau" techniques are used. For both scheduler, I used the architecture with hidden sizes 768-512 and learning rate 1e-3. "get_linear_schedule_with_warmup" performed slightly better.

In another setups, changed the model architecure with different number of linear layers and hidden sizes. Three different architecture is used and these have 768, 768-512, 768-512-256-128-32 hidden sizes respectively in the linear layers. Also dropout with 0.2 drop rate and ReLU activation function is used in all architecures. For all these architectures, learning rate was 1e-3 and scheduler was "get_linear_schedule_with_warmup". Artchitecture with hidden size 768-512 performed slightly better in the experiments. Results and confusion matrix are shareed in the "Results" part.

In this part I only share the configurations of the best resulted experiments;

    batch_size = 32
    epochs = 30
    learnin_rate =  1e-3
    optimizer = AdamW
    early_stopping = 10
    loss_function = Binary Cross Entropy
    activation_function = ReLU
    drop_rate = 0.2
    learning_rate_scheduler = get_linear_schedule_with_warmup (from Transformers library)


and the architecture;

    Input ---> BERT ---> linear_layer(hidden_size = 768) ---> Dropout(drop_rate = 0.2) ---> ReLU ---> linear_layer(hidden_size = 512) ---> Sigmoid ---> Output

## Results 

For the test set 4582 reviews sample are used and F1 score is chosen as the evaluation metric. Same data preprocessing techniques are applied for the test set. Here is the results for some of the experiments:


Results with different learning rates:
| lr_rate/metrics  | Precision  | Recall | f1-score | Accuracy |
| :-----: | :-------------: | :-------------: | :------------: | :------------: |
| 2e-5 | 0.83  | 0.81  | 0.82 | 0.82 |
| 1e-3 | 0.84  | 0.83  | 0.84 | 0.84 |

2e-5             |  1e-3
:-------------------------:|:-------------------------:
![](https://github.com/yunsemr/Sentiment-Analysis-For-IMDB-Reviews/blob/main/IMDB_Sentiment_Analysis/cm_lr_2e-5.png?raw=true)  |  ![](https://github.com/yunsemr/Sentiment-Analysis-For-IMDB-Reviews/blob/main/IMDB_Sentiment_Analysis/confusion_matrix.png?raw=true)



Results with different hidden sizes: 
| Hidden Size/metrics  | Precision  | Recall | f1-score | Accuracy |
| :-----: | :-------------: | :-------------: | :------------: | :------------: |
| 768 | 0.83  | 0.77  | 0.80 | 0.79 |
| 768-512 | 0.84  | 0.83  | 0.84 | 0.84 |
| 768-512-256-128-32 | 0.79 | 0.85  | 0.82 | 0.83 |


768            |  768-512   |   768-512-256-128-32
:-------------------------:|:-------------------------: | :-----------------:
![](https://github.com/yunsemr/Sentiment-Analysis-For-IMDB-Reviews/blob/main/IMDB_Sentiment_Analysis/cm_768.png?raw=true)  |  ![](https://github.com/yunsemr/Sentiment-Analysis-For-IMDB-Reviews/blob/main/IMDB_Sentiment_Analysis/confusion_matrix.png?raw=true) | ![](https://github.com/yunsemr/Sentiment-Analysis-For-IMDB-Reviews/blob/main/IMDB_Sentiment_Analysis/cm768-512-256-128-32.png?raw=true)



Here is the best resulted confusion matrix, F1 score is 84%:

![alt text](https://github.com/yunsemr/Sentiment-Analysis-For-IMDB-Reviews/blob/main/IMDB_Sentiment_Analysis/confusion_matrix.png?raw=true)  

## Discussion and Future Improvement Suggestions

I benefitted from many other sentiment analysis projects and faced with some code or shape errors as classical during an AI prject. I solved these issues and debug the code by researching the error message and using Stackoverflow, PyTorch forum etc. 


One of the other problem during the project was limited GPU access therefore, I couldn't apply the all strategies in my mind. For example, when I increased the number of epoch from 10 to 30, F1 score was improved from ~80% to 84% and the validation loss was continuing to decrease on the last epoch also. Probably, increasing the epoch size from 30 to 50 or higher will improve the F1 score but I couldn't apply. Besides, cross validation technique would be applied to evaluate the results more accurate with different data samples. However, this takes a lot of time with limited GPU access too.


Also, one of the most creative solutions to improve F1 score was cropping the text data from middle with a token size which is able to work with BERT model and used the data I cropped from the original text. I read this solutions on internet. However, this method did not improve the results suprisingly. Actually, I understand the method that I read while writing these words. I guess, instead of using cropped data from the original text, truncating the data from the middle of the original text and use remaning part after truncation with a suitable size for BERT was the suggestion. Therefore, this method can be used for improvement of the results.


On the other hand, I just changed parameters such as learning rate by hand with a limited values therefore, hyperparameter tunining can be applied for better results. Also, using different and larger models such as larger BERT model ("bert-large-uncased") would be helpful since the F1 score is stuck in a range between 80%-84% during the experiments. Apart from these methods, using different train-validation-test split sizes, using more data or weight initilization techniques can be applied to improve the results.

## Conclusion

Aimed to make sentiment analysis for the IMDB reviews dataset. Preprocessed the dataset and cleaned from unnecessary elements for tokenization and tokenized by usind pre-trained BERT tokenizer. After data preprocessing, different configurations and architectures are used with pre-trained BERT model for training experiments and got 84% as the highest F1 score on the test set. Finally, mentioned about the problems during the project and some suggestions to improve the model performance are given.

## References

* https://www.kaggle.com/code/joydeb28/text-classification-with-bert-pytorch/notebook
* https://www.kaggle.com/code/angyalfold/hugging-face-bert-with-custom-classifier-pytorch#PyTorch-setup
* https://www.kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification#Loading-the-best-model
* https://github.com/Prerna5194/IMDB_BERT_TRANSFORMER


