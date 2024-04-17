# Sentiment-Analysis-For-IMDB-Reviews

Main goal of this project is to make sentiment analysis to IMDB reviews dataset. BERT model is used for the task. Codes are written with PyTorch framework.

## Methodology

### Data Preprocessing

For data preprocessing, firstly applied drop_duplicates method to the data since there are some duplicate values in the 'review' columns. Then applied a function to clean HTML tags, URLs, emojis, punctuations etc. from the text and also fix some pronouns and nots, for example "don't" ---> 'do not' or "you're" ---> "you are".

After cleaning the text data then encoded the labels as 0 for 'negative' labels and 1 for 'positive' labels. Then generated train-validation splits with ratios %85-%15 from ~90% of the whole dataset and left the remaining part for the testing.

At the last step, tokenized the text data by BERT tokenizer with parameters,

    max_len = 512 # max tokens that BERT can work with
    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
    truncation = True,           # Pad & truncate all sentences.
    padding = True,
    return_attention_mask = True,   # Construct attention. masks.
    return_token_type_ids=False
        
then extraxt the input data and attention mask then converted them to PyTorch tensor. Finally, generated a dataloader with PyTorch's DataLoader library with batch size of 32.

### Models, Configurations and experiments

For each experiments, BERT ("bert-base-uncased") is used as the base and added some linear layers end of it. First of all, with small epoch size, decided whether to clean stopwords or not. After running training sessions with both, data with stopwords performed better therefore did not clean the stopwords for next experiments. 

After deciding not to clean stopwords, used differents setups and configurations for experiments. For example, used two different learning rates with values 2e-5 and 1e-3. Added two different learning reate scheduler as "get linear schedule with warmup" and "reduce on plateau". Also, added more linear layers after BERT layer and generated different models. Applied early stopping technique etc. Finally, got the best result with the following configurations;

    batch_size = 32
    epochs = 30
    learnin_rate =  1e-3
    optimizer = AdamW
    early_stopping = 10
    loss_function = Binary Cross Entropy
    learning_rate_scheduler = get_linear_schedule_with_warmup (from Transformers library)


and the model;

    Input ---> BERT ---> linear_layer(768) ---> Dropout(0.2) ---> ReLU ---> linear_layer(512) ---> Sigmoid ---> Output

### Results 

Used F1 score as the metrcis for evaluation on the test set. Except the model and configurations that is shared above, F1 score was between 78% - 83% approximately. With the best setup, F1 score was 84% with the following confusion matrix:

![alt text](https://github.com/yunsemr/Sentiment-Analysis-For-IMDB-Reviews/image.jpg?raw=true)

    
