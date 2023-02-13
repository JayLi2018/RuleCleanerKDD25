import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, BertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
import time
import sklearn.metrics as metrics


def predict_transformers(texts, lfrs, labels, classes, dataset, batch_size, model, tokenizer):
    dataset = TextLabellingDataset((texts, lfrs, labels), classes, tokenizer, max_len=512)
    iterator = data.DataLoader(dataset=dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=1,
                               collate_fn=TextLabellingDataset.pad)

    Y_hat = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            _, x, _, mask, y, _ = batch
            _y = y
            
            loss, logits = model(x, 
                         attention_mask=mask,
                         labels=_y)
            logits = model(x, attention_mask=mask, labels=None)
            y_hat = logits[0].argmax(-1)
            Y_hat.extend(y_hat.tolist())
    return np.array(Y_hat)


def train(model, train_set, optimizer, scheduler=None, batch_size=32):
    """Perfrom one epoch of the training process.
    """
    iterator = data.DataLoader(dataset=train_set,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=1,
                               collate_fn=TextLabellingDataset.pad)

    model.train()
    total_train_loss = 0
    for i, batch in enumerate(iterator):
        # for monitoring
        words, x, tags, mask, y, seqlens = batch
#         print(len(words), len(x), len(tags), len(y), len(seqlens))
        _y = y

        # forward
        optimizer.zero_grad()         
        loss, logits = model(x, 
                     attention_mask=mask,
                     labels=_y)
#         print(loss, logits)
        total_train_loss += loss.item()

        # back propagation    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        if i%100 == 0: # monitoring
            print(f"step: {i},  loss: {loss.item()}")
            del loss
     # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataset)            
    print("  Average training loss: {0:.2f}".format(avg_train_loss))


def eval_bert_classification(model, dataset, batch_size):
    iterator = data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=1,
                           collate_fn=TextLabellingDataset.pad)

    Y = []
    Y_hat = []
    loss_list = []
    total_size = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            _, x, _, mask, y, _ = batch
            _y= y

            loss, logits = model(x, 
                         attention_mask=mask,
                         labels=_y)
            y_hat = logits.argmax(-1)
            loss_list.append(loss.item() * y.shape[0])
            total_size += y.shape[0]

            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    num_classes = len(set(Y))
    # Binary classification
    if num_classes <= 2:
        accuracy = metrics.accuracy_score(Y, Y_hat)
        precision = metrics.precision_score(Y, Y_hat)
        recall = metrics.recall_score(Y, Y_hat)
        f1 = metrics.f1_score(Y, Y_hat)
        print("accuracy=%.3f"%accuracy)
        print("precision=%.3f"%precision)
        print("recall=%.3f"%recall)
        print("f1=%.3f"%f1)
        print("======================================")
        return accuracy, precision, recall, f1, loss
    else:
        accuracy = metrics.accuracy_score(Y, Y_hat)
        f1 = metrics.f1_score(Y, Y_hat, average='macro')
        precision = recall = accuracy # We might just not return anything
        print("accuracy=%.3f"%accuracy)
        print("macro_f1=%.3f"%f1)
        print("======================================")
        return accuracy, f1, loss

    
def train_model_for_text_labelling(model, train_dataset, valid_dataset=None, test_dataset=None, batch_size=8, n_epochs=5, save_model=False):
    optimizer = AdamW(bert_model.parameters(), lr = 3e-5, eps = 1e-8)
    from transformers import get_linear_schedule_with_warmup

    num_steps = (len(train_dataset) // batch_size) * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    best_dev_f1 = best_test_f1 = 0.0
    model.train()
    for epoch in range(1, n_epochs+1):
        t0 = time.time()

        train(bert_model,
              train_dataset,
              optimizer,
              scheduler=scheduler,
              batch_size=batch_size)


        print(f"=========eval at epoch={epoch}=========")
        if valid_dataset is not None:
            accuracy, precision, recall, f1, loss = eval_bert_classification(model, valid_dataset, batch_size)
            if f1 > best_dev_f1:
                best_dev_f1 = f1
                if save_model:
                    torch.save(model.state_dict(), 'best_dev.pt')

        if test_dataset is not None:    
            accuracy, precision, recall, f1, loss = eval_bert_classification(model, test_dataset, batch_size)
            print(accuracy, precision, recall, f1, loss)
            if f1 > best_test_f1:
                best_test_f1 = f1
                if save_model:
                    torch.save(model.state_dict(), 'best_test.pt')

        training_time = str(time.time() - t0)
        print("")
        print("  Training epcoh took: {:}".format(training_time))



def ModelForTextLabelling(model_type='distilbert', num_labels=2, model_path=None, lfs=[], lf_vocab=[], res2tag={}, save_model=False):
    if model_type == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',  
            num_labels = num_labels,
            output_attentions = False,
            output_hidden_states = False
        )
    elif model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = num_labels,
            output_attentions = False,
            output_hidden_states = False
        )
        
    # add the labelling result of each LF to the vocabulary
    for i in range(len(lfs)):    
        print(len(tokenizer))
        tokenizer.add_tokens(["LF{}:{}".format(str(i), res2tag[j]) for j in lf_vocab])
        print(len(tokenizer))
        model.resize_token_embeddings(len(tokenizer)) 
        
    if model_path is None:
        train_model_for_text_labelling(model, train_dataset, valid_dataset=None, test_dataset=None, batch_size=8, n_epochs=5, save_model=False)
    else:
        model.load_state_dict(torch.load(model_path))

    return model, tokenizer


    