from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


allcat = ['hep-ph', 'math', 'physics', 'cond-mat', 'gr-qc', 'astro-ph', 'hep-th',
 'hep-ex', 'nlin', 'q-bio', 'quant-ph', 'cs', 'nucl-th', 'math-ph', 'hep-lat',
 'nucl-ex', 'q-fin', 'stat', 'eess', 'econ', 'adap-org', 'alg-geom', 'chao-dyn',
 'cmp-lg', 'comp-gas', 'dg-ga', 'funct-an', 'patt-sol', 'q-alg', 'solv-int']

cat_dict = dict(zip(allcat, range(len(allcat))))


class ArXivAbstractDataset(Dataset):
    """ This class encodes the abstract into the input for pre-trained BERT
        and returns a dictionary that can be fed to the torch Dataloader
    """
    def __init__(self, texts, xlists, targets, tokenizer, max_len=450):
        self.texts = texts
        self.xlists = xlists
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        croscat = self.xlists[item]
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'abstract_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long),
            'cross_list_cats': croscat
        }

class MakeDataloader:

    def __init__(self, df, bert_model_name='bert-base-cased', test_only=False,
                 batch_size=10, max_len=450, rand_seed=66):
        self.df = df
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.test_only = test_only
        self.batch_size = batch_size
        self.max_len = max_len
        self.rand_seed = rand_seed

        if test_only:
            self.df_test = df
            self.ds_test = ArXivAbstractDataset(
            texts=self.df_test.abstract.to_numpy(),
            xlists=self.df_test.cross_lists.to_numpy(),
            targets=self.df_test.local_cat_int.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len
            )
        else:
            bal_targets = df.local_cat_int
            self.df_train, self.df_test = train_test_split(df, test_size=0.1,
                            random_state=self.rand_seed, stratify=bal_targets)
            bal_targets2 = self.df_test.local_cat_int
            self.df_val, self.df_test = train_test_split(self.df_test, test_size=0.5,
                            random_state=self.rand_seed, stratify=bal_targets2)

            self.ds_train = ArXivAbstractDataset(
                texts=self.df_train.abstract.to_numpy(),
                xlists=self.df_train.cross_lists.to_numpy(),
                targets=self.df_train.local_cat_int.to_numpy(),
                tokenizer=tokenizer,
                max_len=max_len
            )
            self.ds_val = ArXivAbstractDataset(
                texts=self.df_val.abstract.to_numpy(),
                xlists=self.df_val.cross_lists.to_numpy(),
                targets=self.df_val.local_cat_int.to_numpy(),
                tokenizer=tokenizer,
                max_len=max_len
            )
            self.ds_test = ArXivAbstractDataset(
                texts=self.df_test.abstract.to_numpy(),
                xlists=self.df_test.cross_lists.to_numpy(),
                targets=self.df_test.local_cat_int.to_numpy(),
                tokenizer=tokenizer,
                max_len=max_len
            )


    def create_data_loaders(self):
        test_loader = DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            num_workers=2)
        if not self.test_only:
            train_loader = DataLoader(
                self.ds_train,
                batch_size=self.batch_size,
                num_workers=2)
            val_loader = DataLoader(
                self.ds_val,
                batch_size=self.batch_size,
                num_workers=2)

            return train_loader, val_loader, test_loader
        else:
            return test_loader




class CategoryClassifier(nn.Module):

    def __init__(self, n_classes, dropout=0.5, bert_model_name='bert-base-cased'):
        super(CategoryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.drop = nn.Dropout(p=dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)

def make_model(data_loader, n_classes, dropout=0.5, bert_model_name='bert-base-cased'):
    model = CategoryClassifier(n_classes=n_classes, dropout=dropout, bert_model_name=bert_model_name)
    model = model.to(device)
    data = next(iter(data_loader))
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    F.softmax(model(input_ids, attention_mask), dim=1)
    return model


def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

class TrainingRun():
    def __init__(self, training_set_name, data_path, model_path, epochs=10, LR=2e-5):
        self.training_set_name = training_set_name
        self.model_path = model_path
        self.data_path = data_path
        self.epochs = epochs
        self.LR = LR


    def train_model(self, model, train_dl, val_dl):
        mod_path = self.model_path
        epochs = self.epochs
        learn_rate = self.LR

        history = defaultdict(list)
        best_accuracy = 0

        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = AdamW(model.parameters(), lr=learn_rate, correct_bias=False)
        total_steps = len(train_dl) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        count = 0
        for epoch in range(epochs):
            count += 1
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)

            train_acc, train_loss = train_epoch(
                model,
                train_dl,
                loss_fn,
                optimizer,
                device,
                scheduler,
                len(train_dl.dataset)
            )

            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = eval_model(
                model,
                val_dl,
                loss_fn,
                device,
                len(val_dl.dataset)
            )
            if epoch == 0:
                val_loss0 = val_loss

            print(f'Val   loss {val_loss} accuracy {val_acc}')

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                count = 0
                print('saving model')
                torch.save(model.state_dict(), mod_path + 'arxiv_'
                           + self.training_set_name + '_best_model_state.bin')
                best_accuracy = val_acc
            if count > 4:
                print('Overfitting')
                break

        return history

    def data_to_model(self):
        mod_path = self.model_path
        dat_path = self.data_path
        train_set_name = self.training_set_name
        df1 = pd.read_csv(dat_path + 'arxiv_' + train_set_name + '.csv')
        print(f'there are {df1.shape[0]} total samples')

        categories = df1.category.unique()
        local_cat_dict = dict(zip(categories, list(range(len(categories)))))
        df1['local_cat_int'] = df1.category.apply(lambda x: local_cat_dict[x])

        dl_train, dl_val, dl_test = MakeDataloader(df1).create_data_loaders()

        model = make_model(data_loader=dl_train, n_classes=len(categories))
        history = self.train_model(model=model, train_dl=dl_train, val_dl=dl_val)

        fig = plt.figure()
        plt.plot(history['train_acc'], label='train accuracy')
        plt.plot(history['val_acc'], label='validation accuracy')
        plt.title('Training history')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylim([0, 1])
        plt.savefig(mod_path + 'acc_hist_' + train_set_name + '.png')

        model = CategoryClassifier(len(categories))
        model.load_state_dict(torch.load(mod_path + 'arxiv_' + train_set_name
                                         + '_best_model_state.bin'))
        model = model.to(device)

        test_acc, _ = eval_model(
            model,
            dl_test,
            nn.CrossEntropyLoss().to(device),
            device,
            len(dl_test.dataset)
        )

        print(f'accuracy on test set = {test_acc.item()}')
        print(f'test set has {len(dl_test.dataset)} abstracts equally distributed among the'
              f' {len(categories)} categories: {categories}')

        return local_cat_dict, dl_test

