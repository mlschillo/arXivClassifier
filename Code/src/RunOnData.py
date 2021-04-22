""" Do an experiment """
import torch
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

import TrainModels as TM

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class EvaluationCase():
    """ This class run models on test sets and gives various results.  Takes
    exactly two models to compare on any number of test sets.  Also needs the
    dictionarys for how each model labels categories, and the two models must
    agree on labeling.  Also takes the test data loaders from the model training
    run to compare with other test sets."""
    def __init__(self,
                 test_dataset_names,
                 model_names,
                 model_dicts,
                 model_test_dls,
                 data_path,
                 model_path
                 ):
        self.test_dataset_names = test_dataset_names
        self.model_names = model_names
        self.model_dicts = model_dicts
        self.model_test_dls = model_test_dls
        self.data_path = data_path
        self.model_path = model_path

    def load_data_and_models(self):
        dfs = [pd.read_csv(self.data_path + 'arxiv_' + item + '.csv')
               for item in self.test_dataset_names]
        if self.model_dicts[0] == self.model_dicts[1]:
            categories = list(self.model_dicts[0].keys())
        else:
            print(f'frist dict = {self.model_dicts[0]}')
            print(f'second dict = {self.model_dicts[1]}')
            raise ValueError('Models to compare do not have the same category keys')

        for df in dfs:
            df['local_cat_int'] = df.category.apply(lambda x: self.model_dicts[0][x])

        models=[]
        for name in self.model_names:
            mod = TM.CategoryClassifier(len(categories))
            mod.load_state_dict(torch.load(self.model_path + 'arxiv_' + name
                       + '_best_model_state.bin'))
            models.append(mod.to(device))

        loaders = [self.model_test_dls[0]] \
                  + [TM.MakeDataloader(df, test_only=True).create_data_loaders() for df in dfs] \
                  + [self.model_test_dls[1]]
        print(f'dataset sizes = {[len(dl.dataset) for dl in loaders]}')
        return models, loaders

    def eval_model(self, model, data_loader, device, n_examples):
        model = model.eval()

        abstract_texts = []
        predictions = []
        prediction_probs = []
        real_values = []
        cross_lists = []
        correct_predictions = 0

        with torch.no_grad():
            for d in data_loader:
                atexts = d["abstract_text"]
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)
                xlists = d['cross_list_cats']

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask)

                _, preds = torch.max(outputs, dim=1)
                probs = F.softmax(outputs, dim=1)
                abstract_texts.extend(atexts)
                cross_lists.extend(xlists)
                predictions.extend(preds)
                prediction_probs.extend(probs)
                real_values.extend(targets)

                correct_predictions += torch.sum(preds == targets)
                acc = correct_predictions.double() / n_examples


        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()

        return [acc.item(), abstract_texts, predictions, prediction_probs,
                real_values, cross_lists]

    def get_results_multicat(self):
        all_info = [[], []]
        models, loaders = self.load_data_and_models()
        for m in range(len(models)):
            for i in range(len(loaders)):
                results = self.eval_model(
                    models[m],
                    loaders[i],
                    device,
                    len(loaders[i].dataset))

                all_info[m].append(results)

        accsing = [[self.single_cat_acc(x[y][5], x[y][4], x[y][2])
                    for y in range(len(loaders))]
                   for x in all_info]

        acc = [[x[y][0] for y in range(len(loaders))]
               for x in all_info]

        fig = plt.figure()
        year_labels = ['mod_1'] + self.test_dataset_names + ['mod_2']
        sns.set_style("whitegrid")
        plt.plot(accsing[0], ':b', marker='o',
                 label='single cat ' + self.model_names[0], linewidth=3, markersize=10)
        plt.plot(accsing[1], ':r', marker='D', markersize=10,
                 label='single cat ' + self.model_names[1], linewidth=3)

        plt.plot(acc[0], '-b', marker='o', label=self.model_names[0], linewidth=2,
                 markersize=10)
        plt.plot(acc[1], '-r', marker='D', markersize=10, label=self.model_names[1],
                 linewidth=2)

        plt.xticks(ticks=range(len(loaders)), labels=year_labels,
                   fontsize=14)
        plt.yticks(fontsize=14)
        plt.title('Accuracy', fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                   prop={'size': 15})
        plt.savefig(self.model_path +self.model_names[0]+self.model_names[1]+ 'acc_results.png', bbox_inches='tight')

        accuracy_per_cat = [
            [self.cat_acc(x[y][4], x[y][2]) for y in range(len(loaders))]
            for x in all_info]
        cat_accuracy_permodel = [
            [[accuracy_per_cat[m][y][j] for y in range(len(loaders))] for j in
             range(len(self.model_dicts[0].keys()))] for m in range(2)]

        nplts = math.ceil(len(cat_accuracy_permodel[0]) / 3)
        marks = ['^', 'o', 's']
        with sns.axes_style('white'):
            sns.despine()
            palette = iter(sns.color_palette('tab10', 14))
            f = plt.figure(figsize=(10, 10))

            plt.tick_params(bottom=False, top=False, left=False,
                            labelleft=False, labelbottom=False)
            for pos in ['right', 'top', 'bottom', 'left']:
                plt.gca().spines[pos].set_visible(False)

            plt.title('Subfield accuracy of early model', fontsize=16)
            nrow = math.ceil(0.5 * nplts)
            ncol = 1 if nplts == 1 else 2
            gs = f.add_gridspec(nrow, ncol)
            for p in range(nplts):
                m=0
                ax = f.add_subplot(gs[math.floor(0.5 * p), p % 2])
                for i in range(3 * p, 3 * p + 3):
                    if i > len(cat_accuracy_permodel[0]) - 1 : break
                    plt.plot(cat_accuracy_permodel[0][i], ':', marker=marks[m],
                             markersize=10,
                             label=list(self.model_dicts[0].keys())[
                                 list(self.model_dicts[0].values()).index(i)],
                             linewidth=2, color=next(palette))
                    plt.ylim(.45, 1)
                    plt.legend(fontsize=14)
                    plt.xticks(ticks=range(len(loaders)), labels=year_labels,
                               fontsize=14)
                    m += 1

        plt.savefig(self.model_path + self.model_names[0] + self.model_names[
            1] + 'cat_acc_results.png', bbox_inches='tight')

        return acc, accsing, accuracy_per_cat, cat_accuracy_permodel

    def single_cat_acc(self, y_xlists, y_test, y_pred):
        right_tensor = y_test == y_pred
        right = list(right_tensor.numpy())
        wrong = list(~right_tensor.numpy())

        right_y_xlists = [y_xlists[i] for i in range(len(y_xlists)) if
                          right[i]]
        wrong_y_xlists = [y_xlists[i] for i in range(len(y_xlists)) if
                          wrong[i]]

        right_cros_list_len = [0 if len(y) == 2 else 1 for y in right_y_xlists]
        wrong_cros_list_len = [0 if len(y) == 2 else 1 for y in wrong_y_xlists]

        acc_sing = right_cros_list_len.count(0) / (
                    right_cros_list_len.count(0) + wrong_cros_list_len.count(
                0))

        return acc_sing


    def cat_acc(self, y_test, y_pred):
        yt = y_test.numpy()
        yp = y_pred.numpy()
        right = yt == yp

        categories = list(self.model_dicts[0].keys())
        category_acc = []
        for i in range(len(categories)):
            booli = [i == y for y in yt]
            if sum(booli) == 0:
                category_acc.append(1)
                continue
            cat_right = sum(
                [1 if booli[j] and right[j] else 0 for j in range(len(booli))])
            acc = cat_right / sum(booli)
            category_acc.append(acc)

        return category_acc


    def get_results_sincat(self):
        all_info = [[], []]
        models, loaders = self.load_data_and_models()
        for m in range(len(models)):
            for i in range(len(loaders)):
                results = self.eval_model(
                    models[m],
                    loaders[i],
                    device,
                    len(loaders[i].dataset))

                all_info[m].append(results)

        year_labels = [self.model_names[0][:-len('_sincat')]] \
                      + [item[:-len('_sincat')] for item in self.test_dataset_names] \
                      + [self.model_names[1][:-len('_sincat')]]

        accuracy_per_cat = [
            [self.cat_acc(x[y][4], x[y][2]) for y in range(len(loaders))]
            for x in all_info]
        cat_accuracy_permodel = [
            [[accuracy_per_cat[m][y][j] for y in range(len(loaders))] for j in
             range(len(self.model_dicts[0].keys()))] for m in range(2)]

        nplts = math.ceil(len(cat_accuracy_permodel[0]) / 3)
        nrow = math.ceil(0.5 * nplts)
        ncol = 1 if nplts == 1 else 2
        marks = ['^', 'o', 's']
        with sns.axes_style('white'):
            sns.despine()
            palette = iter(sns.color_palette('tab10', 14))
            f = plt.figure(figsize=(10, nrow*5))

            plt.tick_params(bottom=False, top=False, left=False,
                            labelleft=False, labelbottom=False)
            for pos in ['right', 'top', 'bottom', 'left']:
                plt.gca().spines[pos].set_visible(False)

            plt.title('Subfield accuracy of early model', fontsize=16)

            gs = f.add_gridspec(nrow, ncol)
            for p in range(nplts):
                m=0
                ax = f.add_subplot(gs[math.floor(0.5 * p), p % 2])
                for i in range(3 * p, 3 * p + 3):
                    if i > len(cat_accuracy_permodel[0]) - 1 : break
                    plt.plot(cat_accuracy_permodel[0][i], ':', marker=marks[m],
                             markersize=10,
                             label=list(self.model_dicts[0].keys())[
                                 list(self.model_dicts[0].values()).index(i)],
                             linewidth=2, color=next(palette))
                    plt.ylim(.45, 1)
                    plt.legend(fontsize=14)
                    plt.xticks(ticks=range(len(loaders)), labels=year_labels,
                               fontsize=14)
                    m += 1

        plt.savefig(self.model_path + self.model_names[0] + self.model_names[
            1] + 'sincat_acc_results.png', bbox_inches='tight')

        return accuracy_per_cat, cat_accuracy_permodel

