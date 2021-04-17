""" Functions and classes needed to make a dataset """

import dask.bag as db
import json
import pandas as pd
import textwrap
import threading



trim = lambda x: {'id': x['id'],
                  'category': x['categories'].split(' '),
                  'abstract': x['abstract'],
                  'year': x['versions'][0]['created'].split(' ')[3]}

first_v = lambda x: x['versions'][0]['created']

allcat = ['hep-ph', 'math', 'physics', 'cond-mat', 'gr-qc', 'astro-ph',
          'hep-th',
          'hep-ex', 'nlin', 'q-bio', 'quant-ph', 'cs', 'nucl-th',
          'math-ph', 'hep-lat',
          'nucl-ex', 'q-fin', 'stat', 'eess', 'econ', 'adap-org',
          'alg-geom', 'chao-dyn',
          'cmp-lg', 'comp-gas', 'dg-ga', 'funct-an', 'patt-sol',
          'q-alg', 'solv-int']

cat_dict = dict(zip(allcat, range(len(allcat))))


class data:
    """ For selecting loading and preparing data
        Example Use:
        my_data = DataSelection.data(years=2008, single_cat_subs=True)
        df = my_data.choose_cats()
        or to simply write df to csv:
        my_data.write_csv()

        Requires: access to "arxiv-metadata-oai-snapshot.json". If path to this
        file is not provided, assumption is that it's in current directory.
        -- the same path will be used to write the data to .csv

        Parameters:
        - years is a single int or a list of ints for years to look at
        - single_cat_subs: False takes all papers,
                           True only takes papers submitted to a single subfield
        - threshold: minimum number of papers for a category to be considered
        - n_percat_peryear: if this is set >0 then instead of using a threshold
          for a minimum #papers/cat/year, this fixed number will be sampled
        - subfields: default will keep all subfields that meet threshold of
          n_percat_peryear requirements, but if only a specific subfield(s) is
          desired, enter a list of strings e.g.: ['hep-th', 'hep-ex']
    """

    def __init__(self,
                 years,
                 single_cat_subs=False,
                 threshold=500,
                 n_percat_peryear=0,
                 subfields='all',
                 path='../../Data/',
                 extra_name='',
                 randomseed=666,
                 verbose=False):
        if isinstance(years, int):
            self.years = [years]
        elif isinstance(years, list):
            self.years = years
        else:
            raise ValueError(
                'Try again, years should be an int or list of ints')
        if any([y < 1997 for y in self.years]):
            raise ValueError(
                'ArXiv category conventions prior to 1997 not supported')
        self.single_cat_subs = single_cat_subs
        self.threshold = threshold
        self.n_percat_peryear = n_percat_peryear
        self.subfields = subfields
        self.path = path
        self.extra_name = extra_name
        self.randomseed = randomseed
        self.verbose = verbose
        self.info = 0
        self.df = self.get_data()

    def use_dask(self):
        location = self.path + "arxiv-metadata-oai-snapshot.json"
        info = db.read_text(location).map(json.loads)
        self.info = info

    def get_data(self):
        print('Using dask to load ArXiv metadata')
        thread = threading.Thread(target=self.use_dask())
        thread.start()
        thread.join()

        info = self.info

        years = self.years
        info_years = info.filter(
            lambda x: int(first_v(x).split(' ')[3]) in years).map(
            trim).compute()
        df = pd.DataFrame(info_years)

        df['category'] = df.category.apply(
            lambda x: [a.split('.')[0] for a in x])
        df['category'] = df.category.apply(
            lambda x: list(dict.fromkeys(x)))
        df['cross_lists'] = df.category.apply(
            lambda x: x[1:])
        df['category'] = df.category.apply(lambda x: x[0])

        df['cat_int'] = df.category.apply(lambda x: cat_dict[x])

        return df


    def get_all_cats(self):
        df = self.df
        return list(df.category.unique())


    def choose_cats(self):
        df = self.df
        if self.subfields == 'all':
            allcats = list(df.category.unique())
        else:
            allcats = self.subfields
        tosscats = []
        catcounts = df.groupby(['year', 'category'])['id'].count()

        for yr in self.years:
            y = str(yr)
            if self.n_percat_peryear == 0:
                smallcats = [cat for cat in list(catcounts[y].index) if
                             catcounts[y, cat] < self.threshold]
            else:
                smallcats = [cat for cat in list(catcounts[y].index) if
                             catcounts[y, cat] < self.n_percat_peryear]
            abscats = [cat for cat in allcats if
                       cat not in list(catcounts[y].index)]
            tosscats.extend(smallcats + abscats)
            if self.verbose:
                print(f'For {y}------------------')
                print(
                    f'{catcounts[y].index[catcounts[y].argmax()]} has the most '
                    f'with {catcounts[y].max()} submissions')
                if self.n_percat_peryear==0:
                    print(
                        f'{smallcats} all have less than {self.threshold} submissions')
                else:
                    print(
                        f'{smallcats} all have less than {self.n_percat_peryear} submissions')
                print(f'{abscats} are absent for this year')

        toss = list(dict.fromkeys(tosscats))
        keepcats = [cat for cat in allcats if cat not in toss]
        print(f'Keeping the following categories: {keepcats}')
        df_cut = df.loc[df['category'].isin(keepcats)]
        if self.single_cat_subs:
            df_cut_sincat = df_cut[
                df_cut.cross_lists.apply(lambda x: len(x) == 0)]
            df_use = df_cut_sincat
        else:
            df_use = df_cut

        df_use_grouped = df_use.groupby(['year', 'category'])
        if self.n_percat_peryear==0:
            smallest_cat = df_use_grouped['id'].count().min()
            print(
                f'taking {smallest_cat} of each category each year to balance data')
            df_sample = df_use_grouped.sample(n=smallest_cat,
                                              random_state=self.randomseed)
        else:
            df_sample = df_use_grouped.sample(n=self.n_percat_peryear,
                                              random_state=self.randomseed)

        return self.clean_text(df_sample)

    def clean_text(self, df):
        df['abstract'] = df.abstract.str.replace('\n', ' ')
        df['abstract'] = df.abstract.str.replace(r'\$(.+?)\$', 'equation',
                                                 regex=True)
        greek = [r'\\alpah', r'\\beta', r'\\gamma', r'\\delta', r'\\epsilon',
                 r'\\zeta', r'\\theta', r'\\eta', r'\\iota', r'\\kappa',
                 r'\\lambda', r'\\mu', r'\\nu', r'\\xi', r'\\omicron', r'\\pi',
                 r'\\rho', r'\\sigma', r'\\tau', r'\\upsilon', r'\\phi',
                 r'\\chi', r'\\psi', r'\\omega']
        Greek = [r'\\Alpah', r'\\Beta', r'\\Gamma', r'\\Delta', r'\\Epsilon',
                 r'\\Zeta', r'\\Theta', r'\\Eta', r'\\Iota', r'\\Kappa',
                 r'\\Lambda', r'\\Mu', r'\\Nu', r'\\Xi', r'\\Omicron', r'\\Pi',
                 r'\\Rho', r'\\Sigma', r'\\Tau', r'\\Upsilon', r'\\Phi',
                 r'\\Chi', r'\\Psi', r'\\Omega']
        for letter in greek + Greek:
            df['abstract'] = df.abstract.str.replace(letter, 'letter',
                                                     regex=True)
        df['abstract'] = df.abstract.str.replace(r'[\|\{\}]', '', regex=True)
        df['abstract'] = df.abstract.str.replace(r'\`\`', r"'", regex=True)
        df['abstract'] = df.abstract.str.replace(r'\'\'', r"'", regex=True)
        df['abstract'] = df.abstract.str.replace(r'[\^\_\+]', '', regex=True)
        df['abstract'] = df.abstract.str.replace(r'\\"o', 'o', regex=True)
        df['abstract'] = df.abstract.str.replace(r'\\"a', 'a', regex=True)
        df['abstract'] = df.abstract.str.replace(r'\\"u', 'u', regex=True)
        df['abstract'] = df.abstract.str.replace(r'\\pm', '', regex=True)
        df['abstract'] = df.abstract.str.replace(r'-->', 'to', regex=True)
        df['abstract'] = df['abstract'].str[2:-1]
        return df

    def write_csv(self):
        yr_string = ''.join([str(x)[-2:] for x in self.years])
        if self.single_cat_subs:
            sincat_string = '_sincat'
        else:
            sincat_string = ''
        place = self.path + self.extra_name + 'arxiv_' + yr_string + sincat_string + '.csv'

        df2write = self.choose_cats()
        print('Writing selected data to:', place)
        df2write.to_csv(place, index=False, header=True)


# ___________________________________________________
def main():
    dat = data(years=[2004,2005],
                 single_cat_subs=False,
                 threshold=500,
                 n_percat_peryear=200,
                 subfields=['hep-th', 'hep-ph', 'hep-ex'],
                 path='../../Data/',
                 extra_name='small_test_',
                 randomseed=666,
                 verbose=True)

    dat.write_csv()



if __name__ == '__main__':
    main()
