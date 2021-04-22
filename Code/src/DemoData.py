import DataSelection

def main():
    """Make two model training sets from 1999 and 2019, then make two test sets
    from 2005 and 2010 where the test test are sized equally to the test set that
    will be taken out of the model training sets.
    """
    for year in [1999, 2019]:
        dat = DataSelection.data(years=year,
                   single_cat_subs=True,
                   threshold=1000,
                   n_percat_peryear=800,
                   subfields=['astro-ph', 'hep-th', 'gr-qc'],
                   path='../../Data/',
                   randomseed=666,
                   verbose=True)

        dat.write_csv()

    for year in [2005, 2010]:
        dat = DataSelection.data(years=year,
                   single_cat_subs=True,
                   threshold=40,
                   n_percat_peryear=40,
                   subfields=['astro-ph', 'hep-th', 'gr-qc'],
                   path='../../Data/',
                   randomseed=666,
                   verbose=True)

        dat.write_csv()

if __name__ == '__main__':
    main()