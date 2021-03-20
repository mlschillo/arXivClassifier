# Dear Theoretical Physics, if we pass eachother on the street in 10 years, will we even recognize eachother?

As I bid a fond farewell to academia, and an enthusiastic 'Hello' to *world*, I face the same questions regarding my identity that so many have faced before me: 'Will people still think I'm smart?' 'What will I do? will it be fascinating, challenging, and rewarding?' 'Will I still be a physicist?' In this project, I would like to focus on the question of my continuation to be a physicist.  **The idea, as the title suggests, is to ask whether 10 years of progress will render theoretical physics unrecognizable.**  I am currently an expert in theoretical cosmology and string phenomenology, and very-well versed in high energy theoretical phsyics in general, but for how long?

To answer this question, I will train a new expert, not just in high energy theory, but in all subfields of physics. At least, **I will train an expert AI to recognize and classify the abstract of a physics paper into its appropriate subfield.** Then, I will train this AI specifically using physics papers published to arxiv.org in, say, 2008, and see how well this AI performs on classifying papers published in 2018.

Other than answering questions pertaining to my personal identity crisis, this type of analysis might be of interest to the history of physics in **quantifiably identifying paradigm shifts** -- pivotal points where the face of a field shifts so as to render it unrecognizable to expers of the past.  Another possible use is to help extremely indecisive authors who simply cannot decide which arXiv category to submit to; just download the model, feed it your abstract, and let the expert decide where your submission fits best.

## Method

I use the pretrained bert-base-cased model provided by Google and the Transformers library provided by Hugging Face. The architecture as well as intuitive descriptions of how all the pieces work are provided by LINKTUTORIAL. I train the models on Google Colab's GPUs.  So overall, all of the heavy lifting is done for me. Thanks!

All that remains for me to do is clean and separate the data into the various training sets, fine tune the various models -- choosing hyperparameters such as learning rate, batch size, and training epochs -- and collect and analyze the results.

## Data

The arXiv has revolutionized the way we do physics. All physics papers are put onto the arXiv and everyday physicists browse their particular subfield like a newsfeed, checking out the latest developments.  All papers are submitted to at least one subfield, but may be cross-listed to other subfields.  

In cleaning the data I:
- removed sub-sub-fields, e.g. astro-ph.co (cosmology and nongalactic astrophysics) and astro-ph.ga (astrophysics of galaxies) are lumped together in astro-ph.
- Threw out a few subfields that did not consistantly have enough submissions/year: High Energy Physics-Lattice, Nuclear Experiment, Statistics,Quantitative Finance, Electrical Engineering and Systems Science, and Economics
- Clean up newline and other special characters and replaced all LaTeX mathematical equations with the word 'equation'.

The data was then split by year with each year begin balanced w.r.t. the categories.

## Training

I trained two models: the "early" and the "late" model.  The **early model was trained using a dataset of papers from 2008, 2009, and 2010**. The **late model was trained using papers from 2018, 2019 and 2020**.  Each dataset was balanced across categories and contained 21,000 abstracts which were then divided using a 90/5/5 split for training/validation/test sets (retaining balance across categories.)

To prevent overfitting I used a dropout rate of .5, and also used an early stopping condition if the validation loss increased by 30% from its initial value.  

## Results

I evaluate the models on the test set separated out from the training data, and also on equally sized and balanced datasets comprised of papers from each of the intervening years: 2011-2017.  The results bear out the naive expectation, the early model is better at classifying papers in the beginning of the intervening years, and the late model is better towards the end, with a cross-over at 2013.  The models are "pretty good" with the early model getting 74.6% correct on the early papers and the late model getting 77.1% correct on the late papers.  However, for the 14 categories we are classifying, this is over 10 times better than random.  Also, these are highly technical texts with a lot of jargon; I would be interested to know my own accuracy... but I don't have time to read all those abstract right now.

One subtelty is the fact that some papers are only submitted to one subfield, whereas many papers submitted to a main subfield and then cross-listed to several others.  These latter papers may straddle sub-fields and make classifying difficult, where as single-category submissions should be more representative of their category and there fore easier to classify. Indeed, accuracy is substantially higer for single category submissions.

<img src="./Pix/accuracy.png" width="524" height="324">

A clear extension of this project is to make a multi-lable classifier to pick up the cross-listed categories. This would be a simple matter of adding some labels and removing the softmax function.  However, for this project I am more interested in looking at the evolution of the subfields and therefore looking at single category submissions is a better indicator.  

To look at the evolution of subfields I evaluated the models on a test set containing only single-category submissions.  The accuracy of the early model per category is shown below. 

<img src="./Pix/subfield_accuracy.png" width="524" height="524">

A subfield staying flat may be interpreted as stagnation while accuracy decreasing rapidly may indicate progess. The odd case of accuracy increasing perhaps indicates a major even happening directly before the early model years and coming to dominate the field more and more.  
