# Datasets

Datasets are defined as .jsonls. Each line in the .jsonl is an entry of the dataset. It needs to minimally contain the following fields:
- `string`: the text of the entry that is used to in the string field.

Additional fields—such as `target`—can also be included and will be written into the experiment output.

The .jsonl should be pointed to from the config file in `conf/task`. This might also contain a prompt that should go with the dataset if the dataset itself requires a wrapper prompt.

Ideally, the dataset is split here into a train and validation part to prevent cross-contamination of the training and validation set down the pipeline. Use `scripts/split_jsonl_into_train_test.py` to do this.

By default, datasets are kept with a precomputed test/validation split computed at creation to be really sure that we don't have leakage.
The naming scheme is `all_<dataset_name>.jsonls` for the full dataset and `train_<dataset_name>.jsonls` and `val_<dataset_name>.jsonls` for the train and validation set, respectively.
Following this scheme is necessary for the default configuration files to work, where a `set` field contains either `all`, `train` or `val` to indicate which part of the dataset is used.

## Datasets
### Dear Abbie
https://data.world/the-pudding/dear-abby
> About:
>
> This dataset contains all of the data used in The Pudding essay 30 Years of American Anxieties: What 20,000 letters to an advice columnist tell us about what—and who—concerns us most. published in November 2018.
>
> These data will not be updated.
>
> Source(s) & Methods:
>
> This includes questions publicly available on https://www.uexpress.com/, as well as a number of questions from the mid-1980s, obtained through digital copies of newspapers which included Dear Abby questions.
>
> At the beginning of the project, we had collected and OCR’d data ranging from the 1950s through to 2017. Earlier articles, available through JSTOR, proved to be difficult to convert to structured text, so we omitted them from analysis. Consequently, our corpus of questions spanned between the mid-1980s and 2017.
>
> The writers of these questions likely skew roughly ⅔ female (according to Pauline Phillips, who mentions the demographics of responses to a survey she disseminated in 1987), and consequently, their interests are overrepresented; we’ve been unable to find other demographic data surrounding their origins. There is, doubtless, a level of editorializing here: only a fraction of the questions that people have written in have seen publication, because agony aunts (the writers of advice columns) must selectively filter what gets published. Nevertheless, the concerns of the day seem to be represented, such as the HIV/AIDS crisis in the 1980s. Additionally, we believe that the large sample of questions in our corpus (20,000+) that have appeared over recent decades gives a sufficient directional sense of broad trends.
>
> After initially exploring the corpus, we began to identify a number of common themes which Dear Abby’s readers frequently brought up, and decided to focus on three: sex, LGBTQ issues, and religion. For each relevant issue, we created a list of relevant keywords for each issue and used those to first create a broad grouping of question, before breaking them down into categories.
>
> For the final section, we searched for questions dealing with parents, children, friends, and bosses of Dear Abby’s readers. We then employed a technique to cluster these posts visually, called t-SNE, and proceeded to manually create categories using both these visual groupings, as well as a manual classification process consisting of tagging all relevant entries.
>
> Caveats:
>
> Last Modified: 11/28/2018.
> Contact Information: Ilia Blinderman
> Temporal Applicability: 1985-Fall of 2017
> License
>
> Data available under the MIT License.

### personal_preferences
Asking the model about its mundane preferences ("What is your favorite color")

### self_referential
Asking the model single word questions about itself ("What is your name?")

### writing_stories
Asking the model to write a story with a given prompt. The first 40 were written by me, the rest by Claude 3 Opus.
