## Red Company Classifier
Red companies are mainly companies that are considered as "risky" in terms of any sort of business deal. It can be from any industry and any business. Mostly it is essential in the B2B market space.

The `data` folder contains the following files: 
* __train_ids.parquet__ : You can use this company list for model training purpose.
* __test_ids.parquet__ : You **must** to use this company list to test/evaluate your model.

In these two files, `mid` represents a unique identifier of a company. `target_label` referes if a company is a red company(`1`) or not (`0`).

You have access to the following 3 different data sources:
* __data_source_1.parquet__ : This file contains the companies' demographic  information along with some usages.
* __data_source_2.parquet__ : This file contains the companies' firmographic data.
* __data_source_3.parquet__ : This file contains some kind of survey information(categorical data that is flattened).




Your task is to submit code that allows us to determine whether a given company is a red company or not. Here are some expectations: 


We have defined a helper `red_company_classifier.py` file that would help you. It has some boiler plate defined but you need to fill out yourself and write some extra functions or two.

If you use extra libraries, please amend the provided `requirements.txt` file and this readme with instructions. Once everything is ready, we could use the tool by running:

`python red_company_classifier.py --in-folder <path-to-data> --out-folder <path-to-model-destination>` , where
-  `<path-to-data>` corresponds to the data containing the data files
- `<path-to-model-destination>` corresponds to a folder where the trained model will be serialised to.


Your code in `red_company_classifier.py` should:
* generate a training / eval / test split (and do any necessary data pre-processing)
* train a model
* print evaluation metrics
* save it to a destination

Once you are done, __submit a pull request__ for evaluation.

## How to :
1. Create virtual environment, activate it and Run `python3.10 -m pip install -r requirements.txt`
2. Create a `./model` folder, here our joblib model file will be generated
3. RUn `python3 red_company_classifier.py --in-folder ./data --out-folder ./model`

## PoC Link : [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aUfmE3gAYzSN9oa-6zHKg5eO8w71HqE2?usp=sharing)
## Implementation Steps :
![mermaid-diagram-2024-11-12-213736](https://github.com/user-attachments/assets/98a2828b-3ec1-461e-b698-3008124073a7)

